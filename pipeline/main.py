"""Main pipeline with explicit steps and both DTW/STTC grouping."""
import itertools
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import logging
from typing import Dict, List, Optional, Tuple
import yaml
import numpy as np
import pandas as pd
from joblib import load, Parallel, delayed
import time

# Clean imports using __init__.py aggregators
from data_classes import Experiment, Timepoint, Video, ROI, Neuron, Spike

# Import from individual modules to avoid circular import

from pipeline.io_handlers import save_video_summary, save_timepoint_summary, save_filtered_suite2p
from utils.io_utils import load_experiment_structure, load_models
from utils.cascade_utils import load_cascade_model

logger = logging.getLogger(__name__)


class PipelineStats:
    """Track pipeline execution statistics."""
    
    def __init__(self):
        self.start_time = time.perf_counter()
        self.videos_processed = 0
        self.videos_failed = 0
        self.total_rois = 0
        self.good_rois = 0
        self.total_neurons = 0
        self.total_spikes = 0
        self.timepoints_processed = 0
        self.video_times = []
    
    def add_video(self, results: Optional[Dict]):
        """Record stats from a processed video."""
        if results is None:
            self.videos_failed += 1
            return
        
        self.videos_processed += 1
        self.total_rois += len(results.get('all_rois', []))
        self.good_rois += len(results.get('good_rois', []))
        
        neurons = results.get('filtered_neurons', [])
        self.total_neurons += len(neurons)
        
        for neuron in neurons:
            if hasattr(neuron, 'spikes') and neuron.spikes:
                self.total_spikes += len(neuron.spikes)
        
        if 'timings' in results:
            total_time = sum(results['timings'].values())
            self.video_times.append(total_time)
    
    def add_timepoint(self):
        """Record a processed timepoint."""
        self.timepoints_processed += 1
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        elapsed = time.perf_counter() - self.start_time
        
        return {
            'elapsed_time': elapsed,
            'videos_processed': self.videos_processed,
            'videos_failed': self.videos_failed,
            'total_rois': self.total_rois,
            'good_rois': self.good_rois,
            'roi_pass_rate': self.good_rois / self.total_rois if self.total_rois > 0 else 0,
            'total_neurons': self.total_neurons,
            'total_spikes': self.total_spikes,
            'timepoints_processed': self.timepoints_processed,
            'avg_time_per_video': np.mean(self.video_times) if self.video_times else 0,
            'videos_per_minute': (self.videos_processed / elapsed) * 60 if elapsed > 0 else 0,
            'rois_per_second': self.total_rois / elapsed if elapsed > 0 else 0,
        }
    
    def print_summary(self):
        """Print formatted summary to logger."""
        stats = self.get_summary()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total time:           {stats['elapsed_time']:.2f}s ({stats['elapsed_time']/60:.1f} min)")
        logger.info(f"  Videos processed:     {stats['videos_processed']} ({stats['videos_failed']} failed)")
        logger.info(f"  Timepoints processed: {stats['timepoints_processed']}")
        logger.info("-" * 60)
        logger.info(f"  Total ROIs:           {stats['total_rois']}")
        logger.info(f"  Good ROIs:            {stats['good_rois']} ({stats['roi_pass_rate']:.1%} pass rate)")
        logger.info(f"  Total neurons:        {stats['total_neurons']}")
        logger.info(f"  Total spikes:         {stats['total_spikes']}")
        logger.info("-" * 60)
        logger.info(f"  Avg time per video:   {stats['avg_time_per_video']:.2f}s")
        logger.info(f"  Throughput:           {stats['videos_per_minute']:.1f} videos/min")
        logger.info(f"                        {stats['rois_per_second']:.1f} ROIs/sec")
        logger.info("=" * 60)
        logger.info("")


def process_video_explicit(video: Video, models: Dict, config: Dict) -> Optional[Dict]:
    """
    Process a single video through all pipeline steps explicitly.
    """
    video_path = video.path
    suite2p_path = video.suite2p_path
    logger.info(f"Processing video: {video_path.name}")
    results = {'video_path': video_path}
    timings = {}
    
    current_video = video
    
    # Step 1-2: Cascade inference + smoothing
    t0 = time.perf_counter()
    cascade_prob, norm_sm_f, norm_sg_f, sm_sp = current_video.process_fluorescence_traces()
    timings['cascade_and_smoothing'] = time.perf_counter() - t0
    logger.info(f"    Cascade + smoothing: {timings['cascade_and_smoothing']:.2f}s")
    
    current_video.norm_sm_f = norm_sm_f
    current_video.norm_sg_f = norm_sg_f
    current_video.sm_sp = sm_sp
    suite2p_data = current_video.suite2p_data
    n_rois, n_frames = suite2p_data['F'].shape
    logger.info(f"    Loaded {n_rois} ROIs, {n_frames} frames")
    results['suite2p_data'] = suite2p_data
    results['n_frames'] = n_frames
    results['cascade_prob'] = cascade_prob
    
    # Step 3: Create ROI objects
    logger.info("  Step 3: Creating ROI objects...")
    t0 = time.perf_counter()
    all_rois : List[ROI] = []
    for i in range(n_rois):
        roi = ROI(
            index=i,
            f_trace=suite2p_data['F'][i],
            stats=suite2p_data['stat'][i] if 'stat' in suite2p_data else None,
            fneu=suite2p_data['Fneu'][i] if 'Fneu' in suite2p_data else None,
        )
        all_rois.append(roi)
    timings['create_rois'] = time.perf_counter() - t0
    logger.info(f"    Create ROIs: {timings['create_rois']:.2f}s")
    
    # Step 4: Filter ROIs
    t0 = time.perf_counter()
    good_rois, bad_rois, good_roi_mask = current_video.filter_rois(all_rois, models['roi_classifier'])
    timings['filter_rois'] = time.perf_counter() - t0
    logger.info(f"    Filter ROIs: {timings['filter_rois']:.2f}s")
    
    bad_roi_indices = [roi.index for roi in bad_rois]
    bad_roi_features = [roi.features for roi in bad_rois]

    logger.info(f"    {len(good_rois)}/{len(all_rois)} ROIs passed filtering")
    results['all_rois'] = all_rois
    results['good_rois'] = good_rois
    results['bad_roi_indices'] = bad_roi_indices
    results['bad_roi_features'] = bad_roi_features
    results['good_roi_mask'] = good_roi_mask

    t0 = time.perf_counter()
    filtered_suite2p_path = save_filtered_suite2p(
        video_path=video_path,
        good_roi_mask=good_roi_mask,
        suite2p_data=suite2p_data,
        cascade_prob=cascade_prob
    )
    timings['save_filtered_suite2p'] = time.perf_counter() - t0
    logger.info(f"    Save filtered Suite2p: {timings['save_filtered_suite2p']:.2f}s")
    results['filtered_suite2p_path'] = filtered_suite2p_path
    
    if len(good_rois) == 0:
        logger.warning("    No good ROIs found")
        results['timings'] = timings
        return results

    current_video.neurons = [Neuron(roi, i, fs=suite2p_data['fs']) for i, roi in enumerate(good_rois)]
    results['neurons'] = current_video.neurons

    # Step 5: Spike feature extraction
    t0 = time.perf_counter()
    spk_feats_df, spike_mask = current_video.get_all_spike_features(models['spike_classifier'])
    timings['spike_features'] = time.perf_counter() - t0
    logger.info(f"    Spike features: {timings['spike_features']:.2f}s")
    
    current_video.filter_all_spikes(spike_mask)
    logger.info(f"    {spike_mask.sum()}/{len(spike_mask)} spikes passed filtering")
    
    # Step 6: Instantiate spikes
    t0 = time.perf_counter()
    inst_spikes = Parallel(n_jobs=-1)(
        delayed(n.instantiate_spikes)(current_video.norm_sm_f[n.index, :], current_video.norm_sg_f[n.index, :]) 
        for n in current_video.neurons
    )
    timings['instantiate_spikes'] = time.perf_counter() - t0
    logger.info(f"    Instantiate spikes: {timings['instantiate_spikes']:.2f}s")

    logger.info(f"    {len(current_video.neurons)} neurons have valid spikes")
    results['filtered_neurons'] = current_video.neurons
    
    # Step 7: Spike statistics
    t0 = time.perf_counter()
    results['spike_summaries_per_neuron'] = current_video.get_spike_statistics()
    timings['spike_statistics'] = time.perf_counter() - t0
    logger.info(f"    Spike statistics: {timings['spike_statistics']:.2f}s")
    
    # Step 8: Grouping
    if len(current_video.neurons) > 1:
        t0 = time.perf_counter()
        # Get sweep parameters from config, or use defaults
        grouping_config = config.get('grouping', {})
        sttc_config = grouping_config.get('sttc', {})
        dtw_config = grouping_config.get('dtw', {})
        time_window = sttc_config.get('time_window', 0.4)
        distance_threshold = sttc_config.get('distance_threshold', 0.2)
        linkage_method = sttc_config.get('linkage_method', 'average')
        min_group_size = sttc_config.get('min_group_size', 2)
        
        logger.info(f"  Step 8: Grouping parameter sweep...")
        logger.info(f"    Time window: {time_window}")
        logger.info(f"    Distance threshold: {distance_threshold}")
        
        all_combined_stats = []  # Collect combined_stats from all configs
        n_sttc_groups = []
        

        
        grouping_stats, sttc_groups = current_video.get_group_summary(time_window, distance_threshold)
        
        # Append this config's combined_stats (already has t_win, sttc_thresh, etc.)
        all_combined_stats.extend(grouping_stats.get('combined_stats', []))
        n_sttc_groups.append(grouping_stats.get('n_sttc_groups', 0))
        logger.info(f"      tw={time_window:.3f}, dt={distance_threshold:.2f} -> "
                    f"{grouping_stats.get('n_sttc_groups', 0)} STTC groups")
    
        timings['grouping'] = time.perf_counter() - t0
        logger.info(f"    Grouping sweep complete: {timings['grouping']:.2f}s")
        results['grouping_stats'] =  {"combined_stats": all_combined_stats}  # Flat list of all group stats across configs
        results['sttc_matrix'] = current_video.sttc_matrix
        results['sttc_groups'] = current_video.sttc_groups
        results['dtw_groups'] = current_video.dtw_groups if current_video.dtw_matrix is not None else []
        results['dtw_matrix'] = current_video.dtw_matrix
        stat = suite2p_data.get('stat', None)
        ops = suite2p_data.get('ops', {})
        img_size = (ops.get('Ly', 1024), ops.get('Lx', 1024))
        from pipeline.io_handlers import visualize_neuron_groups
        visualize_neuron_groups(
            neuron_groups=current_video.sttc_groups,
            stat=stat,
            img_size=img_size,
            video_path=video_path,
            config_label='sttc_grouping'
        )
    else:
        logger.info("  Step 8: Skipping grouping (need 2+ neurons)")
        results['grouping_stats'] = {"combined_stats": []}  # Empty but valid structure
        results['sttc_groups'] = []
        results['sttc_matrix'] = None
        results['dtw_groups'] = []
        results['dtw_matrix'] = None
    
    # Step 9: Generate summary
    logger.info("  Step 9: Generating summary...")
    t0 = time.perf_counter()
    summary = save_video_summary(results, video_path / 'metrics')
    timings['save_summary'] = time.perf_counter() - t0
    logger.info(f"    Save summary: {timings['save_summary']:.2f}s")
    
    results['summary'] = summary
    results['timings'] = timings
    
    # Print timing summary
    logger.info("  === Timing Summary ===")
    for step, duration in sorted(timings.items(), key=lambda x: -x[1]):
        logger.info(f"    {step}: {duration:.2f}s")
    
    return results


def process_timepoint(timepoint: Timepoint, models: Dict, config: Dict, stats: Optional[PipelineStats] = None) -> Timepoint:
    """Process all videos in a timepoint folder."""

    for video in timepoint.videos:
        results = process_video_explicit(video, models, config)
        
        if stats:
            stats.add_video(results)
        
        if results and results.get('filtered_neurons'):
            video.neurons = results['filtered_neurons']
            video.sttc_groups = results.get('sttc_groups', [])
            video.dtw_groups = results.get('dtw_groups', [])
    
    if stats:
        stats.add_timepoint()
    
    return timepoint


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GCaMP Analysis Pipeline')
    parser.add_argument('--config', type=Path, default="config\\pipeline_config.yaml", help='Config file path')
    parser.add_argument('--video', type=Path, help='Process single video')
    parser.add_argument('--timepoint', type=Path, help='Process timepoint folder')
    parser.add_argument('--experiment', type=Path, help='Process full experiment')
    parser.add_argument('--debug', action='store_true', help='Debug logging')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    models = load_models(config)
    
    # Initialize pipeline stats
    stats = PipelineStats()
    
    if args.video:
        video = Video(args.video)
        results = process_video_explicit(video, models, config)
        stats.add_video(results)
        
        if results:
            logger.info(f"Video complete: {len(results.get('filtered_neurons', []))} neurons")
    
    elif args.timepoint:
        timepoint = Timepoint(args.timepoint)
        timepoint = process_timepoint(timepoint, models, config, stats)
        logger.info(f"Timepoint complete: {len(timepoint.videos)} videos processed")
        save_timepoint_summary(timepoint, args.timepoint / 'summary.xlsx')
            
    elif args.experiment:
        experiment_path = Path(args.experiment)
        treatment_folders = [d for d in experiment_path.iterdir() if d.is_dir()]
        
        if not treatment_folders:
            logger.error(f"No treatment folders found in {experiment_path}")
            return

        for treatment_dir in treatment_folders:
            logger.info(f"Processing treatment: {treatment_dir.name}")
            experiment = load_experiment_structure(treatment_dir)
            
            # Process each timepoint and video
            for timepoint in experiment.timepoints:
                timepoint = process_timepoint(timepoint, models, config, stats)
                from pipeline.io_handlers import save_timepoint_summary_by_video
                tp_output_path = treatment_dir / f'{timepoint.treatment}_{timepoint.name}_video_summary.xlsx'
                save_timepoint_summary_by_video(timepoint, tp_output_path)
            
            # Save experiment summary
            if experiment.timepoints:
                from pipeline.io_handlers import save_experiment_summary
                output_path = treatment_dir / 'experiment_summary.xlsx'
                save_experiment_summary(experiment, output_path)
                logger.info(f"Saved experiment summary to {output_path}")
                
    else:
        logger.error("Must specify --video, --timepoint, or --experiment")
        return
    
    # Print final pipeline summary
    stats.print_summary()


if __name__ == '__main__':
    main()