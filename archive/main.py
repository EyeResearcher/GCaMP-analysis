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
from data_classes.video import VideoStatistics, VideoStatisticsWriter

# Import from individual modules to avoid circular import

from pipeline.io_handlers import save_video_summary, save_timepoint_summary, save_filtered_suite2p
from utils.io_utils import load_experiment_structure, load_models
from archive.cascade_utils import load_cascade_model

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
    v_path = video.path
    logger.info(f"Processing video: {v_path.name}")
    results = {'video_path': v_path}
    
    
    # Step 1-2: Cascade inference + smoothing

    norm_sm_f, norm_sg_f = video.process_fluorescence_traces()
    
    suite2p_data = video.suite2p_data
    
    # Step 3: Create ROI objects
    logger.info("  Step 3: Creating ROI objects...")

    all_rois = video._create_roi_objects()

    
    
    # Step 4: Filter ROIs
   
    good_rois, good_roi_mask = video.filter_rois(all_rois, models['roi_classifier'])

    video.get_bad_rois_features_df()

    logger.info(f"    {len(good_rois)}/{len(all_rois)} ROIs passed filtering")

    filtered_suite2p_path = save_filtered_suite2p(
        video_path=v_path,
        good_roi_mask=good_roi_mask,
        suite2p_data=suite2p_data,
    )

    if len(good_rois) == 0:
        logger.warning("    No good ROIs found")
        return results

    video._create_neuron_objects(good_rois)
    
    # Step 5: Spike feature extraction

    spk_feats_df = video._extract_spike_features_parallel()
    _, spike_mask = video.filter_all_spikes(spk_feats_df, models['spike_classifier'])


    logger.info(f"    {spike_mask.sum()}/{len(spike_mask)} spikes passed filtering")
    
    # Step 6: Get Spike Statistics per Neuron

    logger.info(f"    {len(video.neurons)} neurons have valid spikes")
    
    video.get_spike_statistics()

    
    # Step 8: Grouping
    if len(video.neurons) > 1:
        grouping_config = config.get('grouping', {})
        sttc_config = grouping_config.get('sttc', {})
        dtw_config = grouping_config.get('dtw', {})
        time_window = sttc_config.get('time_window', 0.4)
        distance_threshold = sttc_config.get('distance_threshold', 0.2)
        
        video.get_group_summary(time_window, distance_threshold)
    
        video._visualize_neuron_groups(config_label='sttc_grouping')
    else:
        logger.info("  Step 8: Skipping grouping (need 2+ neurons)")
        

    stats = VideoStatistics.from_video(video)
    writer = VideoStatisticsWriter()
    writer.write(stats, v_path.parent)

    
    

    
    return stats


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