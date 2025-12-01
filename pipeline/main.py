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
from joblib import load, dump, Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
# Clean imports using __init__.py aggregators
from data_classes import Experiment, Timepoint, Video, ROI, Neuron, Spike

# Import from individual modules to avoid circular import

from pipeline.io_handlers import save_video_summary, save_timepoint_summary, save_filtered_suite2p
from utils.io_utils import load_experiment_structure
from utils.cascade_utils import load_cascade_model

logger = logging.getLogger(__name__)
def load_models(config: Dict) -> Dict:
    """Load all required models and normalize wrappers to sklearn estimators."""
    models: Dict = {}

    models['roi_classifier'] = None
    roi_path = Path(config['models'].get('roi_model_path', ''))
    if roi_path.exists():
        models['roi_classifier'] = load(roi_path)
        print(f"Loaded ROI classifier from {roi_path}")
    else:
        raise RuntimeError(f"ROI classifier not found at {roi_path}")

    models['spike_classifier'] = None
    spike_path = Path(config['models'].get('spike_model_path', ''))
    if spike_path.exists():
        models['spike_classifier'] = load(spike_path)
        print(f"Loaded Spike classifier from {spike_path}")
    else:
        raise RuntimeError(f"Spike classifier not found at {spike_path}")
    
    model_name = config['models']['cascade_model_name']
    model_dir = config['models']['cascade_model_dir']
    try: 
        models['cascade'] = load_cascade_model(model_name=model_name, model_dir=model_dir)
        logger.info("Loaded Cascade model")
    except Exception as e:
        logger.warning(f"Failed to load Cascade model from {model_dir} with name {model_name}: {e}")
        print(f"Failed to load Cascade model from {model_dir} with name {model_name}: {e}")

    return models


def get_savgol_params(fs, sensor_type='gcamp6s') -> Tuple[int, int]:
    """Get Savitzky-Golay parameters based on sampling frequency and sensor.
        Window size depends on kinetics of sensor type."""
    if sensor_type == 'gcamp8s':
        window_frames = int(0.6 * fs)  # 600ms window
    elif sensor_type == 'gcamp6f':
        window_frames = int(0.3 * fs)  # 300ms window
    else:
        window_frames = int(0.4 * fs)  # 400ms window
    
    window_length = 2 * (window_frames // 2) + 1
    window_length = max(9, window_length)  # Minimum 9 for slow sensors
    
    polyorder = 3  # Cubic fit better for slow, smooth transients
    
    return window_length, polyorder

def process_video_explicit(video: Video, models: Dict, config: Dict) -> Optional[Dict]:
    """
    Process a single video through all pipeline steps explicitly.
    
    Each step is clearly visible for debugging.
    Includes bad ROI tracking and filtered Suite2p saving.
    """
    video_path = video.path
    suite2p_path = video.suite2p_path
    logger.info(f"Processing video: {video_path.name}")
    results = {'video_path': video_path}
    
    current_video = video
    cascade_prob, norm_sm_f, norm_sg_f, sm_sp = current_video.process_fluorescence_traces()
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
    all_rois : List[ROI] = []
    for i in range(n_rois):
        roi = ROI(
            index=i,
            f_trace=suite2p_data['F'][i],
            stats=suite2p_data['stat'][i] if 'stat' in suite2p_data else None,
            fneu=suite2p_data['Fneu'][i] if 'Fneu' in suite2p_data else None,
            
        )
        all_rois.append(roi)
    good_rois, bad_rois, good_roi_mask = current_video.filter_rois(all_rois, models['roi_classifier'])
    bad_roi_indices = [roi.index for roi in bad_rois]
    bad_roi_features = [roi.features for roi in bad_rois]

    logger.info(f"    {len(good_rois)}/{len(all_rois)} ROIs passed filtering")
    logger.info(f"    Bad ROI indices: {bad_roi_indices[:10]}..." if len(bad_roi_indices) > 10 else f"    Bad ROI indices: {bad_roi_indices}")
    results['all_rois'] = all_rois
    results['good_rois'] = good_rois
    results['bad_roi_indices'] = bad_roi_indices
    results['bad_roi_features'] = bad_roi_features
    results['good_roi_mask'] = good_roi_mask

    filtered_suite2p_path = save_filtered_suite2p(
        video_path=video_path,
        good_roi_mask=good_roi_mask,
        suite2p_data=suite2p_data,
        cascade_prob=cascade_prob
    )
    logger.info(f"    Saved filtered Suite2p to {filtered_suite2p_path}")
    results['filtered_suite2p_path'] = filtered_suite2p_path
    
    if len(good_rois) == 0:
        logger.warning("    No good ROIs found")
        return results

    current_video.neurons = [Neuron(roi, i, fs=suite2p_data['fs']) for i, roi in enumerate(good_rois)]
    results['neurons'] = current_video.neurons

    logger.info("  Step 6: Extracting spike features in parallel...")
    spk_feats_df, spike_mask = current_video.get_all_spike_features(models['spike_classifier'])
    current_video.filter_all_spikes(spike_mask)
    logger.info(f"    {spike_mask.sum()}/{len(spike_mask)} spikes passed filtering")
    inst_spikes = Parallel(n_jobs=-1)(
        delayed(n.instantiate_spikes)(current_video.norm_sm_f[n.index, :], current_video.norm_sg_f[n.index, :]) for n in current_video.neurons
    )
    logger.info(f"    {len(current_video.neurons)} neurons have valid spikes")
    
    results['filtered_neurons'] = current_video.neurons
    results['spike_summaries_per_neuron'] = current_video.get_spike_statistics()
    if len(current_video.neurons) > 1:
        logger.info("  Step 8: Grouping neurons...")
        logger.info("    Computing STTC groups...")
        grouping_stats = current_video.get_group_summary(config)
        results['grouping_stats'] = grouping_stats
        results['sttc_groups'] = current_video.sttc_groups
        results['sttc_matrix'] = current_video.sttc_matrix
        results['dtw_groups'] = current_video.dtw_groups if current_video.dtw_matrix is not None else []
        results['dtw_matrix'] = current_video.dtw_matrix
    else:
        logger.info("  Step 8: Skipping grouping (need 2+ neurons)")
        results['sttc_groups'] = []
        results['dtw_groups'] = []
    
    # Step 9: Generate summary with bad ROI tracking
    logger.info("  Step 9: Generating summary...")
    summary = save_video_summary(results, video_path / 'metrics')
    results['summary'] = summary
    
    return results

def process_timepoint(timepoint: Timepoint, models: Dict, config: Dict) -> Timepoint:
    """Process all videos in a timepoint folder."""

    for video in timepoint.videos:
        results = process_video_explicit(video, models, config)
        if results and results.get('filtered_neurons'):
            video.neurons = results['filtered_neurons']
            video.sttc_groups = results.get('sttc_groups', [])
            video.dtw_groups = results.get('dtw_groups', [])
    
    return timepoint
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GCaMP Analysis Pipeline')
    parser.add_argument('--config', type=Path, default = "config\\pipeline_config.yaml", help='Config file path')
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
    
    if args.video:
        results = process_video_explicit(args.video, models, config)
        if results:
            logger.info(f"Video complete: {len(results.get('filtered_neurons', []))} neurons")
    
    elif args.timepoint:
        timepoint = process_timepoint(args.timepoint, models, config)
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
                timepoint = process_timepoint(timepoint, models, config)
                # Note: timepoint is already in experiment.timepoints from load_experiment_structure
                from pipeline.io_handlers import save_timepoint_summary_by_video
                tp_output_path = treatment_dir / f'{timepoint.name}_video_summary.xlsx'
                save_timepoint_summary_by_video(timepoint, tp_output_path)
            
            # Save experiment summary
            if experiment.timepoints:
                from pipeline.io_handlers import save_experiment_summary
                output_path = treatment_dir / 'experiment_summary.xlsx'
                save_experiment_summary(experiment, output_path)
                logger.info(f"Saved experiment summary to {output_path}")
                
    else:
        logger.error("Must specify --video, --timepoint, or --experiment")

if __name__ == '__main__':
    main()