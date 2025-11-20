"""Main pipeline with explicit steps and both DTW/STTC grouping."""
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import logging
from typing import Dict, List, Optional
import yaml
import numpy as np
import pandas as pd
from joblib import load

# Clean imports using __init__.py aggregators
from data_classes import Experiment, Timepoint, Video, ROI, Neuron, Spike

# Import from individual modules to avoid circular import
from pipeline.preprocessing import load_suite2p_data, compute_cascade_probabilities
from pipeline.roi_processing import extract_roi_features, filter_rois
from pipeline.spike_detection import detect_spikes_from_cascade
from pipeline.spike_filtering import extract_spike_features, filter_spikes
from pipeline.neuron_grouping import group_neurons_by_sttc, group_neurons_by_dtw, compare_groupings
from pipeline.io_handlers import save_video_summary, save_timepoint_summary, save_filtered_suite2p

logger = logging.getLogger(__name__)

def load_models(config: Dict) -> Dict:
    """Load all required models."""
    models = {}
    
    # ROI classifier (optional - for filtering bad ROIs)
    roi_path = Path(config['models']['roi_model_path'])
    if roi_path.exists():
        try:
            models['roi_classifier'] = load(roi_path)
            logger.info(f"Loaded ROI classifier from {roi_path}")
        except (EOFError, Exception) as e:
            models['roi_classifier'] = None
            logger.warning(f"Failed to load ROI classifier from {roi_path}: {e}")
            logger.warning("Continuing without ROI classifier - all ROIs will be kept")
    else:
        models['roi_classifier'] = None
        logger.warning(f"ROI classifier not found at {roi_path}, continuing without it")
        
    # Spike classifier (optional)
    spike_path = config['models'].get('spike_model_path')
    if spike_path and Path(spike_path).exists():
        try:
            models['spike_classifier'] = load(spike_path)
            logger.info(f"Loaded spike classifier from {spike_path}")
        except (EOFError, Exception) as e:
            models['spike_classifier'] = None
            logger.warning(f"Failed to load spike classifier from {spike_path}: {e}")
            logger.warning("Continuing without spike classifier - all spikes will be kept")
    else:
        models['spike_classifier'] = None
        logger.warning("No spike classifier, will keep all detected spikes")
        
    # Cascade model - using utils module
    from utils import load_cascade_model
    models['cascade'] = load_cascade_model(
        model_name=config['models']['cascade_model_name'],
        model_dir=config['models']['cascade_model_dir']
    )
    logger.info("Loaded Cascade model")
    
    return models

def process_video_explicit(video_path: Path, models: Dict, config: Dict) -> Optional[Dict]:
    """
    Process a single video through all pipeline steps explicitly.
    
    Each step is clearly visible for debugging.
    Includes bad ROI tracking and filtered Suite2p saving.
    """
    logger.info(f"Processing video: {video_path.name}")
    results = {'video_path': video_path}
    
    # Step 1: Load Suite2p data
    logger.info("  Step 1: Loading Suite2p data...")
    suite2p_path = video_path / 'suite2p' / 'plane0'
    if not (suite2p_path / 'F.npy').exists():
        logger.error(f"    No Suite2p outputs at {suite2p_path}")
        return None
        
    suite2p_data = load_suite2p_data(suite2p_path)
    n_rois, n_frames = suite2p_data['F'].shape
    logger.info(f"    Loaded {n_rois} ROIs, {n_frames} frames")
    results['suite2p_data'] = suite2p_data
    results['n_frames'] = n_frames
    
    # Step 2: Compute cascade probabilities
    logger.info("  Step 2: Computing spike probabilities with Cascade...")
    cascade_prob = compute_cascade_probabilities(
        suite2p_data['F'], 
        models['cascade'],
        batch_size=config.get('cascade_batch_size', 64)
    )
    logger.info(f"    Computed probabilities shape: {cascade_prob.shape}")
    results['cascade_prob'] = cascade_prob
    
    # Step 3: Create ROI objects
    logger.info("  Step 3: Creating ROI objects...")
    all_rois = []
    for i in range(n_rois):
        roi = ROI(
            index=i,
            f_trace=suite2p_data['F'][i],
            cascade_prob=cascade_prob[i],
            spatial_footprint=suite2p_data['stat'][i] if 'stat' in suite2p_data else None,
            fneu=suite2p_data['Fneu'][i] if 'Fneu' in suite2p_data else None
        )
        all_rois.append(roi)
    results['all_rois'] = all_rois
    
    # Step 4: Extract features and filter ROIs
    logger.info("  Step 4: Filtering ROIs (derivative_skew, spike_prom_mean)...")
    normalization = config.get('roi_filtering', {}).get('normalization', 'minmax')
    logger.info(f"    Using {normalization} normalization")
    roi_features = extract_roi_features(all_rois, normalization=normalization)
    good_roi_mask = filter_rois(roi_features, models['roi_classifier'])
    good_rois = [roi for roi, keep in zip(all_rois, good_roi_mask) if keep]
    bad_rois = [roi for roi, keep in zip(all_rois, good_roi_mask) if not keep]
    
    # Track bad ROI indices and features for Excel report
    bad_roi_indices = [roi.index for roi in bad_rois]
    bad_roi_features = {
        'derivative_skew': [roi.features.get('derivative_skew', np.nan) for roi in bad_rois],
        'spike_prom_mean': [roi.features.get('spike_prom_mean', np.nan) for roi in bad_rois]
    }
    
    logger.info(f"    {len(good_rois)}/{len(all_rois)} ROIs passed filtering")
    logger.info(f"    Bad ROI indices: {bad_roi_indices[:10]}..." if len(bad_roi_indices) > 10 else f"    Bad ROI indices: {bad_roi_indices}")
    
    results['good_rois'] = good_rois
    results['bad_roi_indices'] = bad_roi_indices
    results['bad_roi_features'] = bad_roi_features
    results['good_roi_mask'] = good_roi_mask
    
    # Step 4b: Save filtered Suite2p files
    logger.info("  Step 4b: Saving filtered Suite2p files...")
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
    
    # Step 5: Create neurons with per-video MinMax normalization
    logger.info("  Step 5: Creating Neuron objects with normalized traces...")
    neurons = []
    for roi in good_rois:
        # MinMax normalize F and cascade_prob at per-video level
        # This ensures consistency between training and inference for both ROI and spike features
        f_min, f_max = roi.f_trace.min(), roi.f_trace.max()
        f_normalized = (roi.f_trace - f_min) / (f_max - f_min + 1e-10) if f_max > f_min else roi.f_trace
        
        prob_min, prob_max = roi.cascade_prob.min(), roi.cascade_prob.max()
        prob_normalized = (roi.cascade_prob - prob_min) / (prob_max - prob_min + 1e-10) if prob_max > prob_min else roi.cascade_prob
        
        neuron = Neuron(
            row_index=roi.index,
            f_trace=f_normalized,  # Store normalized trace
            cascade_prob=prob_normalized,  # Store normalized prob
            fs=suite2p_data.get('fs', 30.0)
        )
        neurons.append(neuron)
    logger.info(f"    Normalized {len(neurons)} neuron traces to [0,1] range")
    results['neurons'] = neurons
    
    # Step 6: Detect spikes
    logger.info("  Step 6: Detecting spikes from cascade probability...")
    total_spikes = 0
    for neuron in neurons:
        spikes = detect_spikes_from_cascade(
            neuron.raw_fluorescence,
            neuron.cascade_prob,
            **config['spike_detection']
        )
        neuron.spikes = spikes
        total_spikes += len(spikes)
    logger.info(f"    Found {total_spikes} candidate spikes")
    
    # Step 7: Filter spikes (8 features)
    if models['spike_classifier'] is not None:
        logger.info("  Step 7: Filtering spikes (8 features)...")
        valid_count = 0
        for neuron in neurons:
            if len(neuron.spikes) > 0:
                spike_features = extract_spike_features(
                    neuron.spikes,
                    neuron.raw_fluorescence,
                    neuron.cascade_prob
                )
                # Apply per-video scaling before classification (features are per-neuron subset of video)
                valid_mask = filter_spikes(spike_features, models['spike_classifier'], per_video_scale=True)
                neuron.spikes = [s for s, keep in zip(neuron.spikes, valid_mask) if keep]
                valid_count += len(neuron.spikes)
        logger.info(f"    {valid_count}/{total_spikes} spikes passed filtering")
    else:
        logger.info("  Step 7: Skipping spike filtering (no model)")
    
    # Remove neurons with no spikes
    neurons_with_spikes = [n for n in neurons if len(n.spikes) > 0]
    for i, n in enumerate(neurons_with_spikes):
        n.filtered_index = i
    logger.info(f"    {len(neurons_with_spikes)} neurons have valid spikes")
    results['filtered_neurons'] = neurons_with_spikes
    
    # Step 8: Group neurons using BOTH methods
    if len(neurons_with_spikes) > 1:
        logger.info("  Step 8: Grouping neurons...")
        
        # STTC grouping
        logger.info("    Computing STTC groups...")
        sttc_groups, sttc_matrix = group_neurons_by_sttc(
            neurons_with_spikes, n_frames, **config['grouping']['sttc']
        )
        logger.info(f"      Found {len(sttc_groups)} STTC groups")
        
        # DTW grouping  
        logger.info("    Computing DTW groups...")
        dtw_groups, dtw_matrix = group_neurons_by_dtw(
            neurons_with_spikes, **config['grouping']['dtw']
        )
        
        if dtw_matrix is not None:
            logger.info(f"      Found {len(dtw_groups)} DTW groups")
            # Compare methods
            compare_groupings(sttc_groups, dtw_groups, neurons_with_spikes)
        else:
            logger.info("      DTW groups skipped (GPU not available)")
        
        results['sttc_groups'] = sttc_groups
        results['sttc_matrix'] = sttc_matrix
        results['dtw_groups'] = dtw_groups if dtw_matrix is not None else []
        results['dtw_matrix'] = dtw_matrix
    else:
        logger.info("  Step 8: Skipping grouping (need 2+ neurons)")
        results['sttc_groups'] = []
        results['dtw_groups'] = []
    
    # Step 9: Generate summary with bad ROI tracking
    logger.info("  Step 9: Generating summary...")
    summary = save_video_summary(results, video_path / 'metrics')
    results['summary'] = summary
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GCaMP Analysis Pipeline')
    parser.add_argument('--config', type=Path, default = "C:\\Users\\mzinn1\\Desktop\\Scripts\\GCaMP-analysis\\config\\pipeline_config.yaml", help='Config file path')
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
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load models once
    models = load_models(config)
    
    # Process based on input level
    if args.video:
        results = process_video_explicit(args.video, models, config)
        if results:
            logger.info(f"Video complete: {len(results.get('filtered_neurons', []))} neurons")
    
    elif args.timepoint:
        # Process all videos in timepoint
        timepoint = Timepoint(args.timepoint)
        for video_dir in sorted(args.timepoint.iterdir()):
            if video_dir.is_dir() and (video_dir / 'suite2p' / 'plane0').exists():
                video = Video(video_dir, timepoint)
                results = process_video_explicit(video_dir, models, config)
                if results and results.get('filtered_neurons'):
                    video.neurons = results['filtered_neurons']
                    video.sttc_groups = results.get('sttc_groups', [])
                    video.dtw_groups = results.get('dtw_groups', [])
                    timepoint.add_video(video)
        
        if timepoint.videos:
            save_timepoint_summary(timepoint, args.timepoint / 'summary.xlsx')
            
    elif args.experiment:
        # Process full experiment
        # Expected structure: ex337/treatment/timepoint/video/suite2p/plane0/
        
        # Check if the path contains a treatment folder
        experiment_path = Path(args.experiment)
        
        # Look for treatment folders (subdirectories of experiment root)
        treatment_folders = [d for d in experiment_path.iterdir() if d.is_dir()]
        
        if not treatment_folders:
            logger.error(f"No treatment folders found in {experiment_path}")
            return
        
        # Process each treatment
        for treatment_dir in treatment_folders:
            logger.info(f"Processing treatment: {treatment_dir.name}")
            
            # Use utility function to load structure
            from utils.io_utils import load_experiment_structure
            experiment = load_experiment_structure(treatment_dir)
            
            # Process each timepoint and video
            for timepoint in experiment.timepoints:
                logger.info(f"  Processing timepoint: {timepoint.name}")
                
                for video in timepoint.videos:
                    video_path = video.path
                    logger.info(f"    Processing video: {video.video_id}")
                    
                    results = process_video_explicit(video_path, models, config)
                    if results and results.get('filtered_neurons'):
                        video.neurons = results['filtered_neurons']
                        video.sttc_groups = results.get('sttc_groups', [])
                        video.dtw_groups = results.get('dtw_groups', [])
                        video.summary_df = results.get('summary')  # Add summary DataFrame
                
                # Save timepoint summary with videos as rows
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