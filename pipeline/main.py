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
from typing import Dict, List, Optional
import yaml
import numpy as np
import pandas as pd
from joblib import load, dump, Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
# Clean imports using __init__.py aggregators
from data_classes import Experiment, Timepoint, Video, ROI, Neuron, Spike

# Import from individual modules to avoid circular import
from pipeline.preprocessing import load_suite2p_data, compute_cascade_probabilities
from pipeline.roi_processing import extract_roi_features, filter_rois
from pipeline.spike_detection import detect_spikes_from_cascade
from pipeline.spike_filtering import extract_spike_features, filter_spikes
from pipeline.neuron_grouping import group_neurons_by_sttc, group_neurons_by_dtw, compare_groupings
from pipeline.io_handlers import save_video_summary, save_timepoint_summary, save_filtered_suite2p
from utils.io_utils import load_experiment_structure
from utils.cascade_utils import load_cascade_model

logger = logging.getLogger(__name__)

def load_models(config: Dict) -> Dict:
    """Load all required models."""
    models = {}
    
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
        
    spike_path = Path(config['models']['spike_model_path'])
    try:
        models['spike_classifier'] = load(spike_path)
        logger.info(f"Loaded spike classifier from {spike_path}")
    except (EOFError, Exception) as e:
        raise RuntimeError(f"Failed to load spike classifier from {spike_path}: {e}")
    model_name = config['models']['cascade_model_name']
    model_dir = config['models']['cascade_model_dir']

    models['cascade'] = load_cascade_model(
        model_name=model_name,
        model_dir=model_dir
    )
    logger.info("Loaded Cascade model")
    
    return models
def filter_rois(all_rois: List[ROI], roi_classifier : LogisticRegression,
                 norm_sm_f: np.ndarray, sm_sp: np.ndarray) -> tuple[List[ROI], List[ROI], np.ndarray]:
    """Extract features and filter ROIs using the classifier."""
    all_feats = Parallel(n_jobs=-1)(
        delayed(roi.extract_features)(norm_sm_f[i, :], sm_sp[i, :]) 
        for i, roi in enumerate(all_rois)
    )
    feats_df = pd.DataFrame(all_feats)
    good_roi_mask  = roi_classifier.predict(feats_df)
    for roi, pred, i in zip(all_rois, good_roi_mask, range(len(all_rois))):
        if roi.is_good is False:
            continue
        good_roi_mask[i] = roi.is_good = bool(pred)
    good_rois = [roi for roi in all_rois if roi.is_good]
    bad_rois = [roi for roi in all_rois if not roi.is_good  ]
    return good_rois, bad_rois, good_roi_mask
def get_savgol_params(fs, sensor_type='gcamp8s'):
    """Get Savitzky-Golay parameters based on sampling frequency and sensor."""
    if sensor_type == 'gcamp8s':
        # Target ~500-800ms window for GCaMP8s (slower kinetics)
        window_frames = int(0.6 * fs)  # 600ms window
    elif sensor_type == 'gcamp6f':
        # Faster sensor, shorter window
        window_frames = int(0.3 * fs)  # 300ms window
    else:
        # Default/GCaMP6s
        window_frames = int(0.4 * fs)  # 400ms window
    
    # Ensure odd number
    window_length = 2 * (window_frames // 2) + 1
    # For GCaMP8s, use larger minimum
    window_length = max(9, window_length)  # Minimum 9 for slow sensors
    
    polyorder = 3  # Cubic fit better for slow, smooth transients
    
    return window_length, polyorder

def process_video_explicit(video_path: Path, models: Dict, config: Dict) -> Optional[Dict]:
    """
    Process a single video through all pipeline steps explicitly.
    
    Each step is clearly visible for debugging.
    Includes bad ROI tracking and filtered Suite2p saving.
    """
    logger.info(f"Processing video: {video_path.name}")
    results = {'video_path': video_path}
    
    # Step 1: Load Suite2p data
    from roi_classifier.prepare_data import roi_feature_extraction, normalize_minmax
    logger.info("  Step 1: Loading Suite2p data...")
    suite2p_path = video_path / 'suite2p' / 'plane0'
    if not (suite2p_path / 'F.npy').exists():
        logger.error(f"    No Suite2p outputs at {suite2p_path}")
        return None
        
    suite2p_data = load_suite2p_data(suite2p_path)
    norm_f = normalize_minmax(suite2p_data['F'], suite2p_path / 'F_minmax.npy')
    norm_sm_f = gaussian_filter1d(norm_f, sigma=4.0, axis=0 )
    window_length, polyorder = get_savgol_params(suite2p_data['fs'], sensor_type='gcamp8s')
    norm_sg_f = savgol_filter(norm_f, window_length=window_length, polyorder=polyorder, axis=1)
    n_rois, n_frames = suite2p_data['F'].shape
    logger.info(f"    Loaded {n_rois} ROIs, {n_frames} frames")
    results['suite2p_data'] = suite2p_data
    results['n_frames'] = n_frames
    
    # Step 2: Compute cascade probabilities
    logger.info("  Step 2: Computing spike probabilities with Cascade...")
    cascade_prob = models['cascade'].predict(suite2p_data['F'])
    
    sm_sp = gaussian_filter1d(cascade_prob, sigma=4.0, axis=0 )
    logger.info(f"    Computed probabilities shape: {cascade_prob.shape}")
    results['cascade_prob'] = cascade_prob
    
    # Step 3: Create ROI objects
    logger.info("  Step 3: Creating ROI objects...")
    all_rois : List[ROI] = []
    for i in range(n_rois):
        roi = ROI(
            index=i,
            f_trace=suite2p_data['F'][i],
            sp_trace=cascade_prob[i],
            stats=suite2p_data['stat'][i] if 'stat' in suite2p_data else None,
            fneu=suite2p_data['Fneu'][i] if 'Fneu' in suite2p_data else None,
            
        )
        all_rois.append(roi)
    results['all_rois'] = all_rois
    
    good_rois, bad_rois, good_roi_mask = filter_rois(all_rois, 
                                           models['roi_classifier'],
                                            norm_sm_f, sm_sp)
  
    
    # Track bad ROI indices and features for Excel report
    bad_roi_indices = [roi.index for roi in bad_rois]
    bad_roi_features = [roi.features for roi in bad_rois]
    
    logger.info(f"    {len(good_rois)}/{len(all_rois)} ROIs passed filtering")
    logger.info(f"    Bad ROI indices: {bad_roi_indices[:10]}..." if len(bad_roi_indices) > 10 else f"    Bad ROI indices: {bad_roi_indices}")
    
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

    neurons = [Neuron(roi, i, fs=suite2p_data['fs']) for i, roi in enumerate(good_rois)]
    results['neurons'] = neurons
    # Step 6: Extract Spike features 
    logger.info("  Step 6: Extracting spike features in parallel...")
    spike_features_list = Parallel(n_jobs=-1)(
    delayed(neuron.get_spike_features)(sm_sp[neuron.index, :])
    for neuron in neurons
    )

    # Flatten list of lists of feature dicts into single list
    spike_features_flat = [feat_dict for neuron_feats in spike_features_list 
                        for feat_dict in neuron_feats]
    logger.info(f"    Extracted features for {len(spike_features_flat)} total spikes")

    # Step 7: Filter spikes using the classifier
    logger.info("  Step 7: Filtering spikes...")
    spk_feats_df = pd.DataFrame(spike_features_flat)
    spike_mask = models['spike_classifier'].predict(spk_feats_df)

    # Map predictions back to each neuron
    prev_idx = 0
    for neuron in neurons:
        # Number of spikes actually extracted for this neuron
        n_spikes_extracted = len(neuron.spk_features)
        spike_preds = spike_mask[prev_idx: prev_idx + n_spikes_extracted]
        prev_idx += n_spikes_extracted

    # Filter peaks based on predictions
        neuron.peaks_filtered = neuron.filter_spikes(spike_preds)

    logger.info(f"    {spike_mask.sum()}/{len(spike_mask)} spikes passed filtering")

    # Remove neurons with no spikes
    neurons_with_spikes = [n for n in neurons if len(n.spikes) > 0]
    for i, n in enumerate(neurons_with_spikes):
        n.filtered_index = i

    inst_spikes = Parallel(n_jobs=-1)(
        delayed(n.instantiate_spikes)(norm_sm_f[n.index, :], norm_sg_f[n.index, :]) for n in neurons_with_spikes
    )
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
def process_timepoint(timepoint_path: Path, models: Dict, config: Dict) -> Timepoint:
    """Process all videos in a timepoint folder."""
    timepoint = Timepoint(timepoint_path)
    
    for video_dir in sorted(timepoint_path.iterdir()):
        if video_dir.is_dir() and (video_dir / 'suite2p' / 'plane0').exists():
            video = Video(video_dir, timepoint)
            results = process_video_explicit(video_dir, models, config)
            if results and results.get('filtered_neurons'):
                video.neurons = results['filtered_neurons']
                video.sttc_groups = results.get('sttc_groups', [])
                video.dtw_groups = results.get('dtw_groups', [])
                timepoint.add_video(video)
    
    return timepoint
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
                timepoint = process_timepoint(timepoint.path, models, config)
                experiment.add_timepoint(timepoint)
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