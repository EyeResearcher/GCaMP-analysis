"""Process fluorescence data from multiple videos to extract ROI features for classifier training.

This script extracts all video paths from a specified dataset root directory. For each video, it computes the 
Cascade spike probabilities using a pre-trained model and normalizes the fluorescence traces using Min-Max scaling.
Both the normalized fluorescence and spike probabilities are smoothed using a Gaussian filter with sigma = 4. 

For each ROI in each video, comprehensive features are extracted including:
- Derivative-based: skewness and asymmetry of trace derivatives
- Spike prominence: mean and skew of left-based peak prominences
- Trace dynamics: rolling variance-of-variance, autocorrelation decay
- Signal quality: SNR estimate, peak density, median spike prominence

The extracted features, along with the smoothed trace, are stored in a dictionary format:
    {roi_key: {'smoothed_f_trace': smoothed_f_trace,
               'raw_traces': [raw_f_trace, raw_spike_prob],
               'features': {...comprehensive feature dict...}, 
               'label': {'value': -1, 'source': 'unlabeled'},
               'spikes': {}}}

The output is saved as a .npy file only (no JSON to avoid corruption issues)."""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d
from utils.model_utils.rois import compute_roi_features
from utils.preprocessing import normalize_minmax


# =============================================================================
# Label Format Utilities
# =============================================================================

def normalize_label_format(label_value) -> dict:
    """
    Convert old label format (int) to new format (dict).
    
    Args:
        label_value: Either int (-1/0/1) or dict with 'value' and 'source' keys
    
    Returns:
        dict: {'value': -1/0/1, 'source': 'manual'/'classifier'/'unlabeled'/'auto'}
    """
    if isinstance(label_value, dict):
        if 'value' in label_value and 'source' in label_value:
            return label_value
        return {'value': -1, 'source': 'unlabeled'}
    
    if label_value == -1:
        return {'value': -1, 'source': 'unlabeled'}
    elif label_value in [0, 1]:
        return {'value': int(label_value), 'source': 'auto'}
    else:
        return {'value': -1, 'source': 'unlabeled'}


def get_label_value(label) -> int:
    """Extract numeric label value from either dict or int format."""
    if isinstance(label, dict):
        return label.get('value', -1)
    return label




def process_roi(smoothed_f_trace: np.ndarray, 
                raw_trace: np.ndarray) -> dict:
    """Process a single ROI and extract comprehensive features, auto-label if necessary.
    Args: 
        smoothed_f_trace: Smoothed fluorescence trace (1D array)
        raw_trace: Raw fluorescence trace (1D array)  
    Returns:
        dict: Dictionary containing processed ROI data including features, label, and traces:
            {'smoothed_f_trace': smoothed_f_trace,
             'raw_traces': [raw_trace],
             'features': {...comprehensive feature dict...},
                'label': {'value': -1/0/1, 'source': 'unlabeled'/'auto'},
    """

    features, validity = compute_roi_features(smoothed_f_trace)
    critical_valid = validity.get('valid_deriv_skew', True) and validity.get('valid_prom', True)
    label = {'value': 0, 'source': 'auto'} if not critical_valid else {'value': -1, 'source': 'unlabeled'}
    
    return {
        'smoothed_trace': smoothed_f_trace,
        'raw_trace': raw_trace,
        'features': features,
        'label': label,
        'spikes': {}  # Initialize empty spikes dict for later annotation
    }


def process_video(video_path: Path) -> list:
    """Process a video and extract ROI features.
    Args:
        video_path: Path to the video directory containing Suite2p outputs
    Returns:
        list: List of tuples (roi_key, roi_data_dict) for each ROI in the video
    """
    fluorescence_file = video_path / 'suite2p' / 'plane0' / 'F.npy'
    scaled_f_file = video_path / 'suite2p' / 'plane0' / 'F_minmax.npy'
    
    if not fluorescence_file.exists():
        print(f"Fluorescence file not found for video: {video_path}")
        return []
    
    f: np.ndarray = np.load(fluorescence_file)
    scaled_f = normalize_minmax(f, scaled_f_file) if not scaled_f_file.exists() else np.load(scaled_f_file)
    smoothed_scaled_f = gaussian_filter1d(scaled_f, sigma=4.0, axis=1)
    
    video_rois = []
    for roi_idx in range(f.shape[0]):
        roi_data = process_roi(
            smoothed_f_trace=smoothed_scaled_f[roi_idx],
            raw_trace=f[roi_idx]
        )
        roi_key = f"{video_path.name}_{roi_idx}"
        video_rois.append((roi_key, roi_data))
    
    return video_rois


# =============================================================================
# Update Existing Data
# =============================================================================

def update_roi_features(roi_dict: dict) -> dict:
    """Update ROI features while preserving spike data and labels."""
    updated_dict = {}
    
    n_rois_processed = 0
    n_labels_preserved = 0
    n_spikes_preserved = 0
    
    for roi_key, roi_data in roi_dict.items():
        smoothed_f_trace = roi_data.get('smoothed_f_trace')
        if smoothed_f_trace is None:
            # Support legacy format with smoothed_traces list
            smoothed_traces = roi_data.get('smoothed_traces', [])
            if len(smoothed_traces) < 1:
                print(f"Warning: ROI {roi_key} missing smoothed trace, skipping")
                continue
            smoothed_f_trace = np.asarray(smoothed_traces[0])
        else:
            smoothed_f_trace = np.asarray(smoothed_f_trace)
        
        # Recompute ROI features
        features, validity = compute_roi_features(smoothed_f_trace)
        
        # Normalize label to new dict format
        existing_label = roi_data.get('label', -1)
        label_dict = normalize_label_format(existing_label)
        
        if label_dict['value'] in [0, 1] and label_dict['source'] == 'manual':
            n_labels_preserved += 1
        
        # Preserve all spike data
        spikes = roi_data.get('spikes', {})
        if spikes:
            n_spikes_preserved += len(spikes)
        
        updated_dict[roi_key] = {
            'smoothed_f_trace': smoothed_f_trace,
            'raw_traces': roi_data.get('raw_traces'),
            'features': features,
            'label': label_dict,
            'spikes': spikes
        }
        
        n_rois_processed += 1
    
    print(f"\n✅ Updated {n_rois_processed} ROIs")
    print(f"  - Preserved {n_labels_preserved} manual ROI labels")
    print(f"  - Preserved {n_spikes_preserved} spikes")
    
    return updated_dict


# =============================================================================
# Main Entry Points
# =============================================================================

def main():
    """Extract ROI features from Suite2p data and save to .npy file."""
    argparser = argparse.ArgumentParser(description='Prepare ROI features from Suite2p fluorescence data')
    argparser.add_argument('--dataset_root', type=str, default=r"C:\Users\mzinn1\Desktop\Datasets")
    argparser.add_argument('--update', action='store_true', 
                                        help='Update existing features instead of processing raw videos')
    argparser.add_argument('--input_file', type=str, 
                                        default='training_data/roi_filtering/all_roi_features.npy',
                                        help='Input file for update mode')
    argparser.add_argument('--backup', action='store_true', default=True,
                                       help='Create backup before overwriting (update mode only)')
    args = argparser.parse_args()
    
    dataset_root = Path(args.dataset_root)  
    output_path = Path('training_data/roi_filtering')
    output_path.mkdir(parents=True, exist_ok=True)
    npy_file = output_path / 'all_roi_features.npy'
    
    if args.update:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            return
        
        roi_dict = np.load(input_path, allow_pickle=True).item()
        n_good = sum(1 for roi in roi_dict.values() if get_label_value(roi.get('label')) == 1)
        n_bad = sum(1 for roi in roi_dict.values() if get_label_value(roi.get('label')) == 0)
        n_unlabeled = sum(1 for roi in roi_dict.values() if get_label_value(roi.get('label')) == -1)
        total_spikes = sum(len(roi.get('spikes', {})) for roi in roi_dict.values())
        
        print(f"\nCurrent Data:")
        print(f"  - Good ROIs: {n_good}, Bad ROIs: {n_bad}, Unlabeled: {n_unlabeled}")
        print(f"  - Total spikes: {total_spikes}")
        
        if args.backup:
            import shutil
            from datetime import datetime
            backup_path = input_path.with_suffix(f'.backup_{datetime.now():%Y%m%d_%H%M%S}.npy')
            shutil.copy(input_path, backup_path)
            print(f"\nCreated backup: {backup_path}")
        
        all_roi_dict = update_roi_features(roi_dict)

    else:
        video_paths = [path for path in dataset_root.iterdir() if path.is_dir()]
        all_rois = []
        
        for video_path in video_paths:
            video_rois = process_video(video_path)
            if video_rois:
                all_rois.extend(video_rois)
        
        all_roi_dict = dict(all_rois)
    
    np.save(npy_file, all_roi_dict)
    print(f"\nSaved {len(all_roi_dict)} ROIs to {npy_file}")


if __name__ == '__main__':
    main()
