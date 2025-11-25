"""This script processes fluorescence data from multiple videos to extract ROI features for classifier training.

It extracts all video paths from a specified dataset root directory. For each video, it computes the 
Cascade spike probabilities using a pre-trained model and normalizes the fluorescence traces using Min-Max scaling.
Both the normalized fluorescence and spike probabilities are smoothed using a Gaussian filter with sigma = 4. 

For each ROI in each video, features are extracted: the skewness of the derivative of the smoothed fluorescence 
trace and the mean left-based prominence of peaks in the smoothed spike probability trace. 

The extracted features, along with the smoothed traces, are stored in a dictionary format:
    {roi_key: {'smoothed_traces': [smoothed_f_trace, smoothed_spike_prob],
               'raw_traces': [raw_f_trace, raw_spike_prob],
               'features': {'derivative_skew': value, 'spike_prom_mean': value, 'spike_prom_skew': value}, 
               'label': -1}}

The output is saved as a .npy file only (no JSON to avoid corruption issues)."""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import load_cascade_model
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew
from scipy.signal import find_peaks, peak_prominences


def normalize_minmax(f: np.ndarray, output_file: Path) -> np.ndarray:
    scaler = MinMaxScaler()
    flat_f = f.reshape(-1, 1)
    scaled_flat = scaler.fit_transform(flat_f)
    scaled_f = scaled_flat.reshape(f.shape)
    np.save(output_file, scaled_f)
    return scaled_f

def compute_cascade_probabilities(f: np.ndarray, output_file: Path) -> np.ndarray:
    model = load_cascade_model()
    spike_probs = model.predict(f).squeeze()
    np.save(output_file, spike_probs)
    return spike_probs


def left_based_prominence(spike_prob: np.ndarray) -> np.ndarray:
    peaks, _ = find_peaks(spike_prob)
    if len(peaks) == 0:
        return (0.0, False)
    peak_mean = float(np.mean(spike_prob[peaks])) if len(peaks) > 0 else 0.0
    proms, left_bases, _ = peak_prominences(spike_prob, peaks)
    peak_vals : np.ndarray = spike_prob[peaks]
    left_vals : np.ndarray = spike_prob[left_bases]
    left_base_prominences : np.ndarray = peak_vals - left_vals
    prom_mean : float = np.mean(left_base_prominences)
    prom_skew : float = skew(left_base_prominences) if len(left_base_prominences) > 0 else 0.0
    return (float(prom_mean), float(prom_skew), True, peaks)

def derivative_skewness(smoothed_scaled_f: np.ndarray) -> float:
    derivative = np.diff(smoothed_scaled_f)
    if len(derivative) == 0:
        return (0.0, False)
    if np.any(np.isnan(derivative)) or np.any(np.isinf(derivative)):
        return (0.0, False) 
    return (float(skew(derivative)), True)

def roi_feature_extraction(smoothed_f_trace: np.ndarray, smoothed_spike_prob: np.ndarray) -> tuple[dict, dict, np.ndarray]:
    """Extract ROI features. Returns only the full dict with numpy arrays."""
    deriv_skew, valid_deriv = derivative_skewness(smoothed_f_trace)
    spike_prom_mean, spike_prom_skew, valid_prom, peaks = left_based_prominence(smoothed_spike_prob)
    return ({
            'derivative_skew': deriv_skew,
            'spike_prom_mean': spike_prom_mean,
            'spike_prom_skew': spike_prom_skew,
            'range_trace': float(np.nanmax(smoothed_f_trace) - np.nanmin(smoothed_f_trace))},
            {'valid_deriv': valid_deriv,
              'valid_prom': valid_prom}, peaks)

def process_roi(smoothed_f_trace: np.ndarray, smoothed_spike_prob: np.ndarray, 
                raw_trace: np.ndarray, raw_spike_prob: np.ndarray) -> dict:
    features, validity, _ = roi_feature_extraction(smoothed_f_trace, smoothed_spike_prob)
    label = 0 if not validity['valid_deriv'] or not validity['valid_prom'] else -1
    return {
        'smoothed_traces': [smoothed_f_trace, smoothed_spike_prob],
        'raw_traces': [raw_trace, raw_spike_prob],
        'features': features,
        'label': label
    }



def process_video(video_path: Path):
    """Process a video and extract ROI features. Returns only npy data."""
    fluorescence_file = video_path / 'suite2p' / 'plane0' / 'F.npy'
    scaled_f_file = video_path / 'suite2p' / 'plane0' / 'F_minmax.npy'
    cascade_probs_file = video_path / 'suite2p' / 'plane0' / 'cascade_spike_prob.npy'
    if not fluorescence_file.exists():
        print(f"Fluorescence file not found for video: {video_path}")
        return []
    f: np.ndarray = np.load(fluorescence_file)
    scaled_f = normalize_minmax(f, scaled_f_file) if not scaled_f_file.exists() else np.load(scaled_f_file)
    smoothed_scaled_f = gaussian_filter1d(scaled_f, sigma=4.0, axis=1)
    
    # Always recompute CASCADE probabilities to ensure fresh data
    print(f"Computing CASCADE probabilities for {video_path.name}...")
    cascade_probs = compute_cascade_probabilities(f, cascade_probs_file)
    
    smoothed_probs = gaussian_filter1d(cascade_probs, sigma=4.0, axis=1)
    video_rois = []
    for roi_idx in range(f.shape[0]):
        f_trace_sm = smoothed_scaled_f[roi_idx]
        spike_prob_sm = smoothed_probs[roi_idx]
        features_dict = roi_feature_extraction(f_trace_sm, spike_prob_sm, f[roi_idx], cascade_probs[roi_idx])
        print(cascade_probs[roi_idx][300:350])
        roi_key = f"{video_path.name}_{roi_idx}"
        video_rois.append((roi_key, features_dict))
    return video_rois

# Removed change_dict_format function - no longer needed since we don't use JSON

def main():
    """Extract ROI features from Suite2p data and save to .npy file only."""
    argparser = argparse.ArgumentParser(description='Prepare ROI features from Suite2p fluorescence data')
    argparser.add_argument('--dataset_root', type=str, default=r"C:\Users\mzinn1\Desktop\Datasets")
    args = argparser.parse_args()
    
    video_paths = [path for path in Path(args.dataset_root).iterdir() if path.is_dir()]
    all_rois = []
    
    for video_path in video_paths:
        video_rois = process_video(video_path)
        if video_rois:  # Only extend if not empty
            all_rois.extend(video_rois)
    
    all_roi_dict = dict(all_rois)
    output_path = Path('training_data/roi_filtering')
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_file = output_path / 'all_roi_features.npy'
    np.save(npy_file, all_roi_dict)
    print(f"✅ Saved {len(all_roi_dict)} ROIs to {npy_file}")

if __name__ == '__main__':
    main()
