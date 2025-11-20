"""This script processes fluorescence data from multiple videos to extract ROI features for classifier training.
It first extracts all video paths from a specified dataset root directory. For each video, it computes the 
Cascade spike probabilities using a pre-trained model and normalizes the fluorescence traces using Min-Max scaling.
Both the normalized fluorescence and spike probabilities are smoothed using a Gaussian filter with sigma = 4. For each ROI in
each video, two features are extracted: the skewness of the derivative of the smoothed fluorescence trace and the mean 
left-based prominence of peaks in the smoothed spike probability trace. The extracted features, along with the smoothed traces,
are stored in a dictionary format suitable for training a ROI classifier: 
                {roi_key: {'traces': [smoothed_f_trace, smoothed_spike_prob],
                           'features': {'derivative_skew': value, 'spike_prom_mean': value}, 
                           'label': -1}}. 
An additional dictionary is saved in JSON format for easy inspection.
                {roi_key: {'features': {'derivative_skew': value, 'spike_prom_mean': value},
                           'label': -1}}.
The final output is saved as a NumPy file and a JSON file for easy access during training."""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import json
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
    return (float(prom_mean), float(prom_skew), True)

def derivative_skewness(smoothed_scaled_f: np.ndarray) -> float:
    derivative = np.diff(smoothed_scaled_f)
    if len(derivative) == 0:
        return (0.0, False)
    if np.any(np.isnan(derivative)) or np.any(np.isinf(derivative)):
        return (0.0, False) 
    return (float(skew(derivative)), True)

def roi_feature_extraction(smoothed_f_trace: np.ndarray, smoothed_spike_prob: np.ndarray, 
                           raw_trace: np.ndarray, raw_spike_prob: np.ndarray ) -> dict:
    deriv_skew, valid_deriv = derivative_skewness(smoothed_f_trace)
    spike_prom_mean, spike_prom_skew, valid_prom = left_based_prominence(smoothed_spike_prob)
    label = 0 if not valid_deriv or not valid_prom else -1
    return ({'smoothed_traces': [smoothed_f_trace, smoothed_spike_prob],
             'raw_traces': [raw_trace, raw_spike_prob],
            'features': {'derivative_skew': deriv_skew, 'spike_prom_mean': spike_prom_mean, 'spike_prom_skew': spike_prom_skew},
            'label': label}, {'features': {'derivative_skew': float(deriv_skew), 'spike_prom_mean': float(spike_prom_mean), 'spike_prom_skew': float(spike_prom_skew)}, 'label': int(label)})



def process_video(video_path : Path):
    fluorescence_file = video_path / 'suite2p' / 'plane0' / 'F.npy'
    scaled_f_file = video_path / 'suite2p' / 'plane0' / 'F_minmax.npy'
    cascade_probs_file = video_path / 'suite2p' / 'plane0' / 'cascade_spike_prob.npy'
    if not fluorescence_file.exists():
        print(f"Fluorescence file not found for video: {video_path}")
        return
    f : np.ndarray = np.load(fluorescence_file)
    scaled_f = normalize_minmax(f, scaled_f_file) if not scaled_f_file.exists() else np.load(scaled_f_file)
    smoothed_scaled_f = gaussian_filter1d(scaled_f, sigma=4.0, axis=1)
    cascade_probs = compute_cascade_probabilities(f, cascade_probs_file) #if not cascade_probs_file.exists() else np.load(cascade_probs_file)
    smoothed_probs = gaussian_filter1d(cascade_probs, sigma=4.0, axis=1)
    video_rois = []
    video_rois_json = []
    for roi_idx in range(f.shape[0]):
        f_trace = smoothed_scaled_f[roi_idx]
        spike_prob = smoothed_probs[roi_idx]
        features_dict, json_dict = roi_feature_extraction(f_trace, spike_prob, f[roi_idx], cascade_probs[roi_idx])
        print(cascade_probs[roi_idx][300:350])
        roi_key = f"{video_path.name}_{roi_idx}"
        video_rois.append((roi_key, features_dict))
        video_rois_json.append((roi_key, json_dict))
    return video_rois, video_rois_json

def change_dict_format(npy_dict_path: Path, json_path: Path, dataset_root: Path):
    npy_dict = np.load(npy_dict_path, allow_pickle=True).item()
    with open(json_path, 'r') as f:
        json_dict = json.load(f)
    new_dict = {}
    new_json_dict = {}
    for roi_key, roi_data in npy_dict.items():
        video_path = dataset_root / roi_key.split('_')[0]
        roi_idx = int(roi_key.split('_')[1])
        video_f = np.load(video_path / 'suite2p' / 'plane0' / 'F.npy')
        roi_data['features'].pop('spike_peak_mean', None)
        roi_data['features'].pop('raw_derivative_skew', None)
        new_dict[roi_key] = roi_data

        json_data = json_dict[roi_key]
        json_data['features'].pop('raw_derivative_skew', None)
        json_data['features'].pop('spike_peak_mean', None)
        new_json_dict[roi_key] = json_data
    output_path = npy_dict_path
    np.save(output_path, new_dict)
    with open(json_path, 'w') as f:
        json.dump(new_json_dict, f, indent=2)
    print(f"Updated dictionary saved to {output_path}")

def main():
    argparser = argparse.ArgumentParser(description='Prepare ROI features from labels and train classifier')
    argparser.add_argument('--dataset_root', type=str, default = r"C:\Users\mzinn1\Desktop\Datasets" )
    argparser.add_argument('-c', '--change_dict', action='store_true', help='Change dictionary format to include raw and smoothed traces')
    args = argparser.parse_args()
    if args.change_dict:
        npy_dict_path = Path('training_data/roi_filtering/all_roi_features.npy')
        json_path = Path('training_data/roi_filtering/all_roi_features.json')
        change_dict_format(npy_dict_path, json_path, Path(args.dataset_root))
        return
    video_paths = [path for path in Path(args.dataset_root).iterdir() if path.is_dir()]
    all_rois = []
    all_rois_json = []
    for video_path in video_paths:
        video_rois = process_video(video_path)
        all_rois.extend(video_rois[0])
        all_rois_json.extend(video_rois[1])
    all_roi_dict = dict(all_rois)
    all_roi_dict_json = dict(all_rois_json)
    outputh_path = Path('training_data/roi_filtering')
    outputh_path.mkdir(parents=True, exist_ok=True)
    np.save(outputh_path / 'all_roi_features.npy', all_roi_dict)
    with open('data.json', 'w') as f:
        json.dump(all_roi_dict_json, f, indent=2)

if __name__ == '__main__':
    main()
