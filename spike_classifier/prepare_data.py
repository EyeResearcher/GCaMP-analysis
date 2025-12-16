import argparse
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks, peak_prominences, savgol_filter
from scipy.stats import skew
import pandas as pd
from utils.feature_utils import (
    compute_peak_hierarchy_features,
    compute_spike_constants,
    compute_decay_shape_features,
    compute_additional_decay_features,
    area_asymmetry,
    area_asymmetry_trapz,
    _create_large_window, _create_small_window    
)
def load_roi_data(file_path: str) -> Dict[str, Dict]:
    """
    Load ROI data from a .npy file.
    """
    roi_data = np.load(file_path, allow_pickle=True).item()
    return roi_data

def _compute_spike_features(
    large_window: np.ndarray,
    small_window: np.ndarray,
    spike_prom: float,
    peak_idx: int,
    left_base_idx: int,
    absolute_prev_min: int,
    hierarchy: dict,
    i: int
) -> dict:
    """
    Compute all features for a detected spike.
    Args:
        large_window: Prominence-based window around spike
        small_window: Inter-peak window around spike
        spike_prom: Spike prominence value
        peak_idx: Peak index in valid region coordinates
        left_base_idx: Left base index in valid region coordinates
        absolute_prev_min: Absolute index of previous minimum
    Returns:
        Dictionary of spike features
    """
  

    peak_in_large_window = peak_idx - left_base_idx
    rise_slope, decay_tau = compute_spike_constants(
        small_window, 
        peak_in_large_window, 
        fs=30.0
    )
    decay_shape = compute_decay_shape_features(
        small_window, 
        peak_in_large_window, 
        fs=30.0
    )
    additional_decay = compute_additional_decay_features(
        small_window, 
        peak_in_large_window
    )

    return {
        "spike_prom": float(spike_prom),
        "isolation": int(len(large_window)),
        "distance": int(len(small_window)),
        "iso_skew": float(skew(large_window)) if large_window.size else 0.0,
        "dist_skew": float(skew(small_window)) if small_window.size else 0.0,
        "iso_aai_sum": float(area_asymmetry(large_window, peak_idx - left_base_idx)),
        "dist_aai_sum": float(area_asymmetry(small_window, peak_idx - absolute_prev_min)),
        "iso_aai_trapz": float(area_asymmetry_trapz(large_window, zero_value=peak_idx - left_base_idx)),
        "dist_aai_trapz": float(area_asymmetry_trapz(small_window , zero_value=peak_idx - absolute_prev_min)),
        "rise_slope": float(rise_slope),
        "decay_tau": float(decay_tau),
        
        # Decay shape features (from smoothed trace)
        **decay_shape,
        **additional_decay,
        
        "dominance_score": float(hierarchy["dominance_score"][i]),
        "local_rank": int(hierarchy["local_rank"][i]),
        "local_rank_norm": float(hierarchy["local_rank_norm"][i]),
        "cluster_size": int(hierarchy["cluster_size"][i]),
        "prom_gap": float(hierarchy["prom_gap"][i]),
        "time_to_parent": float(hierarchy["time_to_parent"][i]),
    
    }



def detect_spikes(smoothed_f: np.ndarray = None,
                   peaks : np.ndarray | None = None, roi_idx = None,
                     mode = "train") -> Tuple[Dict[int, Dict], list[int|str]]:
    """
    Detect spikes and compute windows/features per spike.
    Args:
        smoothed_spike_prob (np.ndarray): 1D array of smoothed spike probabilities
        peaks (np.ndarray | None): Optional precomputed peak indices
        roi_idx: Optional ROI index for key construction
        mode: "train" to return detailed spike data, "inference" for feature list only
    Returns:
        Tuple containing for training mode:
            - spike_data (Dict[int, Dict]): Mapping of spike index to its data (windows, features, label)
            - spike_keys (list[int|str]): List of spike indices detected
        or for inference mode:
            - features_list (list[Dict]): List of feature dictionaries for each spike
            - spike_keys (list[int|str]): List of spike keys

    """
    
    
    # Detect peaks in the valid region
    peaks, _ = find_peaks(smoothed_f) if peaks is None else (peaks, None)
    if peaks.size == 0:
        return {}, []

    prominences, left_bases, right_bases = peak_prominences(
        smoothed_f, peaks
    )

    widths = right_bases - left_bases

    hierarchy = compute_peak_hierarchy_features(
    peaks=peaks,
    prominences=prominences,
    widths=widths,
    width_factor=1.5,
)

    spike_data: Dict[int, Dict] = {}
    num_peaks = len(peaks)
    spike_keys = []
    inference_list = []
    for i, peak in enumerate(peaks):
        large_window_f, absolute_left_base, absolute_right_base, spike_prom = _create_large_window(
            smoothed_f, peak, left_bases[i], right_bases[i]
        )
        # Adjust indices back to original array coordinates
        prev_peak = peaks[i - 1] if i > 0 else 0
        next_peak = peaks[i + 1] if i < num_peaks - 1 else len(smoothed_f)
        small_window_f, absolute_prev_min, absolute_next_min = _create_small_window(
            smoothed_f, peak, prev_peak, next_peak
        )
        
        features = _compute_spike_features(
            large_window_f, small_window_f, spike_prom,
            peak, left_bases[i], absolute_prev_min, hierarchy, i
        )
       
        # Ensure the small window is non-empty and ordered.
        
        spike_key = peak if roi_idx is None else f"{roi_idx}_{peak}"
        spike_keys.append(spike_key)
        if mode == "inference":
            features = {"spike_prom": features["spike_prom"],
                        "dominance_score": features["dominance_score"],
                        "prom_gap": features["prom_gap"]}
            inference_list.append(features)
            continue
            
        spike_data[peak] = {
            "windows": {
                'large_window': {
                    'window_values': large_window_f, 
                    'bounds': (absolute_left_base, absolute_right_base)
                },
                'small_window': {
                    'window_values': small_window_f, 
                    'bounds': (absolute_prev_min, absolute_next_min)
                }
            },
            "features": features,
            "label": -1,  
        }
    
      
    
    if mode == "inference":
        return inference_list, spike_keys

    
    return spike_data, spike_keys

def parallel_detect_spikes(args):
    smoothed_f, roi_idx = args
    spike_data, spike_keys = detect_spikes(smoothed_f, roi_idx=roi_idx)
    return spike_data, spike_keys

def validate_roi_label(roi_key: str, roi_data: Dict) -> None:
    """
    Validate that all ROIs have a 'label' field with a 'value'.
    
    Args:
        roi_dict: Dictionary of ROI data
    
    Raises:
        ValueError: If any ROI is missing 'label' or 'value'.
    """

    roi_label = roi_data.get('label', None)
    if roi_label is None:
        raise ValueError(f"ROI {roi_key} is missing 'label' data.")
    
    roi_label_value = roi_label.get('value', None)

    if roi_label_value is None:
        raise ValueError(f"ROI {roi_key} has label without 'value' field.")
    
    if roi_label_value != 1:
        return False
    
    return True

def process_rois(roi_dict: Dict, max_rois: Optional[int] = None) -> Tuple[Dict, list]:
    """
    Iterate through ROI data and collect spike features for labeled-good ROIs (label == 1).
    
    Args:
        roi_dict: Dictionary of ROI data
        max_rois: Maximum number of good ROIs to process (None = all)
    
    Returns: (roi_dict_with_spikes, all_roi_spike_keys)
    """
    all_roi_spike_keys = []
    processed_count = 0
    bad_roi = 0
    for roi_key, roi_data in roi_dict.items():

        valid = validate_roi_label(roi_key, roi_data)
        if not valid:
            bad_roi += 1
            continue 

        if max_rois is not None and processed_count >= max_rois:
            print(f"Reached max_rois limit ({max_rois}), stopping spike extraction")
            break

        smoothed_f_trace = roi_data.get("smoothed_trace", None)
        if smoothed_f_trace is None:
            raise ValueError(f"ROI {roi_key} is missing 'smoothed_trace' data.")
        smoothed_f_trace = np.asarray(smoothed_f_trace)


        roi_spike_data, spike_keys = detect_spikes(smoothed_f_trace)
        
        # Preserve existing labels if spikes already exist
        existing_spikes = roi_data.get('spikes', {})
        for spike_idx in spike_keys:
            if spike_idx in existing_spikes:
                # Preserve the existing label
                roi_spike_data[spike_idx]['label'] = existing_spikes[spike_idx]['label']
        
        roi_spike_keys = [(f"{roi_key}-{spike_idx}", roi_spike_data[spike_idx]['label']) for spike_idx in spike_keys]
        all_roi_spike_keys.extend(roi_spike_keys)
        roi_dict[roi_key]['spikes'] = roi_spike_data
        processed_count += 1

    print(f"Processed {processed_count} good ROIs, found {len(all_roi_spike_keys)} spikes")
    print(f"Skipped {bad_roi} bad ROIs")
    return roi_dict, all_roi_spike_keys


def main(input_path: str, output_path: Optional[str] = None, max_rois: Optional[int] = None) -> Dict[str, Dict[int, Dict]]:
    roi_dict = load_roi_data(input_path)
    
    
    roi_dict_spikes, all_roi_spike_keys = process_rois(roi_dict, max_rois=max_rois)
    
    if output_path:
        np.save(output_path, roi_dict_spikes)
        all_roi_spike_keys_path = output_path.replace('.npy', '_spike_keys.csv')
        df = pd.DataFrame(all_roi_spike_keys, columns=['spike_key', 'label'])
        df.to_csv(all_roi_spike_keys_path, index=False)
    if not output_path:
        np.save(input_path, roi_dict_spikes)
        all_roi_spike_keys_path = input_path.replace('.npy', '_spike_keys.csv')
        df = pd.DataFrame(all_roi_spike_keys, columns=['spike_key', 'label'])
        df.to_csv(all_roi_spike_keys_path, index=False)
    return roi_dict_spikes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare spike feature data from fluorescence traces."
    )
    parser.add_argument(
        "--input_path", 
        help="Path to .npy file containing ROI dictionary.",
        default="training_data/roi_filtering/all_roi_features.npy")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="Optional path to save extracted spike features (.npy).",
    )
    parser.add_argument(
        "--max_rois",
        type=int,
        default=None,
        help="Maximum number of good ROIs to process (default: all ROIs)",
    )

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.max_rois)
