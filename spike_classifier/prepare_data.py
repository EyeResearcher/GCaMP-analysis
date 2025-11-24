import argparse
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import skew
import pandas as pd

def load_roi_data(path: str) -> Dict:
    """Load ROI dictionary saved as a .npy file."""
    data = np.load(path, allow_pickle=True)
    # np.load with allow_pickle returns an array with a single dict item in this layout.
    return data.item()  # type: ignore[no-any-return]

def _compute_min_between(
    trace: np.ndarray, start: int, end: int
) -> int:
    """Return index of minimum value between start and end (exclusive of end)."""
    if start >= end:
        return start
    local_min = int(np.argmin(trace[start:end]))
    return start + local_min

def area_asymmetry(window: np.ndarray, zero_index: int) -> float:
    """
    Compute the Area-Asymmetry Index (AAI) of a 1D signal relative to a given zero index.
    
    AAI = (A_pos - A_neg) / (A_pos + A_neg)
    
    where:
        A_neg = sum(|signal[i]| for i < zero_index)
        A_pos = sum(|signal[i]| for i > zero_index)

    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values.
    zero_index : int
        Index representing the zero-reference boundary.

    Returns
    -------
    float
        Asymmetry index in [-1, 1].
    """
    left = np.sum(np.abs(window[:zero_index]))
    right = np.sum(np.abs(window[zero_index+1:]))

    if left + right == 0:
        return 0.0

    return (right - left) / (right + left)

def area_asymmetry_trapz(
    signal: np.ndarray,
    x: Optional[np.ndarray] = None,
    zero_value: float = 0.0
) -> float:
    """
    Compute the Area-Asymmetry Index (AAI) using trapezoidal integration
    of |signal| on each side of a zero-reference x-value.

        AAI = (A_pos - A_neg) / (A_pos + A_neg)

    where:
        A_neg = ∫_{x < zero_value} |signal(x)| dx
        A_pos = ∫_{x > zero_value} |signal(x)| dx

    Parameters
    ----------
    signal : np.ndarray
        1D array of y-values of the signal.
    x : np.ndarray, optional
        1D array of x-values (same length as signal). If None, uses
        x = np.arange(len(signal)).
    zero_value : float, optional
        The x-coordinate representing the zero boundary.

    Returns
    -------
    float
        Asymmetry index in [-1, 1]. Returns 0.0 if total area is zero.
    """
    signal = np.asarray(signal)
    if x is None:
        x = np.arange(signal.shape[0])
    else:
        x = np.asarray(x)
        assert x.shape == signal.shape, "x and signal must have same shape"

    abs_sig = np.abs(signal)

    left_mask = x < zero_value
    right_mask = x > zero_value

    if np.any(left_mask):
        A_left = np.trapz(abs_sig[left_mask], x[left_mask])
    else:
        A_left = 0.0

    if np.any(right_mask):
        A_right = np.trapz(abs_sig[right_mask], x[right_mask])
    else:
        A_right = 0.0

    total = A_left + A_right
    if total == 0:
        return 0.0

    return (A_right - A_left) / total

def detect_spikes(smoothed_spike_prob: np.ndarray = None,
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
    # Find the valid region (non-NaN) values)
    valid_mask = ~np.isnan(smoothed_spike_prob)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return {}, []
    
    # Get the valid region bounds
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1
    
    # Extract the valid portion
    valid_spike_prob = smoothed_spike_prob[start_idx:end_idx]
    
    # Detect peaks in the valid region
    peaks, _ = find_peaks(valid_spike_prob) if peaks is None else (peaks, None)
    if peaks.size == 0:
        return {}, []

    prominences, left_bases, right_bases = peak_prominences(
        valid_spike_prob, peaks
    )

    spike_data: Dict[int, Dict] = {}
    num_peaks = len(peaks)
    spike_keys = []
    inference_list = []
    for i, peak in enumerate(peaks):
        # Adjust indices back to original array coordinates
        absolute_peak = int(peak + start_idx)
        left_base = int(left_bases[i] + start_idx)
        right_base = int(right_bases[i] + start_idx)

        large_window = valid_spike_prob[left_bases[i]:right_bases[i]]
        spike_prom = smoothed_spike_prob[absolute_peak] - smoothed_spike_prob[left_base]

        # Determine bounds for the smaller, inter-peak window (in valid region coordinates)
        prev_peak = peaks[i - 1] if i > 0 else 0
        next_peak = peaks[i + 1] if i < num_peaks - 1 else len(valid_spike_prob)

        prev_min = _compute_min_between(
            valid_spike_prob, prev_peak, peak
        )
        next_min = _compute_min_between(
            valid_spike_prob, peak, next_peak
        )

        # Ensure the small window is non-empty and ordered.
        if next_min <= prev_min:
            next_min = prev_min + 1 if prev_min + 1 < len(valid_spike_prob) else len(valid_spike_prob)

        small_window = valid_spike_prob[prev_min:next_min]
        
        # Adjust small window bounds to absolute coordinates
        absolute_prev_min = int(prev_min + start_idx)
        absolute_next_min = int(next_min + start_idx)

        features = {
                "spike_prom": float(spike_prom),
                "isolation": int(len(large_window)),
                "distance": int(len(small_window)),
                "iso_skew": float(skew(large_window)) if large_window.size else 0.0,
                "dist_skew": float(skew(small_window)) if small_window.size else 0.0,
                "iso_aai_sum": float(area_asymmetry(large_window, peak - left_base)),
                "dist_aai_sum": float(area_asymmetry(small_window, peak - absolute_prev_min)),
                "iso_aai_trapz": float(area_asymmetry_trapz(large_window, zero_value=peak - left_base)),
                "dist_aai_trapz": float(area_asymmetry_trapz(small_window, zero_value=peak - absolute_prev_min))
            }
        spike_keys.append(absolute_peak if roi_idx is None else f"{roi_idx}_{absolute_peak}")
        if mode == "inference":
            inference_list.append(features)
            continue
            
        spike_data[absolute_peak] = {
            "windows": {'large_window' : {'window_values' : large_window, 'bounds': (left_base, right_base)},
                        'small_window' : {'window_values' : small_window, 'bounds': (absolute_prev_min, absolute_next_min)}},
            "features": features,
            "label": -1,  
        } 
    
      
    
    if mode == "inference":
        return inference_list, spike_keys

    
    return spike_data, spike_keys


def extract_spike_features(roi_dict: Dict, max_rois: Optional[int] = None) -> Tuple[Dict, list]:
    """
    Iterate through ROI data and collect spike features for labeled-good ROIs (label == 1).
    
    Args:
        roi_dict: Dictionary of ROI data
        max_rois: Maximum number of good ROIs to process (None = all)
    
    Returns: (roi_dict_with_spikes, all_roi_spike_keys)
    """
    all_roi_spike_keys = []
    processed_count = 0
    
    for roi_key, roi_data in roi_dict.items():
        if roi_data.get("label", -1) != 1:
            continue
        
        # Check if we've reached the max ROI limit
        if max_rois is not None and processed_count >= max_rois:
            print(f"✓ Reached max_rois limit ({max_rois}), stopping spike extraction")
            break

        smoothed_traces = roi_data.get("smoothed_traces", [])
        if len(smoothed_traces) < 2:
            continue

        smoothed_spike_prob = np.asarray(smoothed_traces[1])
        roi_spike_data, spike_keys = detect_spikes(smoothed_spike_prob)
        
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

    print(f"✓ Processed {processed_count} good ROIs, found {len(all_roi_spike_keys)} spikes")
    return roi_dict, all_roi_spike_keys


def main(input_path: str, output_path: Optional[str] = None, max_rois: Optional[int] = None) -> Dict[str, Dict[int, Dict]]:
    roi_dict = load_roi_data(input_path)
    
    
    roi_dict_spikes, all_roi_spike_keys = extract_spike_features(roi_dict, max_rois=max_rois)
    
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
    parser.add_argument("--input_path", help="Path to .npy file containing ROI dictionary.", default="training_data/roi_filtering/all_roi_features.npy")
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
