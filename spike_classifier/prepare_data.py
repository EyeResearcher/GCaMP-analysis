import argparse
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks
from utils.model_utils.spikes import get_all_spike_features
from scipy.stats import skew
import pandas as pd



def load_roi_data(file_path: str) -> Dict[str, Dict]:
    """
    Load ROI data from a .npy file.
    """
    roi_data = np.load(file_path, allow_pickle=True).item()
    return roi_data



def define_candidate_fluor_events(smoothed_f: np.ndarray = None,
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


    spike_data, spike_keys = get_all_spike_features(
        smoothed_f, peaks, mode=mode, roi_idx=roi_idx
    )   
    return spike_data, spike_keys


def parallel_detect_spikes(args):
    smoothed_f, roi_idx = args
    spike_data, spike_keys = define_candidate_fluor_events(smoothed_f, roi_idx=roi_idx)
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

def process_rois(roi_dict: Dict[str, Dict], max_rois: Optional[int] = None) -> Tuple[Dict, list]:
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


        roi_spike_data, spike_keys = define_candidate_fluor_events(smoothed_f_trace)
        
        existing_spikes = roi_data.get('spikes', {})
        for spike_idx in spike_keys:
            if spike_idx in existing_spikes:
                roi_spike_data[spike_idx]['label'] = existing_spikes[spike_idx]['label']
        
        roi_spike_keys = [(f"{roi_key}-{spike_idx}", roi_spike_data[spike_idx]['label']) for spike_idx in spike_keys]
        all_roi_spike_keys.extend(roi_spike_keys)
        roi_dict[roi_key]['spikes'] = roi_spike_data
        processed_count += 1

    print(f"Processed {processed_count} good ROIs, found {len(all_roi_spike_keys)} spikes")
    print(f"Skipped {bad_roi} bad ROIs")
    return roi_dict, all_roi_spike_keys


def main(input_path: str,
         output_path: Optional[str] = None, 
         max_rois: Optional[int] = None) -> Dict[str, Dict[int, Dict]]:
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
