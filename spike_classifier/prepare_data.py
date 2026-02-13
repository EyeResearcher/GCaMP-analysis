import argparse
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks
from utils.model_utils.spikes import get_all_spike_features
from scipy.stats import skew
from classifier_pipeline.utils import create_label_dict, get_label_value
from classifier_pipeline.io_utils import load_roi_data, save_roi_data



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
    peaks, props = find_peaks(smoothed_f, distance=30) if peaks is None else (peaks, None)
    if peaks.size == 0:
        return {}, []


    spike_data, spike_keys = get_all_spike_features(
        smoothed_f, peaks, props, mode=mode, roi_idx=roi_idx
    )   
    return spike_data, spike_keys


def parallel_detect_spikes(args):
    smoothed_f, roi_idx = args
    spike_data, spike_keys = define_candidate_fluor_events(smoothed_f, roi_idx=roi_idx)
    return spike_data, spike_keys

def validate_roi_label(roi_key: str, roi_data: Dict) -> bool:
    """
    Validate that an ROI has a good label (value == 1).
    
    Parameters
    ----------
    roi_key : str
        Key identifying the ROI.
    roi_data : dict
        ROI data dictionary containing a 'label' field.
    
    Returns
    -------
    is_good : bool
        True if ROI label value is 1, False otherwise.
    
    Raises
    ------
    ValueError
        If ROI is missing the 'label' field.
    """
    roi_label = roi_data.get('label', None)
    if roi_label is None:
        raise ValueError(f"ROI {roi_key} is missing 'label' data.")
    
    return get_label_value(roi_label) == 1

def process_rois(roi_dict: Dict[str, Dict], max_rois: Optional[int] = None) -> Dict:
    """
    Iterate through ROI data and collect spike features for labeled-good ROIs (label == 1).
    
    Parameters
    ----------
    roi_dict : dict
        Dictionary of ROI data.
    max_rois : int, optional
        Maximum number of good ROIs to process (None = all).
    
    Returns
    -------
    roi_dict : dict
        Updated ROI dictionary with spike data added in-place.
    """
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
        
        # Preserve existing labels (e.g. from prior annotation sessions)
        existing_spikes = roi_data.get('spikes', {})
        for spike_idx in spike_keys:
            if spike_idx in existing_spikes:
                existing_label = existing_spikes[spike_idx].get('label', None)
                if existing_label is not None:
                    roi_spike_data[spike_idx]['label'] = existing_label
        
        roi_dict[roi_key]['spikes'] = roi_spike_data
        processed_count += 1

    print(f"Processed {processed_count} good ROIs")
    print(f"Skipped {bad_roi} bad ROIs")
    return roi_dict


def main(input_path: str,
         output_path: Optional[str] = None, 
         max_rois: Optional[int] = None) -> Dict[str, Dict[int, Dict]]:
    from pathlib import Path
    
    roi_dict = load_roi_data(Path(input_path), verbose=True)
    roi_dict = process_rois(roi_dict, max_rois=max_rois)
    
    save_path = Path(output_path) if output_path else Path(input_path)
    save_roi_data(roi_dict, save_path, verbose=True)
    
    return roi_dict


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
