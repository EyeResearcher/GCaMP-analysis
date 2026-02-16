import argparse
from typing import Dict, Optional, Tuple, Any
import numpy as np
from pathlib import Path
from scipy.stats import skew
from utils.label_utils import (
    create_label_dict, get_label_value, get_label_source,
    normalize_spike_label, preserve_existing_label, compute_data_summary
)
from spike_processing.detector import define_candidate_fluor_events
from classifier_pipeline.io_utils import load_roi_data, save_roi_data
from classifier_pipeline.verbose_utils import print_data_summary


# =============================================================================
# Spike Detection
# =============================================================================

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
    
    Preserves manually-annotated spike labels from prior sessions, normalizing
    all labels to the standardized ``{'value': int, 'source': str}`` format.
    
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
    labels_preserved = 0

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
        
        # Preserve existing manual labels, normalizing to dict format
        existing_spikes = roi_data.get('spikes', {})
        for spike_idx in spike_keys:
            preserved = preserve_existing_label(
                existing_spikes, spike_idx, roi_spike_data[spike_idx]['label']
            )
            roi_spike_data[spike_idx]['label'] = preserved
            if get_label_value(preserved) != -1:
                labels_preserved += 1
        
        roi_dict[roi_key]['spikes'] = roi_spike_data
        processed_count += 1

    print(f"Processed {processed_count} good ROIs")
    print(f"Skipped {bad_roi} bad ROIs")
    print(f"Preserved {labels_preserved} existing labels")
    return roi_dict


def prepare_spike_data(input_path: str,
         output_path: Optional[str] = None, 
         max_rois: Optional[int] = None,
         verbose = True) -> Dict[str, Dict[int, Dict]]:
    
    roi_dict = load_roi_data(Path(input_path), verbose=verbose)
    roi_dict = process_rois(roi_dict, max_rois=max_rois)
    
    save_path = Path(output_path) if output_path else Path(input_path)
    save_roi_data(roi_dict, save_path, verbose=verbose)
    if verbose:
        s = compute_data_summary(roi_dict, level="roi")
        print_data_summary(s)
    
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()
    roi_dict = prepare_spike_data(input_path=args.input_path,
                                output_path=args.output_path,
                                max_rois=args.max_rois, 
                                verbose=args.verbose)
