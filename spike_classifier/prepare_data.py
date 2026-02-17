import argparse
from typing import Dict, Optional, Tuple, Any
import numpy as np
from pathlib import Path
from utils.label_utils import ( get_label_value, validate_roi_label, 
                               preserve_existing_label, compute_data_summary)

from gcamp_analysis.spike_processing.detector import get_f_events
from classifier_pipeline.io_utils import load_roi_data, save_roi_data
from classifier_pipeline.verbose_utils import print_data_summary


# =============================================================================
# Spike Detection
# =============================================================================

def process_rois(roi_dict: Dict[str, Dict], max_rois: Optional[int] = None, fs: float = 30.0) -> Dict:
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
    fs : float
        Frame rate in Hz, used for peak detection distance.
    
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
        too_many = max_rois is not None and processed_count >= max_rois

        if not valid or too_many:
            bad_roi += 1
            continue 

        if too_many:
            print(f"Reached max_rois limit ({max_rois}), stopping spike extraction")
            break

        sm_f = np.asarray(roi_data.get("smoothed_trace", None))

        roi_spike_data, spike_keys = get_f_events(sm_f, fs=fs)
        existing_spikes = roi_data.get('spikes', {})

        for spike_idx in spike_keys:
            label = roi_spike_data[spike_idx].get('label')
            preserved = preserve_existing_label(existing_spikes, spike_idx, label)
            roi_spike_data[spike_idx]['label'] = preserved
            if get_label_value(preserved) != -1:
                labels_preserved += 1
        
        roi_dict[roi_key]['spikes'] = roi_spike_data
        processed_count += 1

    return roi_dict


def prepare_spike_data(input_path: str,
         output_path: Optional[str] = None, 
         max_rois: Optional[int] = None,
         fs: float = 30.0,
         verbose = True) -> Dict[str, Dict[int, Dict]]:
    
    roi_dict = load_roi_data(Path(input_path), verbose=verbose)
    roi_dict = process_rois(roi_dict, max_rois=max_rois, fs=fs)
    
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
        default="data/all_roi_features.npy")
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
    parser.add_argument(
        "--fs",
        type=float,
        default=30.0,
        help="Frame rate in Hz (default: 30.0). Controls minimum inter-peak distance.",
    )
    args = parser.parse_args()
    roi_dict = prepare_spike_data(input_path=args.input_path,
                                output_path=args.output_path,
                                max_rois=args.max_rois,
                                fs=args.fs, 
                                verbose=args.verbose)
