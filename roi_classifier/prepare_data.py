"""
Prepare ROI features from Suite2p fluorescence data.

Extracts comprehensive ROI features from fluorescence traces and saves to .npy file.
Supports both processing raw videos and updating existing feature files.
"""
import argparse
from pathlib import Path
from typing import Any
import numpy as np
from scipy.ndimage import gaussian_filter1d

import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from classifier_pipeline.verbose_utils import print_data_summary
from utils.model_utils.rois import compute_roi_features
from utils.preprocessing import normalize_minmax
from utils.label_utils import get_label_value, create_label_dict, normalize_label_format, compute_data_summary
from utils.io_utils import create_backup


# =============================================================================
# ROI Processing
# =============================================================================

def process_roi(smoothed_f_trace: np.ndarray, raw_trace: np.ndarray) -> dict:
    """
    Process a single ROI and extract features.
    
    Parameters
    ----------
    smoothed_f_trace : np.ndarray
        Smoothed fluorescence trace (1D)
    raw_trace : np.ndarray
        Raw fluorescence trace (1D)
    
    Returns
    -------
    roi_data : dict
        Dictionary with traces, features, label, and spikes
    """
    features, validity = compute_roi_features(smoothed_f_trace)
    
    # Auto-label as bad if critical features are invalid
    critical_valid = validity.get('valid_deriv_skew', True) and validity.get('valid_prom', True)
    label = create_label_dict(0, 'auto') if not critical_valid else create_label_dict(-1, 'unlabeled')
    
    return {
        'smoothed_trace': smoothed_f_trace,
        'raw_trace': raw_trace,
        'features': features,
        'label': label,
        'spikes': {}
    }


def process_video(video_path: Path, sigma: float = 4.0) -> list[tuple[str, dict]]:
    """
    Process a video and extract ROI features.
    
    Parameters
    ----------
    video_path : Path
        Path to video directory containing Suite2p outputs
    sigma : float, optional
        Gaussian smoothing sigma, by default 4.0
    
    Returns
    -------
    video_rois : list[tuple[str, dict]]
        List of (roi_key, roi_data) tuples
    """
    fluorescence_file = video_path / 'suite2p' / 'plane0' / 'F.npy'
    scaled_f_file = video_path / 'suite2p' / 'plane0' / 'F_minmax.npy'
    
    if not fluorescence_file.exists():
        print(f"Warning: F.npy not found in {video_path}")
        return []
    
    f = np.load(fluorescence_file)
    if scaled_f_file.exists():
        scaled_f = np.load(scaled_f_file)
    else:
        scaled_f = normalize_minmax(f)
        np.save(scaled_f_file, scaled_f)
    smoothed_f = gaussian_filter1d(scaled_f, sigma=sigma, axis=1)
    
    return [
        (f"{video_path.name}_{idx}", process_roi(smoothed_f[idx], f[idx]))
        for idx in range(f.shape[0])
    ]


# =============================================================================
# Update Existing Data
# =============================================================================

def update_roi_features(roi_dict: dict[str, dict[str, Any]], verbose: bool = True) -> dict:
    """
    Update ROI features while preserving labels and spike data.
    
    Parameters
    ----------
    roi_dict : dict[str, dict[str, Any]]
        Existing ROI dictionary
    verbose : bool, optional
        Whether to print summary, by default True
    
    Returns
    -------
    updated_dict : dict
        Updated ROI dictionary with recomputed features
    """
    updated_dict = {}
    stats = {'processed': 0, 'labels_preserved': 0, 'spikes_preserved': 0}
    
    for roi_key, roi_data in roi_dict.items():
        smoothed_trace = roi_data.get('smoothed_trace', np.array([]))
        if smoothed_trace.size == 0:
            print(f"Warning: Skipping {roi_key} - missing smoothed trace")
            continue
        
        features, _ = compute_roi_features(smoothed_trace)
        label = normalize_label_format(roi_data.get('label', -1))
        spikes = roi_data.get('spikes', {})
        
        # Track preserved data
        if label['value'] in [0, 1] and label['source'] == 'manual':
            stats['labels_preserved'] += 1
        stats['spikes_preserved'] += len(spikes)
        
        updated_dict[roi_key] = {
            'smoothed_trace': smoothed_trace,
            'raw_trace': roi_data.get('raw_trace'),
            'features': features,
            'label': label,
            'spikes': spikes
        }
        stats['processed'] += 1
    
    if verbose:
        print(f"\nUpdated {stats['processed']} ROIs")
        print(f"  - Preserved {stats['labels_preserved']} manual labels")
        print(f"  - Preserved {stats['spikes_preserved']} spikes")
    
    return updated_dict


# =============================================================================
# Main Pipeline Functions
# =============================================================================

def process_dataset(dataset_root: Path, verbose: bool = True) -> dict:
    """
    Process all videos in a dataset directory.

    Recursively searches for directories containing Suite2p outputs
    (``suite2p/plane0/F.npy``).

    Parameters
    ----------
    dataset_root : Path
        Root directory containing video folders (may be nested)
    verbose : bool, optional
        Whether to print progress, by default True

    Returns
    -------
    roi_dict : dict
        Dictionary of all ROIs from all videos
    """
    # Find every directory that has suite2p output
    video_paths = sorted(
        {p.parent.parent.parent for p in dataset_root.rglob("suite2p/plane0/F.npy")}
    )
    if verbose:
        print(f"Found {len(video_paths)} video directories under {dataset_root}")

    all_rois = []
    for video_path in video_paths:
        video_rois = process_video(video_path)
        if video_rois:
            all_rois.extend(video_rois)
            if verbose:
                print(f"  Processed {video_path.name}: {len(video_rois)} ROIs")

    return dict(all_rois)


def prepare_roi_data(
    dataset_root: Path = None,
    input_file: Path = None,
    output_file: Path = None,
    update: bool = False,
    backup: bool = True,
    verbose: bool = True
) -> dict:
    """
    Prepare ROI data by processing videos or updating existing file.
    
    Parameters
    ----------
    dataset_root : Path, optional
        Root directory for raw video processing
    input_file : Path, optional
        Input file for update mode
    output_file : Path, optional
        Output file path
    update : bool, optional
        Whether to update existing file, by default False
    backup : bool, optional
        Whether to create backup in update mode, by default True
    verbose : bool, optional
        Whether to print progress, by default True
    
    Returns
    -------
    roi_dict : dict
        Processed ROI dictionary
    
    Raises
    ------
    ValueError
        If required paths are not provided
    """
    if update:
        if input_file is None or not input_file.exists():
            raise ValueError(f"Input file required for update mode: {input_file}")
        
        if backup:
            backup_path = create_backup(input_file)
            if verbose:
                print(f"Created backup: {backup_path}")
        
        roi_dict = np.load(input_file, allow_pickle=True).item()
        
        if verbose:
            summary = compute_data_summary(roi_dict)
            print_data_summary(summary)
        
        roi_dict = update_roi_features(roi_dict, verbose=verbose)
    else:
        if dataset_root is None or not dataset_root.exists():
            raise ValueError(f"Dataset root required for processing: {dataset_root}")
        
        roi_dict = process_dataset(dataset_root, verbose=verbose)
    
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, roi_dict)
        if verbose:
            print(f"\nSaved {len(roi_dict)} ROIs to {output_file}")
    
    return roi_dict



# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare ROI features from Suite2p data')
    parser.add_argument('--dataset_root', type=str, default=r"C:\Users\mzinn1\Desktop\Datasets",
                        help='Root directory containing video folders')
    parser.add_argument('--input_file', type=str, 
                        default='training_data/roi_filtering/all_roi_features.npy',
                        help='Input file for update mode')
    parser.add_argument('--output_file', type=str,
                        default='training_data/roi_filtering/all_roi_features.npy',
                        help='Output file path')
    parser.add_argument('--update', action='store_true',
                        help='Update existing features instead of processing raw videos')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip backup creation in update mode')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress output')
    args = parser.parse_args()
    
    prepare_roi_data(
        dataset_root=Path(args.dataset_root),
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        update=args.update,
        backup=not args.no_backup,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()