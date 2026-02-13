import datetime
import json
from pathlib import Path
from utils.label_utils import get_label_source, get_label_value
from utils.io_utils import load_config          # noqa: F401  (re-export)
import joblib
import numpy as np
import yaml
from .optimize import OptimizationResults
import pandas as pd


def load_labeled_roi_data(
    roi_dict: dict[str, dict[str, np.ndarray | dict]],
    manual_only: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load labeled ROI data as DataFrame.
    
    Parameters
    ----------
    roi_dict : dict
        Dictionary containing ROI data
    manual_only : bool, optional
        If True, only use manually labeled ROIs, by default True
    
    Returns
    -------
    X : pd.DataFrame
        Feature matrix with named columns
    y : pd.Series
        Labels
        
    Raises
    ------
    ValueError
        If no labeled data found
    """
    rows = []
    
    for roi_key, roi_data in roi_dict.items():
        label_value = get_label_value(roi_data['label'])
        label_source = get_label_source(roi_data['label'])
        
        if label_value == -1:
            continue
        if manual_only and label_source != 'manual':
            continue

        row = roi_data['features'].copy()
        row['label'] = label_value
        row['roi_key'] = roi_key
        rows.append(row)
    
    if len(rows) == 0:
        raise ValueError("No labeled data found! Please annotate some ROIs first.")
    
    df = pd.DataFrame(rows)
    
    # Separate features, labels, and keys
    y = df['label'].astype(int)
    X = df.drop(columns=['label', 'roi_key'])
    
    return X, y


def load_labeled_spike_data(
    npy_dict: dict[str, dict[str, np.ndarray | dict]],
    manual_only: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load labeled spike data as DataFrame.
    
    Parameters
    ----------
    npy_dict : dict
        Dictionary of ROI data with nested spikes
    manual_only : bool, optional
        If True, only include manually labeled spikes, by default True
    
    Returns
    -------
    X : pd.DataFrame
        Feature matrix with named columns
    y : pd.Series
        Labels
        
    Raises
    ------
    ValueError
        If no labeled spikes found
    """
    rows = []
    
    for roi_key, roi_data in npy_dict.items():
        spikes = roi_data.get('spikes', {})
        if not isinstance(spikes, dict):
            continue
        
        for spike_idx, spike_data in spikes.items():
            label = spike_data.get('label', {'value': -1, 'source': 'unlabeled'})
            label_value = get_label_value(label)
            label_source = get_label_source(label)
            
            if label_value == -1:
                continue
            if manual_only and label_source != 'manual':
                continue
            
            features = spike_data.get('features', {})
            if not features:
                continue
            
            row = features.copy()
            row['label'] = label_value
            row['spike_key'] = f"{roi_key}-{spike_idx}"
            rows.append(row)
    
    if len(rows) == 0:
        raise ValueError("No labeled spikes found! Please annotate some spikes first.")
    
    df = pd.DataFrame(rows)
    
    y = df['label'].astype(int)
    X = df.drop(columns=['label', 'spike_key'])
    
    return X, y


def load_roi_data(npy_path: Path, verbose: bool = True) -> dict:
    """Load ROI data from .npy file."""
    if not npy_path.exists():
        raise FileNotFoundError(f"ROI data file not found: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()
    if verbose:
        print(f"Loaded {len(data)} ROIs from {npy_path}")
    return data


def save_roi_data(npy_dict: dict, npy_path: Path, verbose: bool = True) -> None:
    """Save ROI data to .npy file."""
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, npy_dict, allow_pickle=True)
    if verbose:
        print(f"Saved to {npy_path}")


def save_results(results: OptimizationResults, output_path: Path, verbose: bool = True) -> None:
    """Save optimization results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results.to_dict(include_model=True), f, indent=2)
        
    if verbose:
        print(f"Saved results to {output_path}")


def save_model(model, output_path: Path, verbose: bool = True) -> None:
    """Save trained model to joblib file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    
    if verbose:
        print(f"Saved model to {output_path}")


def save_optimization_outputs(
    results: OptimizationResults, 
    output_dir: Path, 
    model_name: str,
    verbose: bool = True
) -> tuple[Path, Path]:
    """
    Save both model and results JSON.
    
    Parameters
    ----------
    results : OptimizationResults
        Results object containing model and metrics
    output_dir : Path
        Directory to save outputs
    model_name : str
        Name of model being saved
    verbose : bool, optional
        Whether to print confirmation, by default True
        
    Returns
    -------
    model_path : Path
        Path to saved model
    results_path : Path
        Path to saved results JSON
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.joblib"
    results_path = output_dir / f"{model_name}_results.json"
    
    save_model(results.model, model_path, verbose=verbose)
    save_results(results, results_path, verbose=verbose)
    
    return model_path, results_path