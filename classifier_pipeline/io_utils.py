

import datetime
import json
from pathlib import Path
from classifier_pipeline.utils import get_label_source, get_label_value
import joblib
import numpy as np
import yaml
from .optimize import OptimizationResults

def load_config(path: Path | str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_labeled_roi_data(roi_dict: dict[str, dict[str, np.ndarray | dict ]],
                        manual_only: bool = True) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Load ROI data and filter for labeled ROIs.
    
    Parameters
    ----------
    roi_dict : dict
        Dictionary containing ROI data
    manual_only : bool
        If True, only use manually labeled ROIs (default: True)
    
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    feature_names : list
        Names of features
    roi_keys : list
        ROI keys corresponding to each sample
    """
    
    rows = []
    roi_keys_list = []
    
    for roi_key, roi_data in roi_dict.items():
        label_value = get_label_value(roi_data['label'])
        label_source = get_label_source(roi_data['label'])
        
        if (manual_only and label_source != 'manual') or label_value == -1:
            continue

        features = list(roi_data['features'].values())
        rows.append(features + [label_value])
        roi_keys_list.append(roi_key)
    
    if len(rows) == 0:
        raise ValueError("No labeled data found! Please annotate some ROIs first.")
    
    data_array = np.array(rows)
    X = data_array[:, :-1]  
    y = data_array[:, -1]   
    
    first_roi = next(iter(roi_dict.values()))
    feature_names = list(first_roi['features'].keys())
    
    return X, y, feature_names, roi_keys_list

def load_roi_data(npy_path: Path, verbose: bool = True,
                 manual_only: bool = True) -> dict:
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
    """
    Save optimization results to a JSON file.
    
    Parameters
    ----------
    results : OptimizationResults
        Results object to save
    output_path : Path
        Path to save JSON file
    verbose : bool, optional
        Whether to print confirmation, by default True
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results.to_dict(include_model=True), f, indent=2)
        
    if verbose:
        print(f"Saved results to {output_path}")


def save_model(model, output_path: Path, verbose: bool = True) -> None:
    """
    Save trained model to joblib file.
    
    Parameters
    ----------
    model : sklearn model
        Trained model to save
    output_path : Path
        Path to save model
    verbose : bool, optional
        Whether to print confirmation, by default True
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    
    if verbose:
        print(f"Saved model to {output_path}")


def save_optimization_outputs(results: OptimizationResults, output_dir: Path,
                              model_name : str, verbose: bool = True,
                              overwrite: bool = False) -> tuple[Path, Path]:
    """
    Save both model and results JSON.
    
    Parameters
    ----------
    results : OptimizationResults
        Results object containing model and metrics
    output_dir : Path
        Directory to save outputs
    verbose : bool, optional
        Whether to print confirmation, by default True
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
        
    Returns
    -------
    model_path : Path
        Path to saved model
    results_path : Path
        Path to saved results JSON
    """
    from datetime import datetime
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = f"{model_name}.joblib"
    results_file = f"{model_name}_results.json"
    
    if not overwrite:
        model_file = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        results_file = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.json"

    model_path = output_dir / model_file
    results_path = output_dir / results_file
    
    save_model(results.model, model_path, verbose=verbose)
    save_results(results, results_path, verbose=verbose)
    
    return model_path, results_path