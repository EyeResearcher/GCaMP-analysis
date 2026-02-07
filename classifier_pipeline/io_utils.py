

import datetime
import json
from pathlib import Path
from classifier_pipeline.utils import get_label_source, get_label_value
import joblib
import numpy as np

def load_labeled_roi_data(roi_dict: dict[str, dict[str, np.ndarray | dict ]], manual_only: bool = True):
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
    X = data_array[:, :-1]  # All columns except last
    y = data_array[:, -1]   # Last column
    
    # Get feature names from first ROI
    first_roi = next(iter(roi_dict.values()))
    feature_names = list(first_roi['features'].keys())
    
    return X, y, feature_names, roi_keys_list
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

def save_model_and_config(tuned_config: dict, feature_names: list, 
                          output_dir: Path, n_train: int, n_test: int,
                          manual_only: bool, verbose=True) -> tuple[Path, Path]:
    """
    Save the tuned model and its configuration.
    
    Parameters
    ----------
    tuned_config : dict
        Dictionary returned by tune_hyperparameters
    feature_names : list
        All feature names
    output_dir : Path
        Directory to save outputs
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
    manual_only : bool
        Whether only manual labels were used
    
    Returns
    -------
    model_path : Path
        Path to saved model
    config_path : Path
        Path to saved config JSON
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = type(tuned_config['model']).__name__
    cm = tuned_config['confusion_matrix']
    
    model_path = output_dir / f"roi_classifier_{timestamp}.joblib"
    joblib.dump(tuned_config['model'], model_path)
    
    config = {
        'timestamp': timestamp,
        'model_type': model_name,
        'transform': tuned_config['transform'],
        'feature_names': feature_names,
        'selected_features': list(tuned_config['features'].keys()),
        'n_features': len(tuned_config['features']),
        'best_params': tuned_config['best_params'],
        'metrics': {
            'cv_accuracy': float(tuned_config['cv_acc']),
            'test_accuracy': float(tuned_config['test_acc']),
            'roc_auc': float(tuned_config['roc_auc']),
            'f1': float(tuned_config['f1']),
            'precision': float(tuned_config['precision']),
            'recall': float(tuned_config['recall'])
        },
        'confusion_matrix': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        },
        'data': {
            'n_train': n_train,
            'n_test': n_test,
            'manual_only': manual_only
        }
    }
    
    config_path = output_dir / f"roi_classifier_config_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    if verbose:
        print(f"\nModel saved to:  {model_path}")
        print(f"Config saved to: {config_path}")
    
    return model_path, config_path