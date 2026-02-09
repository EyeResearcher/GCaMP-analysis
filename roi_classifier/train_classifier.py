"""
Train ROI classifiers (Random Forest and Logistic Regression).

Tests different feature transformations and compares model performance.
Only trains on manually labeled ROIs.
"""
import argparse
import json
from datetime import datetime
from typing import Any
from xml.parsers.expat import model

from sklearn.svm import SVC
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
from classifier_pipeline.datasets import ClassifierDataset, DataSplit
from classifier_pipeline.utils import get_label_value, get_label_source, merge_dicts, get_model, train, get_feature_importance
from classifier_pipeline.verbose_utils import print_dataset_summary, print_split_summary, print_tuned_summary
from classifier_pipeline.optimize import test_configurations, test_feature_selection, tune_hyperparameters
from classifier_pipeline.io_utils import load_roi_data, load_labeled_roi_data, save_model_and_config








    






# =============================================================================
# Model Saving
# =============================================================================




# =============================================================================
# Main Entry Point
# =============================================================================

def train_roi_classifier(
    data_path: Path,
    output_dir: Path = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = None,
    manual_only: bool = True,
    save_model: bool = True,
    hp_config: dict = None,
    verbose: bool = True
) -> dict:
    """
    Train ROI classifier with feature transformation testing.
    
    This is the main entry point for training. Tests Random Forest and 
    Logistic Regression with different feature transformations and 
    feature subsets.
    
    Parameters
    ----------
    data_path : Path
        Path to ROI data .npy file
    output_dir : Path, optional
        Directory to save models. If None, models won't be saved.
    test_size : float
        Fraction of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    n_estimators : int
        Number of trees for Random Forest (default: 100)
    max_depth : int or None
        Max depth for Random Forest (default: None = unlimited)
    manual_only : bool
        Only use manually labeled ROIs (default: True)
    save_model : bool
        Whether to save the best model (default: True)
    verbose : bool
        Print detailed output (default: True)
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'best_model': The trained best model instance
        - 'config': Configuration dict with model settings
        - 'results_df': DataFrame with all results
        - 'feature_names': List of feature names
        - 'model_path': Path to saved model (if save_model=True)
        - 'config_path': Path to saved config (if save_model=True)
    """
    
    roi_dict = load_roi_data(data_path, verbose=False)
    
    X, y, feature_names, roi_keys = load_labeled_roi_data(roi_dict, manual_only=manual_only)
    
    if verbose:
        print_dataset_summary(feature_names, y, manual_only=manual_only)
   
    splits = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    dataset = ClassifierDataset.build(splits, feature_names)
    if verbose:
        print_split_summary(dataset.y_train.raw, dataset.y_test.raw)

    best_config = test_configurations(dataset, verbose=verbose)
    
    best_config_tuned = tune_hyperparameters(best_config, hp_config)
    
    if verbose:
        print_tuned_summary(best_config_tuned)
    
    model_path, config_path = save_model_and_config(
            tuned_config=best_config_tuned,
            feature_names=feature_names,
            output_dir=output_dir,
            n_train=len(dataset.y_train.raw),
            n_test=len(dataset.y_test.raw),
            manual_only=manual_only
        )
    best_config_tuned['model_path'] = model_path
    best_config_tuned['config_path'] = config_path
    return best_config_tuned


def main():
    parser = argparse.ArgumentParser(description="Train ROI classifier with feature transformation testing")
    parser.add_argument("--data_path", type=str, 
                       default="training_data/roi_filtering/all_roi_features.npy",
                       help="Path to ROI data file")
    parser.add_argument("--output_dir", type=str,
                       default="roi_classifier/models",
                       help="Directory to save models")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Fraction of data for testing")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--n_estimators", type=int, default=100,
                       help="Number of trees for Random Forest")
    parser.add_argument("--max_depth", type=int, default=None,
                       help="Max depth for Random Forest (None=unlimited)")
    parser.add_argument("--include_auto", action='store_true',
                       help="Include auto-labeled ROIs (default: manual only)")
    args = parser.parse_args()

    train_roi_classifier(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        manual_only=not args.include_auto,
        save_model=True,
        verbose=True
    )


if __name__ == "__main__":
    main()
