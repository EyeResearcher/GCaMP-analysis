from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

def train(model, X_train_var : np.ndarray, y_train : np.ndarray, X_test_var: np.ndarray, y_test: np.ndarray, feature_names, model_name, transform):
    cv_scores = cross_val_score(model, X_train_var, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    cv_acc = cv_scores.mean()
    cv_std = cv_scores.std()
  
    model.fit(X_train_var, y_train)
    test_acc = model.score(X_test_var, y_test)
    y_pred_proba = model.predict_proba(X_test_var)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    importance_df = get_feature_importance(model, feature_names)
    
    result = {
        'model': model_name,
        'transform': transform,
        'n_features' : len(feature_names),
        'cv_acc': cv_acc,
        'cv_std': cv_std,
        'test_acc': test_acc,
        'roc_auc': roc_auc,
        'model_instance': model,
        'model_type': type(model),
        'X_train': X_train_var,
        'X_test': X_test_var,
        'y_train': y_train,
        'y_test': y_test,
        'features': {name: transform for name in feature_names},
        'feature_importance': importance_df
    }
    return result
def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Works with Random Forest (feature_importances_) and 
    Logistic Regression/SVM (coef_).
    
    Returns
    -------
    importance_df : DataFrame
        DataFrame with features sorted by importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df
def get_model(model_class : RandomForestClassifier | LogisticRegression | SVC, **hp_kwargs) -> RandomForestClassifier | LogisticRegression | SVC:
    """Create model instance based on class and hyperparameters.
     Args:
        model_class: sklearn model class 
            - RandomForestClassifier, LogisticRegression, SVM
        **hp_kwargs: hyperparameters for the model
            - For Random Forest: n_estimators, max_depth, random_state, class_weight, n_jobs
            - For Logistic Regression: random_state, max_iter, class_weight
            - For SVM: C, kernel, gamma, random_state
    Returns:
        model: instance of the specified model class
    """
    model = model_class(**hp_kwargs)
    return model

def merge_dicts(dict1: dict, dict2: dict) -> dict:
    merged = {}
    for key in dict1.keys():
        merged[key] = (dict1[key], dict2[key])
    return merged

def create_label_dict(value: int, source: str = 'manual') -> dict:
    """Create a standardized label dictionary."""
    return {'value': value, 'source': source}

def get_keys(roi_dict : dict[str,dict[str, Any]], 
             unlabeled_only : bool = False, 
             labeled_only : bool = False,
             verbose: bool = True) -> list[str]:
    """Return list of ROI keys based on labeling criteria."""
    if unlabeled_only and labeled_only:
        raise ValueError("Cannot set both unlabeled_only and labeled_only to True.")
    if unlabeled_only:
        keys = [k for k in roi_dict.keys() if get_label_value(roi_dict[k]['label']) == -1]
        if verbose:
            print(f"Found {len(keys)} unlabeled ROIs out of {len(roi_dict)}.")
    if labeled_only:
        keys = [k for k in roi_dict.keys() if get_label_value(roi_dict[k]['label']) != -1]
        if verbose:
            print(f"Found {len(keys)} labeled ROIs out of {len(roi_dict)}.")
    else:
        keys = list(roi_dict.keys())
        if verbose :
            print(f"Returning all {len(roi_dict)} ROI keys.")
    if len(keys) == 0:
        raise ValueError("No ROIs match the specified filtering criteria.")
    return keys

def get_label_value(label):
    """Extract numeric label value from either dict or int format."""
    if isinstance(label, dict):
        return label.get('value', -1)
    return label

def get_label_source(label):
    """Extract label source from either dict or int format."""
    if isinstance(label, dict):
        return label.get('source', 'unknown')
    return 'unknown'