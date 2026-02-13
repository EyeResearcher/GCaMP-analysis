from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.svm import SVC
import numpy as np
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from enum import Enum

def train_and_evaluate(model: RandomForestClassifier | LogisticRegression | SVC,
                       X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame, y_test: pd.Series,
                       metric: str = 'roc_auc') -> float:
    """
    Train model and return evaluation metric.
    
    Parameters
    ----------
    model : RandomForestClassifier | LogisticRegression | SVC
        Model instance to train
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame   
        Test features
    y_test : pd.Series
        Test labels
    metric : str, optional
        Evaluation metric to return, by default 'roc_auc'
        Options: 'roc_auc', 'accuracy', 'balanced_accuracy', 'f1'
        
    Returns
    -------
    score : float
        Score for the specified metric
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if metric == 'roc_auc':
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_proba)
    elif metric == 'accuracy':
        return model.score(X_test, y_test)
    elif metric == 'balanced_accuracy':
        return balanced_accuracy_score(y_test, y_pred)
    elif metric == 'f1':
        return f1_score(y_test, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
def get_feature_importance(model : RandomForestClassifier | LogisticRegression | SVC, feature_names: list) -> pd.DataFrame:
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

class ModelClass(Enum): 
    RF = RandomForestClassifier
    LR = LogisticRegression
    SVM = SVC

def get_model(model_class: str, **hp_kwargs) -> RandomForestClassifier | LogisticRegression | SVC:
    """
    Create model instance based on class and hyperparameters.
    
    Parameters
    ----------
    model_class : str
        Model type identifier:
        
        - 'RF': Random Forest
        - 'LR': Logistic Regression
        - 'SVM': Support Vector Machine
    **hp_kwargs : dict
        Hyperparameters for the model
        
    Returns
    -------
    model : RandomForestClassifier | LogisticRegression | SVC
        Instance of the specified model class
    """
    model_cls = ModelClass[model_class].value
    return model_cls(**hp_kwargs)

def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dictionaries with matching keys into tuples.
    
    Parameters
    ----------
    dict1 : dict
        First dictionary
    dict2 : dict
        Second dictionary (must have same keys as dict1)
        
    Returns
    -------
    merged : dict
        Dictionary with values as tuples (dict1[key], dict2[key])
    """
    merged = {}
    for key in dict1.keys():
        merged[key] = (dict1[key], dict2[key])
    return merged

def create_label_dict(value: int, source: str = 'manual') -> dict:
    """
    Create a standardized label dictionary.
    
    Parameters
    ----------
    value : int
        Label value (0, 1, or -1 for unlabeled)
    source : str, optional
        Source of the label, by default 'manual'
        
    Returns
    -------
    label : dict
        Dictionary with 'value' and 'source' keys
    """
    return {'value': value, 'source': source}


def get_label_value(label: dict | int) -> int:
    """
    Extract numeric label value from either dict or int format.
    
    Parameters
    ----------
    label : dict | int
        Label as dict with 'value' key or raw int
        
    Returns
    -------
    value : int
        Numeric label value, -1 if not found
    """
    if isinstance(label, dict):
        return label.get('value', -1)
    return label


def get_label_source(label: dict | int) -> str:
    """
    Extract label source from either dict or int format.
    
    Parameters
    ----------
    label : dict | int
        Label as dict with 'source' key or raw int
        
    Returns
    -------
    source : str
        Label source, 'unknown' if not found
    """
    if isinstance(label, dict):
        return label.get('source', 'unknown')
    return 'unknown'
def get_keys(roi_dict: dict[str, dict[str, Any]], 
             unlabeled_only: bool = False, 
             labeled_only: bool = False,
             verbose: bool = True) -> list[str]:
    """
    Return list of ROI keys based on labeling criteria.
    
    Parameters
    ----------
    roi_dict : dict[str, dict[str, Any]]
        Dictionary of ROI data
    unlabeled_only : bool, optional
        Return only unlabeled ROIs, by default False
    labeled_only : bool, optional
        Return only labeled ROIs, by default False
    verbose : bool, optional
        Whether to print results, by default True
        
    Returns
    -------
    keys : list[str]
        List of ROI keys matching criteria
        
    Raises
    ------
    ValueError
        If both unlabeled_only and labeled_only are True, or no ROIs match
    """
    if unlabeled_only and labeled_only:
        raise ValueError("Cannot set both unlabeled_only and labeled_only to True.")
    
    if unlabeled_only:
        keys = [k for k in roi_dict.keys() if get_label_value(roi_dict[k]['label']) == -1]
    elif labeled_only:  # Changed from 'if' to 'elif'
        keys = [k for k in roi_dict.keys() if get_label_value(roi_dict[k]['label']) != -1]
    else:
        keys = list(roi_dict.keys())
            
    if len(keys) == 0:
        raise ValueError("No ROIs match the specified filtering criteria.")
    return keys