from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from enum import Enum
from typing import Any
from dataclasses import dataclass
# Re-export label utilities so existing imports keep working
from utils.label_utils import (          # noqa: F401
    create_label_dict,
    get_label_value,
    get_label_source,
    get_keys,
    normalize_label_format,
    normalize_spike_label,
    label_to_text,
    update_spike_label,
    matches_label_mode,
    preserve_existing_label,
    normalize_label,
)

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

