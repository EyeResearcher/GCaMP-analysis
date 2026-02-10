
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from typing import Any, Dict, Tuple, List
from classifier_pipeline.datasets import ClassifierDataset, DataSplit
from classifier_pipeline.utils import get_model, merge_dicts, get_feature_importance, train

def test_feature_selection(model_class : str, dataset : ClassifierDataset, 
                           importance_df : pd.DataFrame, transform_name, 
                           verbose=True, **hp_kwargs) -> tuple[list[dict], float, dict[str, Any]]:
    """
    Test model with top N features based on importance.
    
    Parameters
    ----------
    model_class : sklearn model class
    model_name : str
    X_train, X_test, y_train, y_test : arrays of shape (n_samples, n_features)
    feature_names : list
    importance_df : pd.DataFrame
        Feature importance from full model
    transform_name : str
        Which transform was used
    **hp_kwargs : dict
        Model parameters
        
    Returns
    -------
    results : list
        Results for different feature subset sizes
    """
    X_train, X_test = dataset.x_train, dataset.x_test
    y_train, y_test = dataset.y_train, dataset.y_test
    feature_names = dataset.feature_names

    data_transformed = merge_dicts(X_train.transformed_data, X_test.transformed_data)
    X_train_transformed = data_transformed[transform_name][0]
    X_test_transformed = data_transformed[transform_name][1]
    

    feature_subsets = [3, 5, len(feature_names)]  # Top 3, Top 5, All features
    best = 0 
    best_features = []  

    for n_features in feature_subsets:
        
        top_features : list[str] = importance_df.head(n_features)['feature'].tolist()
    
        feature_indices = [feature_names.index(f) for f in top_features]
        X_train_subset = X_train_transformed[:, feature_indices]
        X_test_subset = X_test_transformed[:, feature_indices]

        model = get_model(model_class, **hp_kwargs)
        
        acc = train(model, X_train_subset, y_train.raw, X_test_subset, y_test.raw)
        
        
        if acc > best:
            best = acc
            best_features = top_features
    return best_features, best

def test_model_with_transforms(model_class : str , dataset : ClassifierDataset, verbose = True, **hp_kwargs) -> dict[str, Any]:
    """
    Test a model with different feature transformations.
    
    Args:
        model_class: str
            - RF: Random Forest
            - LR: Logistic Regression
            - SVM: Support Vector Machine
        dataset: ClassifierDataset
        verbose: bool
        **hp_kwargs: hyperparameters for the model
    Returns: 
        dict[str, Any]
            - 'acc': best test accuracy
            - 'model_type': model class
            - 'transform': best transform
            - 'feature_importance': feature importance DataFrame

    """
    X_train, X_test = dataset.x_train, dataset.x_test
    y_train, y_test = dataset.y_train, dataset.y_test
    feature_names = dataset.feature_names

    data_transformed = merge_dicts(X_train.transformed_data, X_test.transformed_data)
    
    best = 0
    best_transform = None
    for transform, (X_train_var, X_test_var) in data_transformed.items():
        
        model = get_model(model_class, **hp_kwargs)
        acc= train(model, X_train_var, y_train.raw, X_test_var, y_test.raw)

        if acc > best:
            best = acc
            best_transform = transform
            feat_importance = get_feature_importance(model, feature_names)
    
    return best_transform, feat_importance

def get_best_version(model_type : str, dataset : ClassifierDataset, verbose=True): 
    transform, feat_importance = test_model_with_transforms(model_type, dataset, verbose=verbose)
    best_features, best = test_feature_selection(model_type, dataset, transform, feat_importance, verbose=verbose)
    return best_features, best, transform
def get_best_model(dataset : ClassifierDataset, verbose=True):
    """This function takes a dataset and returns the best model type and configuration based on accuracy.
    Args: 
        dataset (ClassifierDataset) : the dataset with all x and y data
    Returns:
        best_type (str) : the best model type based on accuracy
        best_features (str) : feature names to be used 
        best_accuracy (float) : the best accuracy achieved
        transform (str) : the best feature transform used
        """
    models = ["RF", "LR"]
    best_acc = 0
    best_type = None
    best_config = None

    for model in models:
        results = get_best_version(model, dataset, verbose=verbose)
        if results[1] > best_acc:
            best_acc = results[1]   
            best_type = model
            best_config = results
    best_features, best_accuracy, transform = best_config

    return best_type, best_features, best_accuracy, transform


def tune_hyperparameters(best_type, dataset : ClassifierDataset, best_features, transform, base_model, hp_grid):
    
    y_train, y_test = dataset.y_train, dataset.y_test
    x_train, x_test = dataset.get_subset(best_features, transform)

    search = GridSearchCV(base_model, hp_grid, cv=5, scoring='accuracy', n_jobs=-1)
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_score = search.best_score_
    test_acc = best_model.score(x_test, y_test)

    y_pred_proba = best_model.predict_proba(x_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    cm = confusion_matrix(y_test, best_model.predict(x_test))
    report = classification_report(y_test, best_model.predict(x_test), output_dict=True)['weighted avg']
    return {
        'model' : best_model,
        'best_params': best_params,
        'cv_acc': cv_score,
        'test_acc': test_acc,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'f1' : report['f1-score'],
        'precision' : report['precision'],
        'recall' : report['recall'],
        'features': best_features,
        'transform': transform
    }