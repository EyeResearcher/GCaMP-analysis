
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from typing import Any, Dict, Tuple, List
from classifier_pipeline.datasets import DataSplit
from classifier_pipeline.utils import get_model, merge_dicts, train
from spike_classifier.train_classifier import get_feature_importance

def test_feature_selection(model_class, model_name, X_train : DataSplit, X_test: DataSplit, y_train: DataSplit, y_test: DataSplit,
                          feature_names, importance_df : pd.DataFrame, transform_name, verbose=True,
                          **hp_kwargs) -> tuple[list[dict], float, dict[str, Any]]:
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
    
    data_transformed = merge_dicts(X_train.transformed_data, X_test.transformed_data)
    X_train_transformed = data_transformed[transform_name][0]
    X_test_transformed = data_transformed[transform_name][1]
    
    results = []
    feature_subsets = [3, 5, len(feature_names)]  # Top 3, Top 5, All features
    best = 0 
    best_config = None   
    for n_features in feature_subsets:
        
        top_features : list[str]= importance_df.head(n_features)['feature'].tolist()
    
        feature_indices = [feature_names.index(f) for f in top_features]
        X_train_subset = X_train_transformed[:, feature_indices]
        X_test_subset = X_test_transformed[:, feature_indices]

        model = get_model(model_class, **hp_kwargs)
        
        result = train(model, X_train_subset, y_train.raw, X_test_subset, y_test.raw, top_features, model_name, transform_name)
        results.append(result)
        
        if result['test_acc'] > best:
            best = result['test_acc']
            best_config = result
    return results, best, best_config

def test_model_with_transforms(model_class, model_name, X_train : DataSplit, X_test: DataSplit, y_train: DataSplit, y_test: DataSplit, 
                               feature_names, verbose = True, **hp_kwargs) -> tuple[list[dict], float, str]:
    """
    Test a model with different feature transformations.
    
    Parameters
    ----------
    model_class : sklearn model class
        Either RandomForestClassifier or LogisticRegression
    model_name : str
        Name of the model for display
    X_train, X_test, y_train, y_test : arrays of shape (n_samples, n_features)
        Train/test split data
    feature_names : list
        Feature names
    n_estimators : int
        For Random Forest only
    max_depth : int or None
        For Random Forest only
    random_state : int
        Random seed
    
    Returns
    -------
    results : list
        List of result dicts
    best : dict
        Best configuration
    """
    data_transformed = merge_dicts(X_train.transformed_data, X_test.transformed_data)
    
    results = []
    best = 0
    best_transform = None
    best_config = None
    for transform, (X_train_var, X_test_var) in data_transformed.items():
        
        model = get_model(model_class, **hp_kwargs)
        result = train(model, X_train_var, y_train.raw, X_test_var, y_test.raw, feature_names, model_name, transform)

        results.append(result)

        if result['test_acc'] > best:
            best = result['test_acc']
            best_transform = transform
            best_config = result
    
    return results, best, best_transform, best_config

def test_configurations(X_train, X_test, y_train, y_test, feature_names, n_estimators, max_depth, random_state, verbose=True):
    
    rf_results, rf_best, rf_transform, rf_best_transform = test_model_with_transforms(
        RandomForestClassifier, "Random Forest",
        X_train, X_test, y_train, y_test, feature_names,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    
    # Test Logistic Regression
    lr_results, lr_best, lr_transform, lr_best_transform = test_model_with_transforms(
        LogisticRegression, "Logistic Regression",
        X_train, X_test, y_train, y_test, feature_names,
        random_state=random_state
    )
    
    
    # Test feature selection for both models
    rf_feature_results, rf_best_acc, rf_best_config = test_feature_selection(
        RandomForestClassifier, "Random Forest",
        X_train, X_test, y_train, y_test, feature_names,
        rf_best_transform['feature_importance'], rf_best_transform['transform'],
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    
    lr_feature_results, lr_best_acc, lr_best_config = test_feature_selection(
        LogisticRegression, "Logistic Regression",
        X_train, X_test, y_train, y_test, feature_names,
        lr_best_transform['feature_importance'], lr_best_transform['transform'],
        random_state=random_state
    )
    results = [rf_best_config, lr_best_config]
    best_acc = [rf_best_acc, lr_best_acc]
    best_config = results[np.argmax(best_acc)]
    return best_config

def tune_hyperparameters(best_config, hp_config):
    best_type = best_config['model_type']
    X_train, X_test, y_train, y_test = best_config['X_train'], best_config['X_test'], best_config['y_train'], best_config['y_test']

    base_config = hp_config.get('base_model', '')[best_type.__name__]
    base_model = get_model(best_type, **base_config)
    hp_grid = hp_config.get(best_config['model_type'].__name__, '') 

    search = GridSearchCV(base_model, hp_grid, cv=5, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_score = search.best_score_
    test_acc = best_model.score(X_test, y_test)

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    cm = confusion_matrix(y_test, best_model.predict(X_test))
    report = classification_report(y_test, best_model.predict(X_test), output_dict=True)['weighted avg']
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
        'features': best_config['features'],
        'transform': best_config['transform']
    }