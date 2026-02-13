import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from typing import Any, Dict, Tuple, List
from classifier_pipeline.datasets import ClassifierDataset
from classifier_pipeline.utils import get_model, get_feature_importance, train_and_evaluate


@dataclass
class OptimizationResults:
    """
    Results from model optimization and hyperparameter tuning.
    
    Attributes
    ----------
    model : RFC | LR | SVC
        The trained model
    best_params : dict
        Best hyperparameters found
    cv_acc : float
        Cross-validation accuracy
    test_acc : float
        Test set accuracy
    roc_auc : float
        ROC AUC score
    confusion_matrix : np.ndarray
        Confusion matrix
    f1 : float
        F1 score
    precision : float
        Precision score
    recall : float
        Recall score
    features : list[str]
        Features used
    transform : str
        Transform used
    """
    model: RFC | LR | SVC
    best_params: dict
    cv_acc: float
    test_acc: float
    roc_auc: float
    confusion_matrix: np.ndarray
    f1: float
    precision: float
    recall: float
    features: list[str]
    transform: str

    def to_dict(self, include_model: bool = False) -> dict:
        """
        Convert results to a JSON-serializable dictionary.
        
        Parameters
        ----------
        include_model : bool, optional
            Whether to include model info, by default False
            
        Returns
        -------
        dict
            Serializable dictionary of results
        """
        d = {
            'best_params': self.best_params,
            'cv_acc': float(self.cv_acc),
            'test_acc': float(self.test_acc),
            'roc_auc': float(self.roc_auc),
            'confusion_matrix': self.confusion_matrix.tolist(),
            'f1': float(self.f1),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'features': self.features,
            'transform': self.transform
        }
        if include_model:
            d['model_type'] = type(self.model).__name__
        return d


class ModelOptimizer:
    """
    Optimizer for finding the best model, transform, and features.
    
    Parameters
    ----------
    dataset : ClassifierDataset
        The dataset with all x and y data
    verbose : bool, optional
        Whether to print results, by default True
    metric : str, optional
        Metric to optimize, by default 'roc_auc'
        
    Attributes
    ----------
    dataset : ClassifierDataset
        The dataset used for optimization
    verbose : bool
        Whether to print results
    metric : str
        Metric used for optimization
    best_model_type : str or None
        The best model type found after optimization
    best_features : list[str] or None
        The best features found after optimization
    best_transform : str or None
        The best transform found after optimization
    best_accuracy : float or None
        The best accuracy achieved
    results : OptimizationResults or None
        Final optimization results after tuning
    """
    
    def __init__(self, dataset: ClassifierDataset, verbose: bool = True, metric: str = 'roc_auc'):
        self.dataset = dataset
        self.verbose = verbose
        self.metric = metric
        self.best_model_type: str = None
        self.best_features: list[str] = None
        self.best_transform: str = None
        self.best_accuracy: float = None
        self.results: OptimizationResults = None

    def _get_labels(self) -> tuple[pd.Series, pd.Series]:
        """
        Get train and test labels.
        
        Returns
        -------
        y_train : pd.Series
            Training labels
        y_test : pd.Series
            Test labels
        """
        return self.dataset.get_labels()

    def _test_transforms(self, model_class: str, **hp_kwargs) -> tuple[str, pd.DataFrame]:
        """
        Test a model with different feature transformations.
        
        Parameters
        ----------
        model_class : str
            Model type identifier ('RF', 'LR', 'SVM')
        **hp_kwargs : dict
            Hyperparameters for the model
            
        Returns
        -------
        best_transform : str
            The best transform found
        feat_importance : pd.DataFrame
            Feature importance DataFrame from the best model
        """
        y_train, y_test = self._get_labels()
        best = 0
        best_transform = None
        feat_importance = None
        
        for transform in self.dataset.x_train.transformed_data.keys():
            X_train, X_test = self.dataset.get_subset(transform_name=transform)
            model = get_model(model_class, **hp_kwargs)
            acc = train_and_evaluate(model, X_train, y_train, X_test, y_test, metric=self.metric)

            if acc > best:
                best = acc
                best_transform = transform
                feat_importance = get_feature_importance(model, X_train.columns.tolist())
                
        if self.verbose: 
            print(f"{model_class}")
            print(f"\t Best transform: {best_transform} with {self.metric}: {best:.4f}")
            
        return best_transform, feat_importance
    
    def _test_feature_selection(self, model_class: str, importance_df: pd.DataFrame, 
                                 transform_name: str, **hp_kwargs) -> tuple[list[str], float]:
        """
        Test model with top N features based on importance.
        
        Parameters
        ----------
        model_class : str
            Model type identifier ('RF', 'LR', 'SVM')
        importance_df : pd.DataFrame
            Feature importance from full model
        transform_name : str
            Which transform was used
        **hp_kwargs : dict
            Model hyperparameters
            
        Returns
        -------
        best_features : list[str]
            The best features selected
        best : float
            The best accuracy achieved
        """
        y_train, y_test = self._get_labels()
        n_features = len(self.dataset.feature_names)
        feature_subsets = [n for n in [3, 5, n_features] if n <= n_features]
        best = 0 
        best_features = []  

        for n in feature_subsets:
            top_features: list[str] = importance_df.head(n)['feature'].tolist()
            X_train, X_test = self.dataset.get_subset(top=top_features, transform_name=transform_name)
            model = get_model(model_class, **hp_kwargs)
            acc = train_and_evaluate(model, X_train, y_train, X_test, y_test, metric=self.metric)
            
            if acc > best:
                best = acc
                best_features = top_features
                
        if self.verbose:
            print(f"{model_class} with transform: {transform_name}")
            print(f"\t Best features: {best_features} with {self.metric}: {best:.4f}")
            
        return best_features, best

    def optimize_model(self, model_type: str) -> tuple[list[str], float, str]:
        """
        Get best version of a given model type.
        
        Parameters
        ----------
        model_type : str
            The type of model to optimize
            
        Returns
        -------
        best_features : list[str]
            The best features selected
        best : float
            The best accuracy achieved
        transform : str
            The best transform used
        """
        transform, feat_importance = self._test_transforms(model_type)
        best_features, best = self._test_feature_selection(model_type, feat_importance, transform)
        return best_features, best, transform

    def find_best_model(self, model_types: list[str] = None) -> 'ModelOptimizer':
        """
        Find the best model type and configuration.
        
        Parameters
        ----------
        model_types : list[str], optional
            Model types to compare, by default ['RF', 'LR']
            
        Returns
        -------
        self : ModelOptimizer
            Returns self for method chaining
        """
        if model_types is None:
            model_types = ["RF", "LR"]

        best_acc = 0
        best_type = None
        best_config = None

        for model in model_types:
            results = self.optimize_model(model)
            if results[1] > best_acc:
                best_acc = results[1]   
                best_type = model
                best_config = results
                
        self.best_model_type = best_type
        self.best_features, self.best_accuracy, self.best_transform = best_config
        
        return self

    def tune_hyperparameters(self, hp_grid: dict, base_model: RFC | LR | SVC = None) -> OptimizationResults:
        """
        Tune hyperparameters for the best model using GridSearchCV.
        
        Parameters
        ----------
        hp_grid : dict
            Hyperparameter grid for tuning
        base_model : RFC | LR | SVC, optional
            Model to tune. If None, uses best_model_type
            
        Returns
        -------
        results : OptimizationResults
            Dataclass containing model and metrics
            
        Raises
        ------
        ValueError
            If find_best_model() hasn't been called and base_model is None
        """
        if base_model is None:
            if self.best_model_type is None:
                raise ValueError("Call find_best_model() first or provide base_model")
            base_model = get_model(self.best_model_type)
            
        if self.best_features is None:
            raise ValueError("Call find_best_model() first")
            
        y_train, y_test = self._get_labels()
        X_train, X_test = self.dataset.get_subset(self.best_features, self.best_transform)

        search = GridSearchCV(base_model, hp_grid, cv=5, scoring='accuracy', n_jobs=-1)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)['weighted avg']
        
        self.results = OptimizationResults(
            model=best_model,
            best_params=search.best_params_,
            cv_acc=search.best_score_,
            test_acc=best_model.score(X_test, y_test),
            roc_auc=roc_auc_score(y_test, y_pred_proba),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            f1=report['f1-score'],
            precision=report['precision'],
            recall=report['recall'],
            features=self.best_features,
            transform=self.best_transform
        )
        return self.results