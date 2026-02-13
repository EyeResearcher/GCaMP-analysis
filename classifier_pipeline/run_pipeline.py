from .datasets import ClassifierDataset, DataSplit
from .utils import get_model
from .verbose_utils import print_dataset_summary
from .optimize import ModelOptimizer, OptimizationResults
from typing import Any
import numpy as np


class PipelineRunner:
    """
    Pipeline for running model optimization and hyperparameter tuning.
    
    Parameters
    ----------
    hp_config : dict[str, Any]
        Configuration containing base_models and model_searches
    feature_names : list[str]
        Names of features in the dataset
    verbose : bool, optional
        Whether to print progress, by default True
        
    Attributes
    ----------
    config : dict[str, Any]
        Hyperparameter configuration
    feature_names : list[str]
        Feature names
    dataset : ClassifierDataset or None
        Dataset after build
    optimizer : ModelOptimizer or None
        Optimizer after finding best model
    """
    
    def __init__(self, hp_config: dict[str, Any], feature_names: list[str], verbose: bool = True):
        self.config = hp_config
        self.feature_names = feature_names
        self.verbose = verbose
        self.dataset: ClassifierDataset = None
        self.optimizer: ModelOptimizer = None
    
    def build_dataset(self, splits: list[np.ndarray]) -> 'PipelineRunner':
        """
        Build and transform the dataset.
        
        Parameters
        ----------
        splits : list[np.ndarray]
            List of [x_train, x_test, y_train, y_test]
            
        Returns
        -------
        self : PipelineRunner
            Returns self for method chaining
        """
        self.dataset = ClassifierDataset.build(splits, self.feature_names)
        self.dataset.transform_data()
        if self.verbose:
            print_dataset_summary(self.feature_names, self.dataset.y_train.raw, self.dataset.y_test.raw)
        return self
    
    def find_best_model(self) -> 'PipelineRunner':
        """
        Find the best model type, features, and transform.
        
        Returns
        -------
        self : PipelineRunner
            Returns self for method chaining
            
        Raises
        ------
        ValueError
            If build_dataset() hasn't been called
        """
        if self.dataset is None:
            raise ValueError("Call build_dataset() first")
            
        self.optimizer = ModelOptimizer(self.dataset, verbose=self.verbose)
        self.optimizer.find_best_model()
        return self
    
    def tune(self) -> OptimizationResults:
        """
        Tune hyperparameters for the best model.
        
        Returns
        -------
        results : OptimizationResults
            Optimization results with tuned model
            
        Raises
        ------
        ValueError
            If find_best_model() hasn't been called
        """
        if self.optimizer is None or self.optimizer.best_model_type is None:
            raise ValueError("Call find_best_model() first")
            
        model_type = self.optimizer.best_model_type
        base_model = get_model(model_type)
        hp_grid = self.config.get('model_searches', {}).get(model_type, {})
        
        return self.optimizer.tune_hyperparameters(hp_grid, base_model)
    
    def run(self, splits: list[np.ndarray]) -> OptimizationResults:
        """
        Run the full pipeline.
        
        Parameters
        ----------
        splits : list[np.ndarray]
            List of [x_train, x_test, y_train, y_test]
            
        Returns
        -------
        results : OptimizationResults
            Final optimization results

        """
        return self.build_dataset(splits).find_best_model().tune()