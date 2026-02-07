from .datasets import ClassifierDataset, DataSplit, DataSplit
from .utils import get_model, merge_dicts, train
from typing import Any
import numpy as np
from .optimize import test_configurations
class PipelineRunner:
    def __init__(self, hp_config: dict[str, Any],feature_names: list[str]):
        self.config = hp_config
        self.feature_names = feature_names
        self.base_models = {name: get_model(name, hp_config) for name in hp_config['base_models']}
        self.model_searches = {name: get_model(name, hp_config) for name in hp_config['model_searches']}
        self.dataset = ClassifierDataset()
    @classmethod
    def configure(cls, hp_config, feature_names):
        return cls(hp_config, feature_names)
    
    def make_dataset(self, splits : list[np.ndarray]) -> tuple[DataSplit, DataSplit, DataSplit, DataSplit]:
        self.dataset.build(splits)
        self.dataset.transform_data()
        return self.dataset.x_train, self.dataset.x_test, self.dataset.y_train, self.dataset.y_test
    
    def optimize_features(self, verbose=False):
        best_config = test_configurations(self.dataset, self.feature_names, verbose=verbose)
        return best_config