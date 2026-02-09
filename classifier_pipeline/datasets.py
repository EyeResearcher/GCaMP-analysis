
import numpy as np
from typing import Any

class DataSplit:
    def __init__(self, data):
        self.raw = data
        self.transformed_data = {'raw': self.raw}
    def log_transform(self):
        self.transformed_data['log'] = np.log1p(np.abs(self.raw))
    def sqrt_transform(self):
        self.transformed_data['sqrt'] = np.sqrt(np.abs(self.raw))
    def square_transform(self):
        self.transformed_data['square'] = self.raw.copy() ** 2
    def collect_transforms(self):
        self.log_transform()
        self.sqrt_transform()
        self.square_transform()

class ClassifierDataset:
    def __init__(self):

        self.x_train : DataSplit = None
        self.x_test : DataSplit = None
        self.y_train : DataSplit = None
        self.y_test : DataSplit = None

        self.feature_names : list[str] = None
        
    @classmethod
    def build(cls, splits: list[np.ndarray], feature_names : list[str] = None) -> "ClassifierDataset":
        instance = cls()
        instance.x_train = DataSplit(splits[0])
        instance.x_test = DataSplit(splits[1])
        instance.y_train = DataSplit(splits[2])
        instance.y_test = DataSplit(splits[3])
        instance.feature_names = feature_names
        return instance
    
    def transform_x(self):
        self.x_train.collect_transforms()
        self.x_test.collect_transforms()
    
    def transform_data(self):
        self.transform_x()
        
    