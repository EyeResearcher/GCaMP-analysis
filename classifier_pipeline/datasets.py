
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

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
    
    def build(self, splits: list[np.ndarray]):
        self.x_train = DataSplit(splits[0])
        self.x_test = DataSplit(splits[1])
        self.y_train = DataSplit(splits[2])
        self.y_test = DataSplit(splits[3])
        
    def transform_x(self):
        self.x_train.collect_transforms()
        self.x_test.collect_transforms()
    
    def transform_data(self):
        self.transform_x()
        return self.x_train, self.x_test, self.y_train, self.y_test
    