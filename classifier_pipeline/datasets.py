"""
Dataset containers for the classifier optimization pipeline.

Provides feature-transform wrappers and a structured dataset class for
train/test splits used during model selection and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from typing import Any


def apply_transform(X: pd.DataFrame | np.ndarray, transform: str | None) -> pd.DataFrame | np.ndarray:
    """Apply a named feature transform to a matrix.

    Must match the transforms in :class:`DataSplit` so that inference
    uses the same mapping the model was trained with.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Feature matrix (n_samples, n_features).
    transform : str or None
        One of ``'raw'``, ``'log'``, ``'sqrt'``, ``'square'``, or *None*.
        ``'raw'`` and *None* return *X* unchanged.

    Returns
    -------
    pd.DataFrame | np.ndarray
        Transformed feature matrix (same shape and type as *X*).
    """
    if transform is None or transform == "raw":
        return X
    
    
    if transform == "log":
        return np.log1p(np.abs(X))
    if transform == "sqrt":
        return np.sqrt(np.abs(X))
    if transform == "square":
        return X ** 2
    
    raise ValueError(f"Unknown transform: {transform!r}")


class DataSplit:
    """Container for a single data split and its feature transforms.

    Stores the raw data and computes all transforms via
    :func:`apply_transform`, making them available in
    ``transformed_data``.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Raw feature matrix or label series.

    Attributes
    ----------
    raw : pd.DataFrame | pd.Series
        Original untransformed data.
    transformed_data : dict[str, pd.DataFrame]
        Mapping of transform name to transformed DataFrame.
        Always contains the ``'raw'`` key. Only populated for DataFrames.
    """

    _transforms = ("raw", "log", "sqrt", "square")

    def __init__(self, data: pd.DataFrame | pd.Series):
        self.raw = data
        self.transformed_data: dict[str, pd.DataFrame] = {"raw": self.raw}
    
    @property
    def values(self) -> np.ndarray:
        """Return underlying numpy array."""
        return self.raw.values
    
    @property
    def columns(self) -> list[str]:
        """Return column names if DataFrame."""
        if isinstance(self.raw, pd.DataFrame):
            return self.raw.columns.tolist()
        return None

    def collect_transforms(self) -> None:
        """Compute and store all available transforms.
        
        Only applies to DataFrames (features), not Series (labels).
        """
        assert isinstance(self.raw, pd.DataFrame)
            
        for name in self._transforms:
            if name not in self.transformed_data:
                self.transformed_data[name] = apply_transform(self.raw, name)


class ClassifierDataset:
    """Structured train/test dataset for classifier optimization.

    Wraps feature DataFrames and label Series in :class:`DataSplit` containers
    so that multiple feature transforms can be evaluated during model
    selection.

    Attributes
    ----------
    x_train : DataSplit or None
        Training feature split.
    x_test : DataSplit or None
        Test feature split.
    y_train : DataSplit or None
        Training label split.
    y_test : DataSplit or None
        Test label split.
    feature_names : list[str] or None
        Ordered feature names corresponding to columns in the X DataFrames.
    """

    def __init__(self):
        self.x_train: DataSplit = None
        self.x_test: DataSplit = None
        self.y_train: DataSplit = None
        self.y_test: DataSplit = None
        self.feature_names: list[str] = None

    @classmethod
    def build(cls, x_train: pd.DataFrame, x_test: pd.DataFrame, 
              y_train: pd.Series, y_test: pd.Series) -> "ClassifierDataset":
        """Create a dataset from pre-split DataFrames/Series.

        Parameters
        ----------
        x_train : pd.DataFrame
            Training features
        x_test : pd.DataFrame
            Test features
        y_train : pd.Series
            Training labels
        y_test : pd.Series
            Test labels

        Returns
        -------
        ClassifierDataset
            Fully initialised dataset ready for transform and subset
            operations.
            
        Raises
        ------
        ValueError
            If inputs are not DataFrames/Series
        """
        if not isinstance(x_train, pd.DataFrame):
            raise ValueError("x_train must be a DataFrame")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train must be a Series")
        
        instance = cls()
        
        instance.x_train = DataSplit(x_train)
        instance.x_test = DataSplit(x_test)
        instance.y_train = DataSplit(y_train)
        instance.y_test = DataSplit(y_test)
        instance.feature_names = x_train.columns.tolist()

        return instance
    
    def transform_data(self) -> None:
        """Compute all feature transforms for both train and test splits."""
        self.x_train.collect_transforms()
        self.x_test.collect_transforms()

    def get_subset(self, top: list[str] = None, 
                   transform_name: str = 'raw') -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract a feature subset from a specific transform.

        Parameters
        ----------
        top : list[str], optional
            Feature names to keep. If *None*, all features are returned.
        transform_name : str, optional
            Key into ``DataSplit.transformed_data`` (e.g. ``'raw'``,
            ``'log'``, ``'sqrt'``, ``'square'``). Default is ``'raw'``.

        Returns
        -------
        train_subset : pd.DataFrame
            Training feature DataFrame restricted to *top* features.
        test_subset : pd.DataFrame
            Test feature DataFrame restricted to *top* features.
        """
        if top is None:
            top = self.feature_names
    
        train_df = self.x_train.transformed_data[transform_name]
        test_df = self.x_test.transformed_data[transform_name]
        
        return train_df[top], test_df[top]
    
    def get_labels(self) -> tuple[pd.Series, pd.Series]:
        """Return train and test labels as Series.
        
        Returns
        -------
        y_train : pd.Series
            Training labels
        y_test : pd.Series
            Test labels
        """
        return self.y_train.raw, self.y_test.raw