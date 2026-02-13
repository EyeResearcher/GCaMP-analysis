
"""
Dataset containers for the classifier optimization pipeline.

Provides feature-transform wrappers and a structured dataset class for
train/test splits used during model selection and hyperparameter tuning.
"""

import numpy as np
from typing import Any

class DataSplit:
    """Container for a single data array and its feature transforms.

    Stores the raw data and lazily computes log, sqrt, and square
    transformations, making them available via ``transformed_data``.

    Parameters
    ----------
    data : np.ndarray
        Raw feature matrix (n_samples, n_features).

    Attributes
    ----------
    raw : np.ndarray
        Original untransformed data.
    transformed_data : dict[str, np.ndarray]
        Mapping of transform name to transformed array.
        Always contains the ``'raw'`` key.
    """

    def __init__(self, data):
        self.raw = data
        self.transformed_data = {'raw': self.raw}

    def log_transform(self):
        """Apply ``log1p(|x|)`` transform and store under ``'log'`` key."""
        self.transformed_data['log'] = np.log1p(np.abs(self.raw))

    def sqrt_transform(self):
        """Apply ``sqrt(|x|)`` transform and store under ``'sqrt'`` key."""
        self.transformed_data['sqrt'] = np.sqrt(np.abs(self.raw))

    def square_transform(self):
        """Apply element-wise squaring and store under ``'square'`` key."""
        self.transformed_data['square'] = self.raw.copy() ** 2

    def collect_transforms(self):
        """Compute and store all available transforms (log, sqrt, square)."""
        self.log_transform()
        self.sqrt_transform()
        self.square_transform()

class ClassifierDataset:
    """Structured train/test dataset for classifier optimization.

    Wraps feature matrices and label arrays in :class:`DataSplit` containers
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
        Ordered feature names corresponding to columns in the X arrays.
    """

    def __init__(self):
        self.x_train : DataSplit = None
        self.x_test : DataSplit = None
        self.y_train : DataSplit = None
        self.y_test : DataSplit = None

        self.feature_names : list[str] = None

    @classmethod
    def build(cls, data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], feature_names : list[str] = None) -> "ClassifierDataset":
        """Create a dataset from pre-split arrays.

        Parameters
        ----------
        data : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(X_train, X_test, y_train, y_test)`` arrays, typically
            produced by ``sklearn.model_selection.train_test_split``.
        feature_names : list[str], optional
            Column names for the feature matrices.

        Returns
        -------
        ClassifierDataset
            Fully initialised dataset ready for transform and subset
            operations.
        """
        instance = cls()
        
        instance.x_train = DataSplit(data[0])
        instance.x_test = DataSplit(data[1])
        instance.y_train = DataSplit(data[2])
        instance.y_test = DataSplit(data[3])
        instance.feature_names = feature_names
        return instance
    
    def transform_x(self):
        """Compute all feature transforms for both train and test splits."""
        self.x_train.collect_transforms()
        self.x_test.collect_transforms()
    
    def transform_data(self):
        """Compute all feature transforms on X data.

        Convenience wrapper around :meth:`transform_x`.
        """
        self.transform_x()

    def get_subset(self, top=None, transform_name='raw'):
        """Extract a feature subset from a specific transform.

        Parameters
        ----------
        top : list[str], optional
            Feature names to keep.  If *None*, all features are returned.
        transform_name : str, optional
            Key into ``DataSplit.transformed_data`` (e.g. ``'raw'``,
            ``'log'``, ``'sqrt'``, ``'square'``).  Default is ``'raw'``.

        Returns
        -------
        train_subset : np.ndarray
            Training feature matrix restricted to *top* features.
        test_subset : np.ndarray
            Test feature matrix restricted to *top* features.
        """
        if top is None:
            top = self.feature_names
    
        train = self.x_train.transformed_data[transform_name]
        test = self.x_test.transformed_data[transform_name]
        feat_idx = [self.feature_names.index(f) for f in top]
        train_subset = train[:, feat_idx]
        test_subset = test[:, feat_idx]
        return train_subset, test_subset