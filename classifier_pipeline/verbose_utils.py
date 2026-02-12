from .optimize import OptimizationResults


def print_dataset_summary(feature_names, y, manual_only=True):
    """
    Print summary of the dataset.
    
    Parameters
    ----------
    feature_names : list[str]
        Names of features
    y : np.ndarray
        Labels array
    manual_only : bool, optional
        Whether only manual labels are used, by default True
    """
    print(f"Dataset Summary")
    print(f"Total labeled ROIs: {len(y)}")
    print(f"Feature names:")
    for i, feat in enumerate(feature_names):
        print(f"\t  {i+1}. {feat}")
    print(f"Label distribution:")
    print(f"  - Bad (0):  {(y == 0).sum()}")
    print(f"  - Good (1): {(y == 1).sum()}")
    print(f"Training on: {'Manual labels only' if manual_only else 'Manual + Auto labels'}")


def print_split_summary(y_train, y_test):
    """
    Print summary of train/test split.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Testing labels
    """
    print(f"Train/Test Split Summary")
    print(f"Total training samples: {len(y_train)}")
    print(f"Total testing samples: {len(y_test)}")
    print(f"Training label distribution:")
    print(f"  - Bad (0):  {(y_train == 0).sum()}")
    print(f"  - Good (1): {(y_train == 1).sum()}")
    print(f"Testing label distribution:")
    print(f"  - Bad (0):  {(y_test == 0).sum()}")
    print(f"  - Good (1): {(y_test == 1).sum()}")


def print_tuned_summary(results: OptimizationResults) -> None:
    """
    Print summary of the tuned model results.
    
    Parameters
    ----------
    results : OptimizationResults
        Results dataclass from tune_hyperparameters()
    """
    model_name = type(results.model).__name__
    cm = results.confusion_matrix
    
    print("\n" + "-" * 50)
    print("TUNED MODEL SUMMARY")
    print("-" * 50)
    
    print(f"Model:     {model_name}")
    print(f"Transform: {results.transform}")
    print(f"Features:  {results.features}")
    
    print(f"\nHyperparameters:")
    for param, value in results.best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nMetrics:")
    print(f"  CV Accuracy:   {results.cv_acc:.4f}")
    print(f"  Test Accuracy: {results.test_acc:.4f}")
    print(f"  ROC AUC:       {results.roc_auc:.4f}")
    print(f"  F1:            {results.f1:.4f}")
    print(f"  Precision:     {results.precision:.4f}")
    print(f"  Recall:        {results.recall:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"              Pred 0  Pred 1")
    print(f"  Actual 0    {cm[0,0]:<7} {cm[0,1]:<7}")
    print(f"  Actual 1    {cm[1,0]:<7} {cm[1,1]:<7}")
    print("-" * 50)


def print_keys(rois: int, key_type: str, keys: int = None):
    """
    Print summary of ROI keys.
    
    Parameters
    ----------
    rois : int
        Total number of ROIs
    key_type : str
        Type of keys ('labeled', 'unlabeled')
    keys : int, optional
        Number of matching keys, by default None
    """
    if rois == keys:
        print(f"Returning all {rois} keys.")
    else:
        print(f"Found {keys} {key_type} ROIs out of {rois} ROIs.")