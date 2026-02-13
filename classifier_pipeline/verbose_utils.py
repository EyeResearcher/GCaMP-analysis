from .optimize import OptimizationResults


def print_dataset_summary(feature_names, y_train, y_test, manual_only=True):
    """
    Print summary of the dataset including train/test label distributions.
    
    Parameters
    ----------
    feature_names : list[str]
        Names of features.
    y_train : np.ndarray
        Training labels array.
    y_test : np.ndarray
        Testing labels array.
    manual_only : bool, optional
        Whether only manual labels are used, by default True.
    """
    total = len(y_train) + len(y_test)
    print(f"Dataset Summary")
    print(f"Total labeled ROIs: {total}")
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"\nFeature names:")
    for i, feat in enumerate(feature_names):
        print(f"\t  {i+1}. {feat}")
    print(f"\nLabel distribution:")
    print(f"  {'':10s} {'Bad (0)':>8s} {'Good (1)':>9s}")
    print(f"  {'Train':<10s} {(y_train == 0).sum():>8d} {(y_train == 1).sum():>9d}")
    print(f"  {'Test':<10s} {(y_test == 0).sum():>8d} {(y_test == 1).sum():>9d}")
    print(f"  {'Total':<10s} {((y_train == 0).sum() + (y_test == 0).sum()):>8d} {((y_train == 1).sum() + (y_test == 1).sum()):>9d}")
    print(f"\nTraining on: {'Manual labels only' if manual_only else 'Manual + Auto labels'}")


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