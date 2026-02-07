def print_dataset_summary(feature_names, y, manual_only=True):
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
    print(f"Train/Test Split Summary")
    print(f"Total training samples: {len(y_train)}")
    print(f"Total testing samples: {len(y_test)}")
    print(f"Training label distribution:")
    print(f"  - Bad (0):  {(y_train == 0).sum()}")
    print(f"  - Good (1): {(y_train == 1).sum()}")
    print(f"Testing label distribution:")
    print(f"  - Bad (0):  {(y_test == 0).sum()}")
    print(f"  - Good (1): {(y_test == 1).sum()}")

def print_tuned_summary(tuned_config: dict) -> None:
    """Print summary of the best tuned model configuration."""
    model_name = type(tuned_config['model']).__name__
    cm = tuned_config['confusion_matrix']
    
    print("\n" + "-" * 50)
    print("TUNED MODEL SUMMARY")
    print("-" * 50)
    
    print(f"Model:     {model_name}")
    print(f"Transform: {tuned_config['transform']}")
    print(f"Features:  {list(tuned_config['features'].keys())}")
    
    print(f"\nHyperparameters:")
    for param, value in tuned_config['best_params'].items():
        print(f"  {param}: {value}")
    
    print(f"\nMetrics:")
    print(f"  CV Accuracy:   {tuned_config['cv_acc']:.4f}")
    print(f"  Test Accuracy: {tuned_config['test_acc']:.4f}")
    print(f"  ROC AUC:       {tuned_config['roc_auc']:.4f}")
    print(f"  F1:            {tuned_config['f1']:.4f}")
    print(f"  Precision:     {tuned_config['precision']:.4f}")
    print(f"  Recall:        {tuned_config['recall']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"              Pred 0  Pred 1")
    print(f"  Actual 0    {cm[0,0]:<7} {cm[0,1]:<7}")
    print(f"  Actual 1    {cm[1,0]:<7} {cm[1,1]:<7}")
    print("-" * 50)