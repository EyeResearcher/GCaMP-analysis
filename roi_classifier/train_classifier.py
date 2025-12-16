"""
Train ROI classifiers (Random Forest and Logistic Regression).

Tests different feature transformations and compares model performance.
Only trains on manually labeled ROIs.
"""
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib


def get_label_value(label):
    """Extract numeric label value from either dict or int format."""
    if isinstance(label, dict):
        return label.get('value', -1)
    return label


def get_label_source(label):
    """Extract label source from either dict or int format."""
    if isinstance(label, dict):
        return label.get('source', 'unknown')
    return 'unknown'


def load_labeled_roi_data(data_path: Path, manual_only: bool = True):
    """
    Load ROI data and filter for labeled ROIs.
    
    Parameters
    ----------
    data_path : Path
        Path to ROI data .npy file
    manual_only : bool
        If True, only use manually labeled ROIs (default: True)
    
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    feature_names : list
        Names of features
    roi_keys : list
        ROI keys corresponding to each sample
    """
    npy_dict = np.load(data_path, allow_pickle=True).item()
    
    rows = []
    roi_keys_list = []
    
    for roi_key, roi_data in npy_dict.items():
        label_value = get_label_value(roi_data['label'])
        label_source = get_label_source(roi_data['label'])
        
        # Skip unlabeled ROIs
        if label_value == -1:
            continue
        
        # Skip auto-labeled ROIs if manual_only is True
        if manual_only and label_source != 'manual':
            continue
        
        features = list(roi_data['features'].values())
        rows.append(features + [label_value])
        roi_keys_list.append(roi_key)
    
    if len(rows) == 0:
        raise ValueError("No labeled data found! Please annotate some ROIs first.")
    
    data_array = np.array(rows)
    X = data_array[:, :-1]  # All columns except last
    y = data_array[:, -1]   # Last column
    
    # Get feature names from first ROI
    first_roi = next(iter(npy_dict.values()))
    feature_names = list(first_roi['features'].keys())
    
    return X, y, feature_names, roi_keys_list


def create_feature_variants(X, feature_names):
    """
    Create multiple feature variants with different transformations.
    
    Returns:
        variants: dict mapping transform name to feature matrix
        variant_names: dict mapping transform name to feature name list
    """
    variants = {
        'raw': X.copy(),
        'log': np.log1p(np.abs(X)),
        'sqrt': np.sqrt(np.abs(X)),
        'square': X ** 2,
    }
    
    variant_names = {
        'raw': feature_names,
        'log': [f"{name}_log" for name in feature_names],
        'sqrt': [f"{name}_sqrt" for name in feature_names],
        'square': [f"{name}_sq" for name in feature_names],
    }
    
    return variants, variant_names


def get_feature_importance(model, feature_names, transform_name):
    """
    Extract feature importance from a trained model.
    
    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame with features sorted by importance
    """
    if hasattr(model, 'feature_importances_'):
        # Random Forest
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Logistic Regression - use absolute coefficients
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    # Apply transform suffix to feature names
    if transform_name != 'raw':
        suffix_map = {'log': '_log', 'sqrt': '_sqrt', 'square': '_sq'}
        display_names = [f"{name}{suffix_map[transform_name]}" for name in feature_names]
    else:
        display_names = feature_names
    
    importance_df = pd.DataFrame({
        'feature': display_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


def test_model_with_transforms(model_class, model_name, X_train, X_test, y_train, y_test, 
                               feature_names, n_estimators=100, max_depth=None, random_state=42):
    """
    Test a model with different feature transformations.
    
    Parameters
    ----------
    model_class : sklearn model class
        Either RandomForestClassifier or LogisticRegression
    model_name : str
        Name of the model for display
    X_train, X_test, y_train, y_test : arrays
        Train/test split data
    feature_names : list
        Feature names
    n_estimators : int
        For Random Forest only
    max_depth : int or None
        For Random Forest only
    random_state : int
        Random seed
    
    Returns
    -------
    results : list
        List of result dicts
    best : dict
        Best configuration
    """
    print(f"\n{'='*70}")
    print(f"TESTING {model_name.upper()}")
    print(f"{'='*70}")
    
    variants_train, _ = create_feature_variants(X_train, feature_names)
    variants_test, _ = create_feature_variants(X_test, feature_names)
    
    transformation_options = ['raw', 'log', 'sqrt', 'square']
    results = []
    best = {'test_acc': -1}
    
    for transform in transformation_options:
        X_train_var = variants_train[transform]
        X_test_var = variants_test[transform]
        
        # Create model instance
        if model_class == RandomForestClassifier:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        else:  # LogisticRegression
            model = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_var, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        cv_acc = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Train and test
        model.fit(X_train_var, y_train)
        test_acc = model.score(X_test_var, y_test)
        y_pred_proba = model.predict_proba(X_test_var)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Get feature importance
        importance_df = get_feature_importance(model, feature_names, transform)
        
        result = {
            'model': model_name,
            'transform': transform,
            'cv_acc': cv_acc,
            'cv_std': cv_std,
            'test_acc': test_acc,
            'roc_auc': roc_auc,
            'model_instance': model,
            'X_test': X_test_var,
            'config': {name: transform for name in feature_names},
            'feature_importance': importance_df
        }
        results.append(result)
        
        print(f"  {transform.upper():6s}: CV Acc: {cv_acc:.4f} (+/- {cv_std*2:.4f}) | Test Acc: {test_acc:.4f} | ROC AUC: {roc_auc:.4f}")
        
        if test_acc > best['test_acc']:
            best = result
            print(f"  ✓ New best!")
    
    return results, best


def test_feature_selection(model_class, model_name, X_train, X_test, y_train, y_test,
                          feature_names, importance_df, transform_name, 
                          n_estimators=100, max_depth=None, random_state=42):
    """
    Test model with top N features based on importance.
    
    Parameters
    ----------
    model_class : sklearn model class
    model_name : str
    X_train, X_test, y_train, y_test : arrays
    feature_names : list
    importance_df : pd.DataFrame
        Feature importance from full model
    transform_name : str
        Which transform was used
    n_estimators, max_depth, random_state : int
        Model parameters
        
    Returns
    -------
    results : list
        Results for different feature subset sizes
    """
    print(f"\n{'='*70}")
    print(f"FEATURE SELECTION - {model_name.upper()} with {transform_name.upper()} transform")
    print(f"{'='*70}")
    
    # Apply transformation
    variants_train, _ = create_feature_variants(X_train, feature_names)
    variants_test, _ = create_feature_variants(X_test, feature_names)
    X_train_transformed = variants_train[transform_name]
    X_test_transformed = variants_test[transform_name]
    
    results = []
    feature_subsets = [3, 5, len(feature_names)]  # Top 3, Top 5, All features
    
    for n_features in feature_subsets:
        n_features = min(n_features, len(feature_names))  # Don't exceed available features
        
        # Get top N feature indices
        top_features = importance_df.head(n_features)['feature'].tolist()
        
        # Map back to original feature names (remove transform suffix)
        suffix_map = {'log': '_log', 'sqrt': '_sqrt', 'square': '_sq'}
        if transform_name != 'raw':
            suffix = suffix_map[transform_name]
            top_feature_names = [f.replace(suffix, '') for f in top_features]
        else:
            top_feature_names = top_features
        
        # Get feature indices
        feature_indices = [feature_names.index(f) for f in top_feature_names]
        
        # Select features
        X_train_subset = X_train_transformed[:, feature_indices]
        X_test_subset = X_test_transformed[:, feature_indices]
        
        # Train model
        if model_class == RandomForestClassifier:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            model = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_subset, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        cv_acc = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Train and test
        model.fit(X_train_subset, y_train)
        test_acc = model.score(X_test_subset, y_test)
        y_pred_proba = model.predict_proba(X_test_subset)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        result = {
            'model': model_name,
            'transform': transform_name,
            'n_features': n_features,
            'features': top_feature_names,
            'cv_acc': cv_acc,
            'cv_std': cv_std,
            'test_acc': test_acc,
            'roc_auc': roc_auc,
            'model_instance': model,
            'X_test': X_test_subset,
            'feature_indices': feature_indices
        }
        results.append(result)
        
        print(f"  Top {n_features:2d} features: CV Acc: {cv_acc:.4f} | Test Acc: {test_acc:.4f} | ROC AUC: {roc_auc:.4f}")
        print(f"           Features: {', '.join(top_feature_names)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train ROI classifier with feature transformation testing")
    parser.add_argument("--data_path", type=str, 
                       default="training_data/roi_filtering/all_roi_features.npy",
                       help="Path to ROI data file")
    parser.add_argument("--output_dir", type=str,
                       default="roi_classifier/models",
                       help="Directory to save models")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Fraction of data for testing")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--n_estimators", type=int, default=100,
                       help="Number of trees for Random Forest")
    parser.add_argument("--max_depth", type=int, default=None,
                       help="Max depth for Random Forest (None=unlimited)")
    parser.add_argument("--include_auto", action='store_true',
                       help="Include auto-labeled ROIs (default: manual only)")
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    print(f"Loading ROI data from {data_path}...")
    
    manual_only = not args.include_auto
    X, y, feature_names, roi_keys = load_labeled_roi_data(data_path, manual_only=manual_only)
    
    print(f"\n{'='*70}")
    print(f"Dataset Summary")
    print(f"{'='*70}")
    print(f"Total labeled ROIs: {len(X)}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution:")
    print(f"  - Bad (0):  {(y == 0).sum()}")
    print(f"  - Good (1): {(y == 1).sum()}")
    print(f"Training on: {'Manual labels only' if manual_only else 'Manual + Auto labels'}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"  - Bad (0):  {(y_train == 0).sum()}")
    print(f"  - Good (1): {(y_train == 1).sum()}")
    print(f"Test set: {len(X_test)} samples")
    print(f"  - Bad (0):  {(y_test == 0).sum()}")
    print(f"  - Good (1): {(y_test == 1).sum()}")
    
    # Test Random Forest
    rf_results, rf_best = test_model_with_transforms(
        RandomForestClassifier, "Random Forest",
        X_train, X_test, y_train, y_test, feature_names,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    
    # Test Logistic Regression
    lr_results, lr_best = test_model_with_transforms(
        LogisticRegression, "Logistic Regression",
        X_train, X_test, y_train, y_test, feature_names,
        random_state=args.random_state
    )
    
    # Display feature importance for best models
    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE - BEST MODELS")
    print(f"{'='*70}")
    
    print(f"\nRandom Forest (best: {rf_best['transform']}):")
    print(rf_best['feature_importance'].to_string(index=False))
    
    print(f"\nLogistic Regression (best: {lr_best['transform']}):")
    print(lr_best['feature_importance'].to_string(index=False))
    
    # Test feature selection for both models
    print(f"\n{'='*70}")
    print("TESTING FEATURE SUBSETS")
    print(f"{'='*70}")
    
    rf_feature_results = test_feature_selection(
        RandomForestClassifier, "Random Forest",
        X_train, X_test, y_train, y_test, feature_names,
        rf_best['feature_importance'], rf_best['transform'],
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    
    lr_feature_results = test_feature_selection(
        LogisticRegression, "Logistic Regression",
        X_train, X_test, y_train, y_test, feature_names,
        lr_best['feature_importance'], lr_best['transform'],
        random_state=args.random_state
    )
    
    # Compare all configurations
    all_configs = rf_results + lr_results + rf_feature_results + lr_feature_results
    
    # Create comprehensive results table
    results_rows = []
    for r in all_configs:
        n_feat = r.get('n_features', len(feature_names))
        results_rows.append({
            'model': r['model'],
            'transform': r['transform'],
            'n_features': n_feat,
            'cv_accuracy': r['cv_acc'],
            'test_accuracy': r['test_acc'],
            'roc_auc': r['roc_auc']
        })
    
    results_df = pd.DataFrame(results_rows).sort_values('test_accuracy', ascending=False)
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE RESULTS - ALL CONFIGURATIONS")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    
    # Find overall best from all configurations
    best_row = results_df.iloc[0]
    overall_best = None
    for config in all_configs:
        if (config['model'] == best_row['model'] and 
            config['transform'] == best_row['transform'] and
            config.get('n_features', len(feature_names)) == best_row['n_features']):
            overall_best = config
            break
    
    print(f"\n{'='*70}")
    print(f"🏆 OVERALL WINNER: {overall_best['model']} with {overall_best['transform']} transform")
    print(f"   Using {overall_best.get('n_features', len(feature_names))} features")
    print(f"{'='*70}")
    print(f"CV Accuracy: {overall_best['cv_acc']:.4f}")
    print(f"Test Accuracy: {overall_best['test_acc']:.4f}")
    print(f"ROC AUC: {overall_best['roc_auc']:.4f}")
    
    if 'features' in overall_best:
        print(f"Selected features: {', '.join(overall_best['features'])}")
    
    # Final evaluation
    y_pred = overall_best['model_instance'].predict(overall_best['X_test'])
    
    print(f"\n{'='*70}")
    print("CLASSIFICATION REPORT (Best Model)")
    print(f"{'='*70}")
    print(classification_report(y_test, y_pred, target_names=['Bad (0)', 'Good (1)']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              Bad   Good")
    print(f"Actual Bad  [{cm[0,0]:4d}  {cm[0,1]:4d}]")
    print(f"       Good [{cm[1,0]:4d}  {cm[1,1]:4d}]")
    
    # Save best model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"roi_classifier_{timestamp}.joblib"
    joblib.dump(overall_best['model_instance'], model_path)
    print(f"\n💾 Best model saved to {model_path}")
    
    # Save configuration
    config = {
        'model_type': overall_best['model'],
        'transform': overall_best['transform'],
        'feature_names': feature_names,
        'n_features': overall_best.get('n_features', len(feature_names)),
        'selected_features': overall_best.get('features', feature_names),
        'test_accuracy': float(overall_best['test_acc']),
        'roc_auc': float(overall_best['roc_auc']),
        'cv_accuracy': float(overall_best['cv_acc']),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'manual_only': manual_only,
        'all_results': [
            {
                'model': r['model'],
                'transform': r['transform'],
                'n_features': r.get('n_features', len(feature_names)),
                'cv_acc': float(r['cv_acc']),
                'test_acc': float(r['test_acc']),
                'roc_auc': float(r['roc_auc'])
            }
            for r in all_configs
        ]
    }
    
    config_path = output_dir / f"roi_classifier_config_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Configuration saved to {config_path}")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
