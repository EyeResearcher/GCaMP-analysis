"""
Train spike classifiers (Random Forest and Logistic Regression).

Tests different feature transformations and compares model performance.
Provides a clean entry point `train_spike_classifier()` for notebook usage.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split


# =============================================================================
# Data Loading
# =============================================================================



# =============================================================================
# Feature Transformations
# =============================================================================

def create_feature_variants(X: np.ndarray, feature_names: list) -> tuple[dict, dict]:
    """
    Create multiple feature variants with different transformations.
    
    Returns
    -------
    variants : dict
        Mapping transform name to feature matrix
    variant_names : dict
        Mapping transform name to feature name list
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


def apply_transform(X: np.ndarray, transform: str) -> np.ndarray:
    """Apply a single transformation to feature matrix."""
    if transform == 'log':
        return np.log1p(np.abs(X))
    elif transform == 'sqrt':
        return np.sqrt(np.abs(X))
    elif transform == 'square':
        return X ** 2
    else:  # raw
        return X.copy()


# =============================================================================
# Feature Importance
# =============================================================================

def get_feature_importance(model, feature_names: list, transform_name: str) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame with features sorted by importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    # Apply transform suffix to feature names
    suffix_map = {'log': '_log', 'sqrt': '_sqrt', 'square': '_sq', 'raw': ''}
    suffix = suffix_map.get(transform_name, '')
    display_names = [f"{name}{suffix}" for name in feature_names]
    
    importance_df = pd.DataFrame({
        'feature': display_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


def get_feature_ranking(model, feature_names: list) -> list[tuple[str, float]]:
    """Get feature ranking as list of (name, importance) tuples."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return [(name, 0.0) for name in feature_names]
    
    return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)


# =============================================================================
# Model Training & Evaluation
# =============================================================================

def create_model(model_class, n_estimators: int = 100, max_depth: int = None, 
                 random_state: int = 42):
    """Create a model instance with appropriate parameters."""
    if model_class == RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
    else:  # LogisticRegression
        return LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )


def evaluate_model(model, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Train and evaluate a model.
    
    Returns
    -------
    metrics : dict
        Dictionary with cv_acc, cv_std, test_acc, roc_auc
    """
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    cv_acc = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Train and test
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'cv_acc': cv_acc,
        'cv_std': cv_std,
        'test_acc': test_acc,
        'roc_auc': roc_auc
    }


def test_model_with_transforms(
    model_class,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = 42,
    verbose: bool = True
) -> tuple[list, dict]:
    """
    Test a model with different feature transformations.
    
    Returns
    -------
    results : list
        List of result dicts
    best : dict
        Best configuration
    """
    if verbose:
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
        
        model = create_model(model_class, n_estimators, max_depth, random_state)
        metrics = evaluate_model(model, X_train_var, X_test_var, y_train, y_test)
        
        importance_df = get_feature_importance(model, feature_names, transform)
        
        result = {
            'model': model_name,
            'transform': transform,
            'cv_acc': metrics['cv_acc'],
            'cv_std': metrics['cv_std'],
            'test_acc': metrics['test_acc'],
            'roc_auc': metrics['roc_auc'],
            'model_instance': model,
            'X_test': X_test_var,
            'config': {name: transform for name in feature_names},
            'feature_importance': importance_df
        }
        results.append(result)
        
        if verbose:
            print(f"  {transform.upper():6s}: CV Acc: {metrics['cv_acc']:.4f} (+/- {metrics['cv_std']*2:.4f}) | "
                  f"Test Acc: {metrics['test_acc']:.4f} | ROC AUC: {metrics['roc_auc']:.4f}")
        
        if metrics['test_acc'] > best['test_acc']:
            best = result
            if verbose:
                print(f"  ✓ New best!")
    
    return results, best


def test_feature_selection(
    model_class,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    importance_df: pd.DataFrame,
    transform_name: str,
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = 42,
    verbose: bool = True
) -> list:
    """
    Test model with top N features based on importance.
    
    Returns
    -------
    results : list
        Results for different feature subset sizes
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"FEATURE SELECTION - {model_name.upper()} with {transform_name.upper()} transform")
        print(f"{'='*70}")
    
    # Apply transformation
    X_train_transformed = apply_transform(X_train, transform_name)
    X_test_transformed = apply_transform(X_test, transform_name)
    
    results = []
    feature_subsets = [3, 5, 7, 10, len(feature_names)]
    
    for n_features in feature_subsets:
        n_features = min(n_features, len(feature_names))
        
        # Get top N feature indices
        top_features = importance_df.head(n_features)['feature'].tolist()
        
        # Map back to original feature names (remove transform suffix)
        suffix_map = {'log': '_log', 'sqrt': '_sqrt', 'square': '_sq'}
        if transform_name != 'raw' and transform_name in suffix_map:
            suffix = suffix_map[transform_name]
            top_feature_names = [f.replace(suffix, '') for f in top_features]
        else:
            top_feature_names = top_features
        
        # Get feature indices
        feature_indices = [feature_names.index(f) for f in top_feature_names if f in feature_names]
        
        if len(feature_indices) == 0:
            continue
        
        # Select features
        X_train_subset = X_train_transformed[:, feature_indices]
        X_test_subset = X_test_transformed[:, feature_indices]
        
        model = create_model(model_class, n_estimators, max_depth, random_state)
        metrics = evaluate_model(model, X_train_subset, X_test_subset, y_train, y_test)
        
        result = {
            'model': model_name,
            'transform': transform_name,
            'n_features': len(feature_indices),
            'features': top_feature_names[:len(feature_indices)],
            'cv_acc': metrics['cv_acc'],
            'cv_std': metrics['cv_std'],
            'test_acc': metrics['test_acc'],
            'roc_auc': metrics['roc_auc'],
            'model_instance': model,
            'X_test': X_test_subset,
            'feature_indices': feature_indices
        }
        results.append(result)
        
        if verbose:
            print(f"  Top {len(feature_indices):2d} features: CV Acc: {metrics['cv_acc']:.4f} | "
                  f"Test Acc: {metrics['test_acc']:.4f} | ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"           Features: {', '.join(top_feature_names[:len(feature_indices)])}")
    
    return results


# =============================================================================
# Results Analysis
# =============================================================================

def find_overall_best(all_configs: list, feature_names: list) -> tuple[dict, pd.DataFrame]:
    """Find the best model configuration from all tested configurations."""
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
    best_row = results_df.iloc[0]
    
    overall_best = None
    for config in all_configs:
        if (config['model'] == best_row['model'] and 
            config['transform'] == best_row['transform'] and
            config.get('n_features', len(feature_names)) == best_row['n_features']):
            overall_best = config
            break
    
    return overall_best, results_df


def print_results_summary(results_df: pd.DataFrame, overall_best: dict, 
                         feature_names: list, y_test: np.ndarray) -> None:
    """Print comprehensive results summary."""
    print(f"\n{'='*70}")
    print("COMPREHENSIVE RESULTS - ALL CONFIGURATIONS")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    
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


# =============================================================================
# Model Saving
# =============================================================================

def save_model_and_config(
    overall_best: dict,
    feature_names: list,
    output_dir: Path,
    X_train: np.ndarray,
    X_test: np.ndarray,
    all_configs: list
) -> tuple[Path, Path]:
    """
    Save the best model and its configuration.
    
    Returns
    -------
    model_path : Path
        Path to saved model
    config_path : Path
        Path to saved config
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"spike_classifier_{timestamp}.joblib"
    joblib.dump(overall_best['model_instance'], model_path)
    print(f"\n💾 Best model saved to {model_path}")
    
    # Get feature ranking from best model
    feature_ranking = get_feature_ranking(overall_best['model_instance'], 
                                          overall_best.get('features', feature_names))
    
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
        'feature_ranking': [(feat, float(imp)) for feat, imp in feature_ranking],
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
    
    config_path = output_dir / f"spike_classifier_config_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Configuration saved to {config_path}")
    
    return model_path, config_path


# =============================================================================
# Main Entry Point
# =============================================================================

def train_spike_classifier(
    data_path: Path,
    output_dir: Path = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = None,
    save_model: bool = True,
    verbose: bool = True
) -> dict:
    """
    Train spike classifier with feature transformation testing.
    
    This is the main entry point for training. Tests Random Forest and 
    Logistic Regression with different feature transformations and 
    feature subsets.
    
    Parameters
    ----------
    data_path : Path
        Path to ROI data .npy file (spike CSV will be inferred)
    output_dir : Path, optional
        Directory to save models. If None, models won't be saved.
    test_size : float
        Fraction of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    n_estimators : int
        Number of trees for Random Forest (default: 100)
    max_depth : int or None
        Max depth for Random Forest (default: None = unlimited)
    save_model : bool
        Whether to save the best model (default: True)
    verbose : bool
        Print detailed output (default: True)
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'best_model': The trained best model instance
        - 'config': Configuration dict with model settings
        - 'results_df': DataFrame with all results
        - 'feature_names': List of feature names
        - 'model_path': Path to saved model (if save_model=True)
        - 'config_path': Path to saved config (if save_model=True)
    """
    data_path = Path(data_path)
    
    # Load data
    if verbose:
        print(f"Loading spike data from {data_path}...")
    
    X, y, feature_names, spike_keys = load_labeled_spike_data(data_path)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Dataset Summary")
        print(f"{'='*70}")
        print(f"Total labeled spikes: {len(X)}")
        print(f"Number of features: {len(feature_names)}")
        print(f"Feature names: {feature_names}")
        print(f"Label distribution:")
        print(f"  - Bad (0):  {(y == 0).sum()}")
        print(f"  - Good (1): {(y == 1).sum()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if verbose:
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
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        verbose=verbose
    )
    
    # Test Logistic Regression
    lr_results, lr_best = test_model_with_transforms(
        LogisticRegression, "Logistic Regression",
        X_train, X_test, y_train, y_test, feature_names,
        random_state=random_state,
        verbose=verbose
    )
    
    # Display feature importance for best models
    if verbose:
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE - BEST MODELS")
        print(f"{'='*70}")
        
        print(f"\nRandom Forest (best: {rf_best['transform']}):")
        print(rf_best['feature_importance'].to_string(index=False))
        
        print(f"\nLogistic Regression (best: {lr_best['transform']}):")
        print(lr_best['feature_importance'].to_string(index=False))
    
    # Test feature selection for both models
    rf_feature_results = test_feature_selection(
        RandomForestClassifier, "Random Forest",
        X_train, X_test, y_train, y_test, feature_names,
        rf_best['feature_importance'], rf_best['transform'],
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        verbose=verbose
    )
    
    lr_feature_results = test_feature_selection(
        LogisticRegression, "Logistic Regression",
        X_train, X_test, y_train, y_test, feature_names,
        lr_best['feature_importance'], lr_best['transform'],
        random_state=random_state,
        verbose=verbose
    )
    
    # Combine all configurations
    all_configs = rf_results + lr_results + rf_feature_results + lr_feature_results
    
    # Find overall best
    overall_best, results_df = find_overall_best(all_configs, feature_names)
    
    if verbose:
        print_results_summary(results_df, overall_best, feature_names, y_test)
    
    # Build config dict
    feature_ranking = get_feature_ranking(overall_best['model_instance'],
                                          overall_best.get('features', feature_names))
    
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
        'feature_ranking': [(feat, float(imp)) for feat, imp in feature_ranking]
    }
    
    # Prepare result dict
    result = {
        'best_model': overall_best['model_instance'],
        'config': config,
        'results_df': results_df,
        'feature_names': feature_names,
        'all_configs': all_configs,
        'model_path': None,
        'config_path': None
    }
    
    # Save model if requested
    if save_model and output_dir is not None:
        output_dir = Path(output_dir)
        model_path, config_path = save_model_and_config(
            overall_best, feature_names, output_dir,
            X_train, X_test, all_configs
        )
        result['model_path'] = model_path
        result['config_path'] = config_path
    
    if verbose:
        print("\n✅ Training complete!")
    
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train spike classifier with feature transformation testing"
    )
    parser.add_argument("--data_path", type=str,
                        default="training_data/roi_filtering/all_roi_features.npy",
                        help="Path to ROI data file")
    parser.add_argument("--output_dir", type=str,
                        default="trained_models/spike_classifier",
                        help="Directory to save models")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data for testing")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees for Random Forest")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Max depth for Random Forest (None=unlimited)")
    args = parser.parse_args()

    train_spike_classifier(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        save_model=True,
        verbose=True
    )


if __name__ == "__main__":
    main()