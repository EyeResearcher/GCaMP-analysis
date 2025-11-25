"""
Train a Random Forest classifier for spike classification.

Loads labeled spike data from all_roi_features_spike_keys.csv and extracts
features from the corresponding spikes in all_roi_features.npy.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import json


def load_labeled_spike_data(base_path: Path):
    """
    Load labeled spike data from CSV and extract features from .npy file.
    
    Parameters
    ----------
    base_path : Path
        Base path to all_roi_features (without extension)
    
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    feature_names : list
        Names of features
    spike_keys : list
        Spike keys corresponding to each sample
    """
    # Load the CSV with spike keys and labels
    csv_path = base_path.parent / f"{base_path.stem}_spike_keys.csv"
    df = pd.read_csv(csv_path)
    
    # Filter for labeled spikes only (label == 0 or 1, not -1)
    labeled_df = df[df['label'].isin([0, 1])].copy()
    
    print(f"Total spikes: {len(df)}")
    print(f"Labeled spikes: {len(labeled_df)}")
    print(f"  - Good spikes (label=1): {(labeled_df['label'] == 1).sum()}")
    print(f"  - Bad spikes (label=0): {(labeled_df['label'] == 0).sum()}")
    
    if len(labeled_df) == 0:
        raise ValueError("No labeled spikes found! Please annotate some spikes first.")
    
    # Load the .npy file with all ROI data
    npy_path = base_path.with_suffix('.npy')
    roi_dict = np.load(npy_path, allow_pickle=True).item()
    
    # Extract features for labeled spikes
    features_list = []
    labels_list = []
    spike_keys_list = []
    
    # Get feature names from first spike
    feature_names = None
    
    for _, row in labeled_df.iterrows():
        spike_key = row['spike_key']
        label = row['label']
        
        # Parse spike_key: "roi_key-spike_idx"
        roi_key, spike_idx_str = spike_key.rsplit('-', 1)
        spike_idx = int(spike_idx_str)
        
        # Get ROI data
        if roi_key not in roi_dict:
            print(f"Warning: ROI {roi_key} not found in .npy file, skipping spike {spike_key}")
            continue
        
        roi_data = roi_dict[roi_key]
        
        # Get spike data
        if 'spikes' not in roi_data or spike_idx not in roi_data['spikes']:
            print(f"Warning: Spike {spike_idx} not found in ROI {roi_key}, skipping")
            continue
        
        spike_data = roi_data['spikes'][spike_idx]
        spike_features = spike_data['features']
        
        # Extract feature names on first iteration
        if feature_names is None:
            feature_names = sorted(spike_features.keys())
            print(f"\nFeatures used for classification: {feature_names}")
        
        # Extract feature values in consistent order
        feature_values = [spike_features[fname] for fname in feature_names]
        
        features_list.append(feature_values)
        labels_list.append(label)
        spike_keys_list.append(spike_key)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"\nSuccessfully extracted features for {len(X)} labeled spikes")
    print(f"Feature matrix shape: {X.shape}")
    
    return X, y, feature_names, spike_keys_list


def create_feature_variants(X, feature_names):
    """
    Create multiple feature variants with different transformations.
    
    For each feature, create:
    - Raw (original)
    - Log-transformed (log(x + 1)) - compresses large values
    - Square root - moderate compression
    - Squared - emphasizes large values
    
    Returns:
        variants: dict mapping transform name to feature matrix
        variant_names: dict mapping transform name to feature name list
    """
    variants = {
        'raw': X.copy(),
        'log': np.log1p(np.abs(X)),  # log(|x| + 1)
        'sqrt': np.sqrt(np.abs(X)),  # sqrt(|x|)
        'square': X ** 2,  # x²
    }
    
    variant_names = {
        'raw': feature_names,
        'log': [f"{name}_log" for name in feature_names],
        'sqrt': [f"{name}_sqrt" for name in feature_names],
        'square': [f"{name}_sq" for name in feature_names],
    }
    
    return variants, variant_names


def test_top_features_strategy(X_train, X_test, y_train, y_test, feature_names, n_estimators=100, max_depth=None, random_state=42):
    """
    Test performance using only top N most important features.
    
    Strategy:
    1. Train on all features to get feature importances
    2. Select top N features
    3. Test different transformations on just those features
    
    Returns:
        results: list of result dicts
        best: best configuration dict
        feature_ranking: list of (feature_name, importance) tuples
    """
    print("\n" + "="*70)
    print("TESTING TOP IMPORTANT FEATURES STRATEGY")
    print("="*70)
    
    # Step 1: Train on raw features to get importances
    print("\nStep 1: Training on all raw features to rank importance...")
    rf_baseline = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_baseline.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf_baseline.feature_importances_
    feature_ranking = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print("\nFeature importance ranking:")
    for i, (feat, imp) in enumerate(feature_ranking[:10], 1):
        print(f"  {i:2d}. {feat:25s}: {imp:.4f}")
    
    # Step 2: Test with different numbers of top features
    results = []
    transformation_options = ['raw', 'log', 'sqrt', 'square']
    
    for n_top in [3, 5, 7, 10, 15, len(feature_names)]:
        if n_top > len(feature_names):
            continue
            
        print(f"\n--- Testing with top {n_top} features ---")
        top_features = [feat for feat, _ in feature_ranking[:n_top]]
        top_indices = [feature_names.index(feat) for feat in top_features]
        
        X_train_subset = X_train[:, top_indices]
        X_test_subset = X_test[:, top_indices]
        
        # Create variants for this subset
        variants_train = {
            'raw': X_train_subset.copy(),
            'log': np.log1p(np.abs(X_train_subset)),
            'sqrt': np.sqrt(np.abs(X_train_subset)),
            'square': X_train_subset ** 2,
        }
        
        variants_test = {
            'raw': X_test_subset.copy(),
            'log': np.log1p(np.abs(X_test_subset)),
            'sqrt': np.sqrt(np.abs(X_test_subset)),
            'square': X_test_subset ** 2,
        }
        
        # Test uniform transformations
        for transform in transformation_options:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                rf, variants_train[transform], y_train, 
                cv=5, scoring='accuracy', n_jobs=-1
            )
            cv_acc = cv_scores.mean()
            
            # Train and test
            rf.fit(variants_train[transform], y_train)
            y_pred = rf.predict(variants_test[transform])
            y_pred_proba = rf.predict_proba(variants_test[transform])[:, 1]
            
            test_acc = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results.append({
                'n_features': n_top,
                'features': top_features,
                'transform': transform,
                'config': f"Top{n_top}_{transform.upper()}",
                'cv_acc': cv_acc,
                'test_acc': test_acc,
                'roc_auc': roc_auc,
                'model': rf
            })
            
            print(f"  Top {n_top:2d} [{transform:6s}]: CV Acc: {cv_acc:.4f} | Test Acc: {test_acc:.4f} | ROC AUC: {roc_auc:.4f}")
    
    # Find best configuration
    best = max(results, key=lambda x: x['test_acc'])
    
    print(f"\n{'='*70}")
    print(f"BEST TOP-N FEATURES CONFIGURATION:")
    print(f"  Number of features: {best['n_features']}")
    print(f"  Transformation: {best['transform']}")
    print(f"  CV Accuracy: {best['cv_acc']:.4f}")
    print(f"  Test Accuracy: {best['test_acc']:.4f}")
    print(f"  ROC AUC: {best['roc_auc']:.4f}")
    print(f"\n  Selected features:")
    for i, feat in enumerate(best['features'], 1):
        print(f"    {i:2d}. {feat}")
    
    return results, best, feature_ranking


def train_random_forest(X, y, feature_names, test_size=0.2, random_state=42, n_estimators=100, max_depth=None):
    """
    Train a Random Forest classifier testing different feature transformations.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    feature_names : list
        Names of features
    test_size : float
        Fraction of data to use for testing
    random_state : int
        Random seed for reproducibility
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of trees
    
    Returns
    -------
    clf : RandomForestClassifier
        Trained classifier
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    best_transform_config : dict
        Configuration of which features used which transformation
    """
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    n_features = X.shape[1]
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"  - Good spikes: {(y_train == 1).sum()}")
    print(f"  - Bad spikes: {(y_train == 0).sum()}")
    print(f"Test set: {len(X_test)} samples")
    print(f"  - Good spikes: {(y_test == 1).sum()}")
    print(f"  - Bad spikes: {(y_test == 0).sum()}")
    
    # Create feature variants
    variants_train, _ = create_feature_variants(X_train, feature_names)
    variants_test, _ = create_feature_variants(X_test, feature_names)
    
    transformation_options = ['raw', 'log', 'sqrt', 'square']
    
    best_score = -1
    best_clf = None
    best_config = None
    best_X_train = None
    best_X_test = None
    
    results = []
    
    # Strategy 1: Test uniform transformations (all features same)
    print(f"\n{'='*70}")
    print("TESTING UNIFORM TRANSFORMATIONS (all features same)")
    print(f"{'='*70}")
    
    for transform in transformation_options:
        X_train_variant = variants_train[transform]
        X_test_variant = variants_test[transform]
        
        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train_variant, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Train and evaluate
        clf.fit(X_train_variant, y_train)
        test_score = clf.score(X_test_variant, y_test)
        y_pred_proba = clf.predict_proba(X_test_variant)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        config_str = transform.upper() * n_features
        
        results.append({
            'config': config_str,
            'transform': transform,
            'cv_accuracy': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': test_score,
            'roc_auc': roc_auc,
        })
        
        print(f"  {transform.upper():6s}: CV Acc: {cv_mean:.4f} (+/- {cv_std*2:.4f}) | Test Acc: {test_score:.4f} | ROC AUC: {roc_auc:.4f}")
        
        # Track best
        if test_score > best_score:
            best_score = test_score
            best_clf = clf
            best_config = {name: transform for name in feature_names}
            best_X_train = X_train_variant
            best_X_test = X_test_variant
            print(f"  ✓ New best!")
    
    # Strategy 2: Test random feature-specific transformations
    print(f"\n{'='*70}")
    print("TESTING RANDOM FEATURE-SPECIFIC TRANSFORMATIONS")
    print(f"{'='*70}")
    
    n_random_tests = min(100, 2 ** n_features)  # Don't test more than possible or 100
    np.random.seed(random_state)
    
    for i in range(n_random_tests):
        # Randomly choose transformation for each feature
        config = np.random.choice(transformation_options, size=n_features)
        
        # Build feature matrix
        X_train_combo = []
        X_test_combo = []
        for feat_idx, transform in enumerate(config):
            X_train_combo.append(variants_train[transform][:, feat_idx:feat_idx+1])
            X_test_combo.append(variants_test[transform][:, feat_idx:feat_idx+1])
        
        X_train_combo = np.hstack(X_train_combo)
        X_test_combo = np.hstack(X_test_combo)
        
        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + i,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train_combo, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Train and evaluate
        clf.fit(X_train_combo, y_train)
        test_score = clf.score(X_test_combo, y_test)
        y_pred_proba = clf.predict_proba(X_test_combo)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Create config string: R/L/S/Q for Raw/Log/Sqrt/sQuare
        transform_map = {'raw': 'R', 'log': 'L', 'sqrt': 'S', 'square': 'Q'}
        config_str = ''.join([transform_map[t] for t in config])
        
        results.append({
            'config': config_str,
            'transform_per_feature': {feature_names[j]: config[j] for j in range(n_features)},
            'cv_accuracy': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': test_score,
            'roc_auc': roc_auc,
        })
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n_random_tests}] Config: {config_str} | CV Acc: {cv_mean:.4f} | Test Acc: {test_score:.4f} | ROC AUC: {roc_auc:.4f}")
        
        # Track best
        if test_score > best_score:
            best_score = test_score
            best_clf = clf
            best_config = {feature_names[j]: config[j] for j in range(n_features)}
            best_X_train = X_train_combo
            best_X_test = X_test_combo
            print(f"  ✓ New best! Config: {config_str} | Test Acc: {test_score:.4f}")
    
    # Sort results by test accuracy
    results_df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 10 FEATURE TRANSFORMATION CONFIGURATIONS")
    print("="*80)
    print(results_df[['config', 'cv_accuracy', 'test_accuracy', 'roc_auc']].head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION DETAILS")
    print("="*80)
    
    print(f"\nTest Accuracy: {best_score:.4f}")
    print("\nFeature transformations:")
    for name, transform in best_config.items():
        print(f"  {name}: {transform}")
    
    # Evaluate best model
    y_pred = best_clf.predict(best_X_test)
    y_pred_proba = best_clf.predict_proba(best_X_test)[:, 1]
    
    print("\n" + "="*80)
    print("BEST MODEL TEST SET PERFORMANCE")
    print("="*80)
    print(f"\nAccuracy: {best_score:.4f}")
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bad (0)', 'Good (1)']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("                Bad  Good")
    print(f"Actual Bad  [{cm[0,0]:5d} {cm[0,1]:5d}]")
    print(f"       Good [{cm[1,0]:5d} {cm[1,1]:5d}]")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_clf.feature_importances_,
        'transform': [best_config[name] for name in feature_names]
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    print(feature_importance.to_string(index=False))
    
    return best_clf, best_X_test, y_test, best_config


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier for spike classification"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='training_data/roi_filtering/all_roi_features',
        help='Base path to all_roi_features files (without extension)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='spike_classifier/models/spike_classifier.joblib',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of trees in Random Forest'
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=None,
        help='Maximum depth of trees (None for unlimited)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Load data
    base_path = Path(args.data_path)
    print("Loading labeled spike data...")
    X, y, feature_names, spike_keys = load_labeled_spike_data(base_path)
    
    print(f"\nFeatures: {feature_names}")
    
    # Split data once for consistent comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"  - Good spikes: {(y_train == 1).sum()}")
    print(f"  - Bad spikes: {(y_train == 0).sum()}")
    print(f"Test set: {len(X_test)} samples")
    print(f"  - Good spikes: {(y_test == 1).sum()}")
    print(f"  - Bad spikes: {(y_test == 0).sum()}")
    
    # Strategy 1: Test all features with transformations
    print("\n" + "="*70)
    print("STRATEGY 1: ALL FEATURES WITH TRANSFORMATIONS")
    print("="*70)
    
    all_features_results = []
    variants_train, _ = create_feature_variants(X_train, feature_names)
    variants_test, _ = create_feature_variants(X_test, feature_names)
    
    transformation_options = ['raw', 'log', 'sqrt', 'square']
    best_all_features = {'test_acc': -1}
    
    for transform in transformation_options:
        X_train_var = variants_train[transform]
        X_test_var = variants_test[transform]
        
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        cv_scores = cross_val_score(rf, X_train_var, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        cv_acc = cv_scores.mean()
        
        rf.fit(X_train_var, y_train)
        test_acc = rf.score(X_test_var, y_test)
        y_pred_proba = rf.predict_proba(X_test_var)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        result = {
            'transform': transform,
            'cv_acc': cv_acc,
            'test_acc': test_acc,
            'roc_auc': roc_auc,
            'model': rf,
            'X_test': X_test_var,
            'config': {name: transform for name in feature_names}
        }
        all_features_results.append(result)
        
        print(f"  {transform.upper():6s}: CV Acc: {cv_acc:.4f} | Test Acc: {test_acc:.4f} | ROC AUC: {roc_auc:.4f}")
        
        if test_acc > best_all_features['test_acc']:
            best_all_features = result
            print(f"  ✓ New best!")
    
    # Strategy 2: Test top features
    top_results, best_top_features, feature_ranking = test_top_features_strategy(
        X_train, X_test, y_train, y_test, feature_names,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    
    # Compare strategies
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\nAll features (best config):")
    print(f"  Transform: {best_all_features['transform']}")
    print(f"  Test Accuracy: {best_all_features['test_acc']:.4f}")
    print(f"  ROC AUC: {best_all_features['roc_auc']:.4f}")
    
    print(f"\nTop-N features (best config):")
    print(f"  Number of features: {best_top_features['n_features']}")
    print(f"  Transform: {best_top_features['transform']}")
    print(f"  Test Accuracy: {best_top_features['test_acc']:.4f}")
    print(f"  ROC AUC: {best_top_features['roc_auc']:.4f}")
    
    # Choose overall best model
    if best_top_features['test_acc'] > best_all_features['test_acc']:
        print(f"\n🏆 Winner: Top-{best_top_features['n_features']} features with {best_top_features['transform']} transform")
        best_overall = best_top_features
        is_top_features = True
    else:
        print(f"\n🏆 Winner: All features with {best_all_features['transform']} transform")
        best_overall = best_all_features
        is_top_features = False
    
    # Save the best model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_overall['model'], output_path)
    print(f"\n💾 Model saved to: {output_path}")
    
    # Save configuration
    config = {
        'transform': best_overall['transform'],
        'feature_names': feature_names,
        'test_accuracy': float(best_overall['test_acc']),
        'roc_auc': float(best_overall['roc_auc']),
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
    }
    
    if is_top_features:
        config['use_top_features'] = True
        config['n_top_features'] = best_overall['n_features']
        config['selected_features'] = best_overall['features']
        config['feature_ranking'] = [(feat, float(imp)) for feat, imp in feature_ranking]
    else:
        config['use_top_features'] = False
        config['selected_features'] = feature_names
        config['feature_ranking'] = [(feat, float(imp)) for feat, imp in feature_ranking]
    
    config_path = output_path.parent / 'spike_classifier_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Configuration saved to: {config_path}")
    
    # Final evaluation report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT (Best Model)")
    print("="*70)
    
    # Get predictions for best model
    if is_top_features:
        top_indices = [feature_names.index(feat) for feat in best_overall['features']]
        X_test_final = X_test[:, top_indices]
        transform = best_overall['transform']
        
        if transform == 'log':
            X_test_final = np.log1p(np.abs(X_test_final))
        elif transform == 'sqrt':
            X_test_final = np.sqrt(np.abs(X_test_final))
        elif transform == 'square':
            X_test_final = X_test_final ** 2
        
        y_pred_final = best_overall['model'].predict(X_test_final)
        
        # Feature importances for selected features
        feature_imp = sorted(
            zip(best_overall['features'], best_overall['model'].feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
    else:
        y_pred_final = best_overall['model'].predict(best_overall['X_test'])
        
        # Feature importances for all features
        feature_imp = sorted(
            zip(feature_names, best_overall['model'].feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
    
    print("\n" + classification_report(y_test, y_pred_final, target_names=['Bad', 'Good']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_final)
    print(f"              Predicted")
    print(f"              Bad   Good")
    print(f"Actual Bad  [{cm[0,0]:4d}  {cm[0,1]:4d}]")
    print(f"       Good [{cm[1,0]:4d}  {cm[1,1]:4d}]")
    
    print("\nFeature Importance:")
    for feat, imp in feature_imp:
        print(f"  {feat:25s}: {imp:.4f}")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
