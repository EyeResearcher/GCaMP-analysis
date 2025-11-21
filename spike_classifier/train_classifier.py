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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import joblib
from itertools import product


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


def train_random_forest(X, y, feature_names, test_size=0.2, random_state=42, n_estimators=100, max_depth=None):
    """
    Train a Random Forest classifier and evaluate its performance.
    
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
    scaler : MinMaxScaler or None
        Fitted scaler (if used)
    feature_config : tuple
        Configuration of which features were scaled
    """
    # Split data first (before any scaling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Try all combinations of raw vs scaled for each feature
    n_features = X.shape[1]
    
    # Generate all binary combinations (0=raw, 1=scaled)
    # This gives us 2^n_features combinations
    all_configs = list(product([False, True], repeat=n_features))
    
    print(f"\nTesting {len(all_configs)} feature scaling combinations...")
    print(f"  (0 = raw feature, 1 = MinMax scaled feature)\n")
    
    best_score = -1
    best_clf = None
    best_config = None
    best_scaler = None
    best_X_train = None
    best_X_test = None
    
    results = []
    
    for config_idx, config in enumerate(all_configs):
        # Create a copy of the data
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        
        # Apply scaling to selected features
        scaler = None
        if any(config):  # If any feature should be scaled
            scaler = MinMaxScaler()
            scaled_features = [i for i, scale in enumerate(config) if scale]
            
            # Fit scaler on training data only
            X_train_copy[:, scaled_features] = scaler.fit_transform(X_train_copy[:, scaled_features])
            X_test_copy[:, scaled_features] = scaler.transform(X_test_copy[:, scaled_features])
        
        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        clf.fit(X_train_copy, y_train)
        
        # Evaluate on test set
        test_score = clf.score(X_test_copy, y_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(clf, X_train_copy, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # ROC AUC
        y_pred_proba = clf.predict_proba(X_test_copy)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Build config string
        config_str = ''.join(['S' if scale else 'R' for scale in config])
        feature_config_names = [f"{name}={'scaled' if scale else 'raw'}" for name, scale in zip(feature_names, config)]
        
        results.append({
            'config_idx': config_idx,
            'config': config_str,
            'cv_accuracy': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': test_score,
            'roc_auc': roc_auc,
            'config_tuple': config
        })
        
        # Print progress more frequently
        if (config_idx + 1) % 10 == 0 or config_idx == 0:
            print(f"  [{config_idx + 1}/{len(all_configs)}] Config: {config_str} | CV Acc: {cv_mean:.4f} | Test Acc: {test_score:.4f} | ROC AUC: {roc_auc:.4f}")
        
        # Track best model
        if test_score > best_score:
            best_score = test_score
            best_clf = clf
            best_config = config
            best_scaler = scaler
            best_X_train = X_train_copy
            best_X_test = X_test_copy
            print(f"  ✓ New best! Config: {config_str} | Test Acc: {test_score:.4f}")
    
    # Sort results by test accuracy
    results_df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 10 FEATURE SCALING CONFIGURATIONS")
    print("="*80)
    print(results_df.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION DETAILS")
    print("="*80)
    best_config_str = ''.join(['S' if scale else 'R' for scale in best_config])
    print(f"\nBest configuration: {best_config_str}")
    print("Feature scaling:")
    for i, (name, scale) in enumerate(zip(feature_names, best_config)):
        print(f"  {name}: {'MinMax scaled' if scale else 'raw'}")
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"  - Good spikes: {(y_train == 1).sum()}")
    print(f"  - Bad spikes: {(y_train == 0).sum()}")
    print(f"Test set: {len(X_test)} samples")
    print(f"  - Good spikes: {(y_test == 1).sum()}")
    print(f"  - Bad spikes: {(y_test == 0).sum()}")
    
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
        'scaling': ['scaled' if scale else 'raw' for scale in best_config]
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    print(feature_importance.to_string(index=False))
    
    return best_clf, best_X_test, y_test, best_scaler, best_config


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
    
    # Train classifier (will test all feature scaling combinations)
    clf, X_test, y_test, scaler, best_config = train_random_forest(
        X, y, feature_names,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"\n✓ Model saved to: {output_path}")
    
    # Save scaler if used
    if scaler is not None:
        scaler_path = output_path.parent / 'scaler.joblib'
        joblib.dump(scaler, scaler_path)
        print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save feature names and scaling configuration
    feature_names_path = output_path.parent / 'feature_names.txt'
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f"✓ Feature names saved to: {feature_names_path}")
    
    # Save scaling configuration
    config_path = output_path.parent / 'scaling_config.txt'
    with open(config_path, 'w') as f:
        for name, scale in zip(feature_names, best_config):
            f.write(f"{name}: {'scaled' if scale else 'raw'}\n")
    print(f"✓ Scaling configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
