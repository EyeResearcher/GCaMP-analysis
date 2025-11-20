import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def main():
    parser = argparse.ArgumentParser(description="Train ROI classifier")
    parser.add_argument("--data_path", type=str, help="Path to ROI data file", 
                       default="training_data/roi_filtering/all_roi_features.npy")
    parser.add_argument("--test_size", type=float, default=0.2, 
                       help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, 
                       help="Random seed (default: 42)")
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    npy_dict = np.load(data_path, allow_pickle=True).item()
    
    rows = []
    for roi_key in npy_dict.keys():
        label = npy_dict[roi_key]['label']
        if label == -1:
            continue  # Skip unlabeled ROIs
        features = list(npy_dict[roi_key]['features'].values())
        row = features + [label]
        rows.append(row)
    
    if len(rows) == 0:
        print("❌ No labeled data found! Please annotate some ROIs first.")
        return
    
    data_array = np.array(rows)
    print(f"Loaded {len(rows)} labeled ROIs")
    
    # Split features and labels
    X = data_array[:, :2]  # First 2 columns as features
    y = data_array[:, -1]  # Last column as label
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train logistic regression
    print("\n" + "="*50)
    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(random_state=args.random_state, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"✓ Training complete!")
    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Detailed metrics
    print("\n" + "="*50)
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Bad (0)', 'Good (1)']))
    
    print("Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print("[[TN FP]\n [FN TP]]")
    
    # Save model
    output_dir = Path("training_data/roi_filtering")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "roi_classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'coefficients': clf.coef_[0].tolist(),
        'intercept': float(clf.intercept_[0])
    }
    metadata_path = output_dir / "roi_classifier_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()