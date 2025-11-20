"""Train ROI classifier model."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import argparse
import logging

logger = logging.getLogger(__name__)

def train_roi_classifier(features_path: Path, 
                        output_path: Path, 
                        test_size: float = 0.15,
                        normalization: str = 'minmax'):
    """
    Train ROI classifier on labeled features.
    
    Args:
        features_path: Path to features CSV
        output_path: Path to save model
        test_size: Fraction for test set (use smaller value to train on more data)
        normalization: Normalization strategy used ('minmax' or 'deltaf')
    """
    
    # Load features
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} labeled ROIs")
    logger.info(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Prepare data
    X = df[['derivative_skew', 'spike_prom_mean']].values
    y = df['label'].values
    
    # Use substantial amount of data for training (85% train, 15% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Create classifier
    # Note: Features are already normalized during extraction using the same strategy
    # for both training and inference, so we don't need additional scaling
    classifier = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Train with cross-validation
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Train on full training set
    classifier.fit(X_train, y_train)
    
    # Evaluate on train set
    y_train_pred = classifier.predict(X_train)
    train_acc = (y_train_pred == y_train).mean()
    logger.info(f"Training accuracy: {train_acc:.3f}")
    
    # Evaluate on test set
    y_pred = classifier.predict(X_test)
    test_acc = (y_pred == y_test).mean()
    logger.info(f"Test accuracy: {test_acc:.3f}")
    
    print("\n" + "="*60)
    print(f"ROI Classifier Training Results ({normalization} normalization)")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bad', 'Good']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives (Bad correctly classified): {cm[0,0]}")
    print(f"False Positives (Bad classified as Good): {cm[0,1]}")
    print(f"False Negatives (Good classified as Bad): {cm[1,0]}")
    print(f"True Positives (Good correctly classified): {cm[1,1]}")
    
    # Calculate important metrics
    precision_good = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
    recall_good = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    print(f"\nGood ROI Precision: {precision_good:.3f}")
    print(f"Good ROI Recall: {recall_good:.3f}")
    
    # Feature importance (coefficients)
    coef = classifier.coef_[0]
    print(f"\nFeature Coefficients:")
    print(f"  derivative_skew: {coef[0]:.3f}")
    print(f"  spike_prom_mean: {coef[1]:.3f}")
    
    # Save model
    model_dict = {
        'classifier': classifier,
        'feature_names': ['derivative_skew', 'spike_prom_mean'],
        'normalization': normalization,
        'performance': {
            'test_accuracy': test_acc,
            'train_accuracy': train_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm.tolist()
        },
        'training_info': {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'test_size': test_size
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model_dict, output_path)
    logger.info(f"Saved model to {output_path}")
    
    return classifier

def evaluate_roi_model(model_path: Path, test_features_path: Path):
    """Evaluate trained model on new data."""
    from joblib import load
    
    # Load model
    model_dict = load(model_path)
    classifier = model_dict.get('classifier') or model_dict.get('pipeline')  # Support both old and new format
    
    # Load test features
    df = pd.read_csv(test_features_path)
    X_test = df[['derivative_skew', 'spike_prom_mean']].values
    y_test = df['label'].values
    
    # Predict
    y_pred = classifier.predict(X_test)
    
    # Report
    print("\nTest Set Evaluation:")
    print(classification_report(y_test, y_pred))
    
    return y_pred

def main():
    parser = argparse.ArgumentParser(description='Train ROI Classifier')
    parser.add_argument('--features', type=Path, required=True, help='Path to features CSV')
    parser.add_argument('--output', type=Path, default=Path('roi_classifier/models/roi_classifier.pkl'))
    parser.add_argument('--test_size', type=float, default=0.15, help='Test set fraction (default: 0.15 to use 85% for training)')
    parser.add_argument('--normalization', type=str, default='minmax', choices=['minmax', 'deltaf'], 
                       help='Normalization strategy used during feature extraction')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_roi_classifier(args.features, args.output, args.test_size, args.normalization)

if __name__ == '__main__':
    main()