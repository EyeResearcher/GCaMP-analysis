"""Train spike classifier model."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import argparse
import logging

logger = logging.getLogger(__name__)

def train_spike_classifier(features_path: Path, output_path: Path, test_size: float = 0.2):
    """
    Train spike classifier on all available features.
    
    Note: Features are expected to be computed from MinMax normalized traces [0,1].
    No additional scaling is needed since normalization is applied consistently
    during both training and inference at the per-video level.
    """
    
    # Load features
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} labeled spikes")
    
    # Feature names - all columns except spike_key and label
    exclude_cols = ['spike_key', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Training with {len(feature_cols)} features")
    
    # Prepare data
    X = df[feature_cols].values
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Create classifier (no scaling needed - features already normalized)
    classifier = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    
    # Train
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Extract feature importances (using absolute coefficients for logistic regression)
    coefficients = np.abs(classifier.coef_[0])
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': coefficients
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importance.to_string(index=False))
    print(f"\nTop 3 features: {feature_importance['feature'].head(3).tolist()}")
    
    # Save model
    model_dict = {
        'classifier': classifier,
        'feature_names': feature_cols,
        'feature_importances': feature_importance.to_dict('records'),
        'performance': {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    }
    
    dump(model_dict, output_path)
    logger.info(f"Saved model to {output_path}")
    
    return classifier, feature_importance

def evaluate_spike_model(model_path: Path, test_features_path: Path):
    """Evaluate trained model on new data."""
    from joblib import load
    
    # Load model
    model_dict = load(model_path)
    pipeline = model_dict['pipeline']
    feature_cols = model_dict['feature_names']
    
    # Load test features
    df = pd.read_csv(test_features_path)
    X_test = df[feature_cols].values
    y_test = df['label'].values
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Report
    print("\nTest Set Evaluation:")
    print(classification_report(y_test, y_pred))
    
    return y_pred

def main():
    parser = argparse.ArgumentParser(description='Train Spike Classifier')
    parser.add_argument('--features', type=Path, 
                       default=Path('training_data/spike_filtering/spike_training_data.csv'),
                       help='Path to labeled training data CSV')
    parser.add_argument('--output', type=Path, default=Path('spike_classifier/models/spike_classifier.pkl'))
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    train_spike_classifier(args.features, args.output, args.test_size)

if __name__ == '__main__':
    main()