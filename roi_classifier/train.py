"""Train ROI classifier model."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import argparse
import logging

logger = logging.getLogger(__name__)

def train_roi_classifier(features_path: Path, output_path: Path, test_size: float = 0.2):
    """Train ROI classifier on labeled features."""
    
    # Load features
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} labeled ROIs")
    
    # Prepare data
    X = df[['derivative_skew', 'spike_prom_mean']].values
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    model_dict = {
        'pipeline': pipeline,
        'feature_names': ['derivative_skew', 'spike_prom_mean'],
        'performance': {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    }
    
    dump(model_dict, output_path)
    logger.info(f"Saved model to {output_path}")
    
    return pipeline

def evaluate_roi_model(model_path: Path, test_features_path: Path):
    """Evaluate trained model on new data."""
    from joblib import load
    
    # Load model
    model_dict = load(model_path)
    pipeline = model_dict['pipeline']
    
    # Load test features
    df = pd.read_csv(test_features_path)
    X_test = df[['derivative_skew', 'spike_prom_mean']].values
    y_test = df['label'].values
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Report
    print("\nTest Set Evaluation:")
    print(classification_report(y_test, y_pred))
    
    return y_pred

def main():
    parser = argparse.ArgumentParser(description='Train ROI Classifier')
    parser.add_argument('--features', type=Path, required=True, help='Path to features CSV')
    parser.add_argument('--output', type=Path, default=Path('roi_classifier/models/roi_classifier.pkl'))
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    train_roi_classifier(args.features, args.output, args.test_size)

if __name__ == '__main__':
    main()