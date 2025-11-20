"""
Train spike classifier on only the top 3 most important features.
Notes:
- Expects features extracted from MinMax normalized traces [0,1].
- If a per-video scaled file exists (spike_training_data_scaled.csv), it will be used.
    This ensures training matches inference when features are min-maxed per video.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Top 3 features identified from initial training
TOP_FEATURES = ['skew_contribution', 'spike_prob_value', 'max_second_derivative_raw']

if __name__ == '__main__':
    ROOT_DIR = Path(__file__).parent.parent  # Go up to main project directory

    default_path = ROOT_DIR / 'training_data' / 'spike_filtering' / 'spike_training_data.csv'
    scaled_path = ROOT_DIR / 'training_data' / 'spike_filtering' / 'spike_training_data_scaled.csv'
    features_path = scaled_path if scaled_path.exists() else default_path
    output_path = ROOT_DIR / 'spike_classifier' / 'models' / 'spike_classifier.pkl'
    
    # Load data
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} labeled spikes")
    
    # Use only top 3 features
    X = df[TOP_FEATURES].values
    y = df['label'].values
    logger.info(f"Training with top 3 features: {TOP_FEATURES}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create classifier (no scaling - features already normalized)
    classifier = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    
    # Train
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    
    print("\nClassification Report (Top 3 Features):")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importances
    coefficients = np.abs(classifier.coef_[0])
    feature_importance = pd.DataFrame({
        'feature': TOP_FEATURES,
        'importance': coefficients
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importances (Top 3 Model):")
    print(feature_importance.to_string(index=False))
    
    # Save model
    model_dict = {
        'classifier': classifier,
        'feature_names': TOP_FEATURES,
        'feature_importances': feature_importance.to_dict('records'),
        'expects_per_video_minmax': scaled_path.exists(),
        'scaling': {
            'type': 'per_video_minmax' if scaled_path.exists() else 'none',
            'feature_names': TOP_FEATURES
        },
        'performance': {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    }
    
    dump(model_dict, output_path)
    logger.info(f"Saved model to {output_path}")
    
    print(f"\n✓ Successfully trained spike classifier with top 3 features")
    print(f"✓ Model saved to {output_path}")
