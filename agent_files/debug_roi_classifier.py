"""
Debug script to check ROI classifier behavior.
"""
from pathlib import Path
from joblib import load
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).parent

# Load the model
model_path = ROOT_DIR / 'roi_classifier' / 'models' / 'roi_classifier.pkl'
model_dict = load(model_path)

print("=== ROI Classifier Debug Info ===\n")
print(f"Model type: {type(model_dict)}")
print(f"Keys: {model_dict.keys()}")

pipeline = model_dict['pipeline']
print(f"\nPipeline steps: {pipeline.named_steps.keys()}")

# Check the scaler
scaler = pipeline.named_steps['scaler']
print(f"\nScaler type: {type(scaler)}")
print(f"Scaler data_min_: {scaler.data_min_}")
print(f"Scaler data_max_: {scaler.data_max_}")
print(f"Scaler data_range_: {scaler.data_range_}")

# Check the classifier
classifier = pipeline.named_steps['classifier']
print(f"\nClassifier type: {type(classifier)}")
print(f"Classifier coefficients: {classifier.coef_}")
print(f"Classifier intercept: {classifier.intercept_}")

# Load some training data to see typical values
training_path = ROOT_DIR / 'training_data' / 'roi__filtering' / 'roi_features.csv'
train_df = pd.read_csv(training_path)

print(f"\n=== Training Data Statistics ===")
print(f"Total samples: {len(train_df)}")
print(f"\nClass distribution:")
print(train_df['label'].value_counts())

print(f"\n=== Feature Statistics (Training) ===")
for col in ['derivative_skew', 'spike_prom_mean']:
    print(f"\n{col}:")
    print(f"  Min: {train_df[col].min():.6f}")
    print(f"  Max: {train_df[col].max():.6f}")
    print(f"  Mean: {train_df[col].mean():.6f}")
    print(f"  Std: {train_df[col].std():.6f}")
    
    # By class
    for label in [0, 1]:
        subset = train_df[train_df['label'] == label]
        print(f"  Class {label} - Mean: {subset[col].mean():.6f}, Std: {subset[col].std():.6f}")

# Test predictions on training data
print(f"\n=== Test on Training Data ===")
X_train = train_df[['derivative_skew', 'spike_prom_mean']].values
y_train = train_df['label'].values
y_pred = pipeline.predict(X_train)

print(f"Predicted class 1 (good): {np.sum(y_pred == 1)} / {len(y_pred)}")
print(f"Predicted class 0 (bad): {np.sum(y_pred == 0)} / {len(y_pred)}")
print(f"Accuracy on training data: {np.mean(y_pred == y_train):.3f}")

# Show some example predictions
print(f"\n=== Example Predictions ===")
for i in range(min(10, len(train_df))):
    features = X_train[i]
    pred = y_pred[i]
    true = y_train[i]
    print(f"Sample {i}: deriv_skew={features[0]:.4f}, spike_prom={features[1]:.4f} -> Pred={pred}, True={true}")
