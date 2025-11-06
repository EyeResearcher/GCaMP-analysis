"""
Test ROI classifier on a real video to see what features it extracts.
"""
from pathlib import Path
import numpy as np
from joblib import load
import pandas as pd
from scipy import signal, stats

ROOT_DIR = Path(__file__).parent

# Load model
model_path = ROOT_DIR / 'roi_classifier' / 'models' / 'roi_classifier.pkl'
model_dict = load(model_path)
pipeline = model_dict['pipeline']

# Load a real video
video_path = Path(r"C:\Users\mzinn1\Desktop\Datasets\roi-classifier-training_01\1-1_1x\suite2p\plane0")
F = np.load(video_path / 'F.npy')
cascade_prob = np.load(video_path / 'cascade_spike_prob.npy')

print(f"=== Video Info ===")
print(f"Path: {video_path}")
print(f"Number of ROIs: {F.shape[0]}")
print(f"Number of frames: {F.shape[1]}")

# Extract features for first 20 ROIs
n_test = min(20, F.shape[0])
features_list = []

for i in range(n_test):
    # Calculate derivative skew
    trace = F[i]
    derivative = np.diff(trace)
    derivative_skew = stats.skew(derivative)
    
    # Calculate mean spike prominence
    peaks, properties = signal.find_peaks(cascade_prob[i], 
                                         prominence=0.01,
                                         distance=8)
    if len(peaks) > 0:
        spike_prom_mean = np.mean(properties['prominences'])
    else:
        spike_prom_mean = 0.0
    
    features_list.append({
        'roi_index': i,
        'derivative_skew': derivative_skew,
        'spike_prom_mean': spike_prom_mean
    })

features_df = pd.DataFrame(features_list)

print(f"\n=== Extracted Features ===")
print(features_df)

print(f"\n=== Feature Statistics ===")
print(features_df[['derivative_skew', 'spike_prom_mean']].describe())

# Make predictions
X = features_df[['derivative_skew', 'spike_prom_mean']].values
predictions = pipeline.predict(X)
probabilities = pipeline.predict_proba(X)

print(f"\n=== Predictions ===")
print(f"Good ROIs (class 1): {np.sum(predictions == 1)} / {n_test}")
print(f"Bad ROIs (class 0): {np.sum(predictions == 0)} / {n_test}")

print(f"\n=== Detailed Predictions ===")
for i in range(n_test):
    print(f"ROI {i}: deriv_skew={features_df.iloc[i]['derivative_skew']:.4f}, "
          f"spike_prom={features_df.iloc[i]['spike_prom_mean']:.4f} -> "
          f"Pred={predictions[i]}, Prob(bad)={probabilities[i][0]:.3f}, Prob(good)={probabilities[i][1]:.3f}")

# Compare to training data ranges
print(f"\n=== Comparison to Training Data ===")
training_path = ROOT_DIR / 'training_data' / 'roi__filtering' / 'roi_features.csv'
train_df = pd.read_csv(training_path)

for col in ['derivative_skew', 'spike_prom_mean']:
    test_mean = features_df[col].mean()
    train_good_mean = train_df[train_df['label'] == 1][col].mean()
    train_bad_mean = train_df[train_df['label'] == 0][col].mean()
    
    print(f"\n{col}:")
    print(f"  Test video mean: {test_mean:.4f}")
    print(f"  Training good ROIs mean: {train_good_mean:.4f}")
    print(f"  Training bad ROIs mean: {train_bad_mean:.4f}")
