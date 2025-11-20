"""
Compare feature distributions with sigma=2.0 vs what we had before.
"""
import pandas as pd
import numpy as np

# Load the new features (sigma=2.0)
features_df = pd.read_csv('training_data/roi__filtering/roi_features.csv')

print("=== Feature Statistics with sigma=2.0 ===\n")

for label in [0, 1]:
    label_name = "Bad (0)" if label == 0 else "Good (1)"
    subset = features_df[features_df['label'] == label]
    
    print(f"{label_name} - Count: {len(subset)}")
    print(f"  derivative_skew: mean={subset['derivative_skew'].mean():.4f}, std={subset['derivative_skew'].std():.4f}")
    print(f"  spike_prom_mean: mean={subset['spike_prom_mean'].mean():.4f}, std={subset['spike_prom_mean'].std():.4f}")
    print()

# Calculate separation between classes
good_df = features_df[features_df['label'] == 1]
bad_df = features_df[features_df['label'] == 0]

deriv_separation = (good_df['derivative_skew'].mean() - bad_df['derivative_skew'].mean()) / np.sqrt(
    (good_df['derivative_skew'].std()**2 + bad_df['derivative_skew'].std()**2) / 2
)

prom_separation = (good_df['spike_prom_mean'].mean() - bad_df['spike_prom_mean'].mean()) / np.sqrt(
    (good_df['spike_prom_mean'].std()**2 + bad_df['spike_prom_mean'].std()**2) / 2
)

print("=== Class Separation (Cohen's d) ===")
print(f"derivative_skew: {deriv_separation:.4f}")
print(f"spike_prom_mean: {prom_separation:.4f}")
print("\nHigher values = better separation. Values > 0.8 are considered large effects.")

# Check overlap
print("\n=== Feature Range Overlap ===")
print(f"derivative_skew:")
print(f"  Bad range: [{bad_df['derivative_skew'].min():.4f}, {bad_df['derivative_skew'].max():.4f}]")
print(f"  Good range: [{good_df['derivative_skew'].min():.4f}, {good_df['derivative_skew'].max():.4f}]")

print(f"\nspike_prom_mean:")
print(f"  Bad range: [{bad_df['spike_prom_mean'].min():.4f}, {bad_df['spike_prom_mean'].max():.4f}]")
print(f"  Good range: [{good_df['spike_prom_mean'].min():.4f}, {good_df['spike_prom_mean'].max():.4f}]")
