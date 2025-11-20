import pandas as pd

labels = pd.read_csv('training_data/roi__filtering/roi_labels.csv')
features = pd.read_csv('training_data/roi__filtering/roi_features.csv')

# Merge properly
merged = pd.concat([labels, features[['derivative_skew', 'spike_prom_mean']]], axis=1)

# Find ROI 919
roi919 = merged[(merged['source_file'].str.contains('1-1_1x')) & (merged['roi_index'] == 919)]
print("ROI 919 from training data:")
print(roi919[['roi_index', 'derivative_skew', 'spike_prom_mean', 'label']])

# Show a few more good ROIs
print("\nAll good ROIs from this video:")
good_rois = merged[(merged['source_file'].str.contains('1-1_1x')) & (merged['label'] == 1)]
print(good_rois[['roi_index', 'derivative_skew', 'spike_prom_mean', 'label']])
