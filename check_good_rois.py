import pandas as pd

train_df = pd.read_csv('training_data/roi__filtering/roi_labels.csv')
video_rois = train_df[train_df['source_file'].str.contains('1-1_1x')]

print(f'Total ROIs from this video: {len(video_rois)}')
print(f'Good (1): {(video_rois["label"] == 1).sum()}')
print(f'Bad (0): {(video_rois["label"] == 0).sum()}')
print('\nFirst 20 Good ROI indices:')
good_indices = video_rois[video_rois['label'] == 1]['roi_index'].tolist()
print(good_indices[:20])

# Load features for these good ROIs
features_df = pd.read_csv('training_data/roi__filtering/roi_features.csv')
train_with_features = train_df.merge(features_df, left_index=True, right_index=True)
good_rois_features = train_with_features[(train_with_features['source_file'].str.contains('1-1_1x')) & (train_with_features['label_x'] == 1)]

print(f'\nGood ROI feature statistics:')
print(good_rois_features[['derivative_skew', 'spike_prom_mean']].describe())
