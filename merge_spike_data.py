"""
Merge spike annotations with spike features for training.
"""
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    ROOT_DIR = Path(__file__).parent
    
    features_path = ROOT_DIR / 'training_data' / 'spike_filtering' / 'spike_features.csv'
    annotations_path = ROOT_DIR / 'training_data' / 'spike_filtering' / 'spike_annotations.csv'
    output_path = ROOT_DIR / 'training_data' / 'spike_filtering' / 'spike_training_data.csv'
    
    # Load both files
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(features_df)} spike features")
    
    logger.info(f"Loading annotations from {annotations_path}")
    annotations_df = pd.read_csv(annotations_path)
    logger.info(f"Loaded {len(annotations_df)} spike annotations")
    
    # Merge on spike_key
    merged_df = features_df.merge(annotations_df, on='spike_key', how='inner')
    logger.info(f"Merged dataset has {len(merged_df)} labeled spikes")
    
    # Save
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved training data to {output_path}")
    
    print(f"\n✓ Successfully created spike training dataset")
    print(f"✓ Total samples: {len(merged_df)}")
    print(f"✓ Feature columns: {len(merged_df.columns) - 2} (excluding spike_key and label)")
    print(f"\n✓ Class distribution:")
    print(merged_df['label'].value_counts())
    print(f"\n✓ Sample feature columns:")
    print(list(merged_df.columns[:10]))
