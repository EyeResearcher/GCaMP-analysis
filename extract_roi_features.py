"""
Extract ROI features from training data labels.
"""
from pathlib import Path
import logging
import sys

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from roi_classifier.feature_extraction import prepare_roi_training_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    labels_path = ROOT_DIR / 'training_data' / 'roi__filtering' / 'roi_labels.csv'
    output_path = ROOT_DIR / 'training_data' / 'roi__filtering' / 'roi_features.csv'
    
    print(f"Extracting ROI features from {labels_path}")
    print(f"Output will be saved to {output_path}")
    
    features_df = prepare_roi_training_data(labels_path, output_path)
    
    print(f"\n✓ Successfully extracted features for {len(features_df)} ROIs")
    print(f"✓ Feature columns: {list(features_df.columns)}")
    print(f"\n✓ Class distribution:")
    print(features_df['label'].value_counts())
