"""Apply the trained ROI classifier to label all ROIs in the features file."""
import numpy as np
from pathlib import Path
import joblib

def main():
    # Load the trained classifier
    model_path = Path('training_data/roi_filtering/roi_classifier.joblib')
    if not model_path.exists():
        print(f"❌ Classifier not found at {model_path}")
        print("You need to train a classifier first or manually label ROIs.")
        return
    
    classifier = joblib.load(model_path)
    print(f"✅ Loaded classifier from {model_path}")
    
    # Load ROI features
    features_path = Path('training_data/roi_filtering/all_roi_features.npy')
    roi_dict = np.load(features_path, allow_pickle=True).item()
    print(f"✅ Loaded {len(roi_dict)} ROIs from {features_path}")
    
    # Extract features for prediction
    roi_keys = []
    feature_matrix = []
    for roi_key, roi_data in roi_dict.items():
        features = roi_data['features']
        # Use the same features the classifier was trained on
        feature_vector = [
            features['derivative_skew'],
            features['spike_prom_mean']
        ]
        roi_keys.append(roi_key)
        feature_matrix.append(feature_vector)
    
    feature_matrix = np.array(feature_matrix)
    print(f"✅ Extracted feature matrix with shape {feature_matrix.shape}")
    
    # Predict labels
    predictions = classifier.predict(feature_matrix)
    print(f"✅ Made predictions for {len(predictions)} ROIs")
    
    # Update labels in the dictionary
    good_count = 0
    bad_count = 0
    for roi_key, pred_label in zip(roi_keys, predictions):
        roi_dict[roi_key]['label'] = int(pred_label)
        if pred_label == 1:
            good_count += 1
        else:
            bad_count += 1
    
    # Save updated dictionary
    np.save(features_path, roi_dict)
    print(f"\n✅ Updated and saved ROI labels to {features_path}")
    print(f"   Good ROIs (label=1): {good_count}")
    print(f"   Bad ROIs (label=0): {bad_count}")
    print(f"\nYou can now run: python spike_classifier/prepare_data.py")

if __name__ == '__main__':
    main()
