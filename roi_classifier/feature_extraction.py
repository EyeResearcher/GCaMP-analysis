"""Convert roi_labels.csv to include extracted features."""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew
from scipy.signal import find_peaks, peak_prominences
import logging

logger = logging.getLogger(__name__)

def extract_roi_features(f_trace: np.ndarray, spike_prob: np.ndarray) -> dict:
    """
    Extract the 2 ROI features we use for classification.
    
    Features:
    1. derivative_skew - Skewness of derivative of smoothed F trace (sigma=4.0)
    2. spike_prom_mean - Mean of spike prominences
    """
    # Smooth fluorescence for derivative
    f_smooth = gaussian_filter1d(f_trace, sigma=4.0)
    
    # Derivative skew
    derivative = np.diff(f_smooth)
    derivative_skew = skew(derivative) if len(derivative) > 0 else 0.0
    
    # Find peaks in spike probability
    peaks, properties = find_peaks(spike_prob, prominence=0.05, distance=10)
    
    # Mean prominence
    if len(peaks) > 0:
        prominences = properties['prominences']
        spike_prom_mean = np.mean(prominences)
    else:
        spike_prom_mean = 0.0
    
    return {
        'derivative_skew': derivative_skew,
        'spike_prom_mean': spike_prom_mean
    }

def prepare_roi_training_data(
    labels_path: Path,
    output_path: Path = None
) -> pd.DataFrame:
    """
    Convert roi_labels.csv with paths to roi_features.csv with extracted features.
    
    Args:
        labels_path: Path to roi_labels.csv with columns [source_file, index, label]
        output_path: Optional path to save the feature file
        
    Returns:
        DataFrame with columns [derivative_skew, spike_prom_mean, label]
    """
    # Load labels
    labels_df = pd.read_csv(labels_path)
    logger.info(f"Loaded {len(labels_df)} ROI labels")
    
    # Extract features for each ROI
    features = []
    for _, row in labels_df.iterrows():
        try:
            # Load the Suite2p data
            source_path = Path(row['source_file'])
            roi_idx = int(row['roi_index'])
            
            # Load F and cascade_spike_prob from the same directory
            f_data = np.load(source_path)
            spike_prob_path = source_path.parent / 'cascade_spike_prob.npy'
            
            if not spike_prob_path.exists():
                logger.warning(f"Spike prob not found for {source_path}, skipping")
                continue
                
            spike_prob_data = np.load(spike_prob_path)
            
            # Get the specific ROI traces
            f_trace = f_data[roi_idx]
            spike_prob = spike_prob_data[roi_idx]
            
            # Extract features
            roi_features = extract_roi_features(f_trace, spike_prob)
            roi_features['label'] = row['label']
            features.append(roi_features)
            
        except Exception as e:
            logger.error(f"Error processing ROI {row['roi_index']} from {row['source_file']}: {e}")
            continue
    
    # Create DataFrame
    features_df = pd.DataFrame(features)
    logger.info(f"Extracted features for {len(features_df)} ROIs")
    
    # Save if output path provided
    if output_path:
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")
    
    return features_df