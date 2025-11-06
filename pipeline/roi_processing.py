"""ROI processing - extract features and filter."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from data_classes import ROI

logger = logging.getLogger(__name__)

def extract_roi_features(rois: List[ROI]) -> pd.DataFrame:
    """
    Extract 2 features for each ROI: derivative_skew and spike_prom_mean.
    
    IMPORTANT: Must match training feature extraction exactly!
    - derivative_skew: uses SMOOTHED F trace (sigma=4.0)
    - spike_prom_mean: uses prominence=0.05, distance=10
    
    Parameters:
        rois: List of ROI objects
        
    Returns:
        DataFrame with features
    """
    from scipy.ndimage import gaussian_filter1d
    
    features_list = []
    
    for roi in rois:
        # Calculate derivative skew on SMOOTHED trace (like training!)
        trace = roi.f_trace
        f_smooth = gaussian_filter1d(trace, sigma=4.0)
        derivative = np.diff(f_smooth)
        derivative_skew = stats.skew(derivative)
        
        # Calculate mean spike prominence (match training parameters!)
        peaks, properties = signal.find_peaks(roi.cascade_prob, 
                                             prominence=0.05,
                                             distance=10)
        if len(peaks) > 0:
            spike_prom_mean = np.mean(properties['prominences'])
        else:
            spike_prom_mean = 0.0
        
        # Store in ROI object
        roi.features = {
            'derivative_skew': derivative_skew,
            'spike_prom_mean': spike_prom_mean
        }
        
        features_list.append({
            'roi_index': roi.index,
            'derivative_skew': derivative_skew,
            'spike_prom_mean': spike_prom_mean
        })
    
    return pd.DataFrame(features_list)

def filter_rois(features_df: pd.DataFrame, classifier_model) -> np.ndarray:
    """
    Filter ROIs using trained classifier.
    
    Parameters:
        features_df: DataFrame with derivative_skew and spike_prom_mean
        classifier_model: Trained sklearn pipeline or model dict
        
    Returns:
        Boolean mask (True for good ROIs)
    """
    # Extract feature matrix
    X = features_df[['derivative_skew', 'spike_prom_mean']].values
    
    # Handle different model formats
    if isinstance(classifier_model, dict):
        # Model dict from joblib
        pipeline = classifier_model['pipeline']
        predictions = pipeline.predict(X)
    else:
        # Direct pipeline
        predictions = classifier_model.predict(X)
    
    # Return boolean mask (1 = good, 0 = bad)
    return predictions == 1

def scale_roi_features(features_df: pd.DataFrame) -> np.ndarray:
    """
    Scale features using MinMaxScaler.
    
    Parameters:
        features_df: DataFrame with features
        
    Returns:
        Scaled feature array
    """
    scaler = MinMaxScaler()
    feature_cols = ['derivative_skew', 'spike_prom_mean']
    scaled = scaler.fit_transform(features_df[feature_cols])
    return scaled