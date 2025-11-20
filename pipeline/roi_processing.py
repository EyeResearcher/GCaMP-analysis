"""ROI processing - extract features and filter."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import List, Dict, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from data_classes import ROI

logger = logging.getLogger(__name__)

def normalize_minmax(trace: np.ndarray) -> np.ndarray:
    """Normalize trace to [0, 1] using min-max scaling (NaN-safe)."""
    t = np.asarray(trace, dtype=float).squeeze()
    # Replace NaNs/Infs to avoid contaminating min/max
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    trace_min = float(np.min(t))
    trace_max = float(np.max(t))
    if trace_max - trace_min < 1e-10:
        return np.zeros_like(t)
    return (t - trace_min) / (trace_max - trace_min)

def normalize_deltaf_f(trace: np.ndarray) -> np.ndarray:
    """Compute deltaF/F: (F_i - F_{i-1}) / F_i (NaN-safe)."""
    t = np.asarray(trace, dtype=float).squeeze()
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    trace_safe = np.where(np.abs(t) < 1e-10, 1e-10, t)
    deltaf = np.diff(t)
    deltaf_f = deltaf / trace_safe[1:]
    deltaf_f = np.concatenate([[0.0], deltaf_f])
    return deltaf_f

def extract_roi_features(rois: List[ROI], normalization: str = 'minmax') -> pd.DataFrame:
    """
    Extract 2 features for each ROI with per-video normalization.
    
    Features are computed on normalized fluorescence/probability data to control
    for video brightness differences.
    
    IMPORTANT: Must match training feature extraction exactly!
    - Normalize F and cascade_prob BEFORE computing features
    - derivative_skew: skew of derivative of SMOOTHED (sigma=4.0) normalized F trace
    - spike_prom_mean: mean prominence of peaks found on SMOOTHED (sigma=4.0) normalized cascade_prob (no explicit thresholds)
    
    Parameters:
        rois: List of ROI objects
        normalization: 'minmax' or 'deltaf' normalization strategy
        
    Returns:
        DataFrame with features
    """
    from scipy.ndimage import gaussian_filter1d
    
    features_list = []
    
    for roi in rois:
        # Normalize fluorescence BEFORE computing features
        if normalization == 'minmax':
            f_normalized = normalize_minmax(roi.f_trace)
        elif normalization == 'deltaf':
            f_normalized = normalize_deltaf_f(roi.f_trace)
        else:
            f_normalized = roi.f_trace

        # Normalize cascade probability to [0, 1] (sanitize NaNs first)
        sp = np.asarray(roi.cascade_prob, dtype=float).squeeze()
        sp = np.nan_to_num(sp, nan=0.0, posinf=1.0, neginf=0.0)
        cascade_normalized = normalize_minmax(sp)

        # Calculate derivative skew on SMOOTHED normalized trace
        f_smooth = gaussian_filter1d(f_normalized, sigma=4.0)
        derivative = np.diff(f_smooth)
        derivative_skew = stats.skew(derivative) if len(derivative) > 0 else 0.0

        # Calculate mean spike prominence on SMOOTHED normalized probability
        sp_smooth = gaussian_filter1d(cascade_normalized, sigma=4.0)
        peaks, _ = signal.find_peaks(sp_smooth)
        if len(peaks) > 0:
            prominences = signal.peak_prominences(sp_smooth, peaks)[0]
            spike_prom_mean = float(np.mean(prominences)) if len(prominences) > 0 else 0.0
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
        classifier_model: Trained sklearn classifier or model dict
        
    Returns:
        Boolean mask (True for good ROIs)
    """
    # Extract feature matrix
    X = features_df[['derivative_skew', 'spike_prom_mean']].values
    
    # Handle different model formats
    if isinstance(classifier_model, dict):
        # Model dict from joblib - support both old (pipeline) and new (classifier) format
        classifier = classifier_model.get('classifier') or classifier_model.get('pipeline')
        predictions = classifier.predict(X)
    else:
        # Direct classifier
        predictions = classifier_model.predict(X)
    
    # Return boolean mask (1 = good, 0 = bad)
    return predictions == 1