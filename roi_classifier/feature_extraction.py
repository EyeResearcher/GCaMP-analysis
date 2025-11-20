"""Convert roi_labels.csv to include extracted features."""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew
from scipy.signal import find_peaks, peak_prominences
import logging
from sklearn.preprocessing import MinMaxScaler
logger = logging.getLogger(__name__)



def extract_roi_features(f_trace: np.ndarray, 
                        spike_prob: np.ndarray) -> dict:
    """
    Extract the 2 ROI features we use for classification.
    
    Features are computed on normalized data to control for video brightness.
    
     Features (aligned with inference path):
     1. derivative_skew - Skewness of derivative of smoothed (sigma=4.0) normalized F trace.
     2. spike_prom_mean - Mean prominence of peaks found on a SMOOTHED (sigma=4.0) normalized spike probability trace.
         (No explicit prominence or distance threshold; prominences computed via peak_prominences.)
    
    Args:
        f_trace: Raw fluorescence trace
        spike_prob: CASCADE spike probability trace
        normalization: 'minmax' or 'deltaf' normalization strategy
    
    Returns:
        Dictionary with feature values
    """
    

    
    # Smooth normalized fluorescence for derivative
    f_smooth = gaussian_filter1d(f_normalized, sigma=4.0)
    
    # Derivative skew (now on normalized data)
    derivative = np.diff(f_smooth)
    derivative_skew = skew(derivative) if len(derivative) > 0 else 0.0
    
    # Smooth spike probability (match inference) before peak detection
    prob_smooth = gaussian_filter1d(spike_prob_normalized, sigma=4.0)
    peaks, _ = find_peaks(prob_smooth)  # no explicit thresholds
    if len(peaks) > 0:
        prominences = peak_prominences(prob_smooth, peaks)[0]
        spike_prom_mean = float(np.mean(prominences)) if len(prominences) > 0 else 0.0
    else:
        spike_prom_mean = 0.0
    
    return {
        'derivative_skew': derivative_skew,
        'spike_prom_mean': spike_prom_mean
    }

