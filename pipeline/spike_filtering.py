"""Spike feature extraction and filtering with top 3 features."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, TYPE_CHECKING, Tuple
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import skew
from scipy.ndimage import gaussian_filter1d
import logging
import warnings

if TYPE_CHECKING:
    from data_classes import Spike

logger = logging.getLogger(__name__)

def extract_spike_features(spikes: List[Spike],
                          f_trace: np.ndarray,
                          cascade_prob: np.ndarray,
                          window_size: int = 30) -> np.ndarray:
    """
    Extract top 3 features for each spike for classification.
    
    Top 3 Features:
    1. skew_contribution: Contribution to derivative skewness (aligned with ROI classifier)
    2. spike_prob_value: Cascade probability at spike
    3. max_second_derivative_raw: Maximum 2nd derivative in pre-spike window
    
    Note: Expects f_trace and cascade_prob to be MinMax normalized [0,1] per-video.
    This normalization happens in the pipeline before feature extraction to ensure
    consistency between training and inference.
    
    Parameters:
        spikes: List of Spike objects
        f_trace: MinMax normalized fluorescence trace [0,1]
        cascade_prob: MinMax normalized cascade probability [0,1]
        window_size: Window around spike for features
        
    Returns:
        Feature array (n_spikes x 3)
    """

    if len(spikes) == 0:
        return np.array([])
    
    features = np.zeros((len(spikes), 3))
    
    # Compute derivative skewness (matching ROI classifier approach)
    # Smooth fluorescence and compute first derivative
    f_smooth = gaussian_filter1d(f_trace, sigma=4.0)
    derivative = np.diff(f_smooth)
    neuron_deriv_skew = skew(derivative) if len(derivative) > 0 else 0.0
    
    for i, spike in enumerate(spikes):
        idx = spike.frame_index
        
        # Feature 1: skew_contribution to derivative skewness
        # How much does removing this spike change the derivative skewness?
        # This aligns with ROI classifier which uses derivative_skew as key feature
        if len(spikes) == 1:
            # Only one spike - it contributes 100% of the derivative skew
            features[i, 0] = 1.0
        elif len(spikes) == 2:
            # Two spikes - each contributes roughly 50%
            features[i, 0] = 0.5
        else:
            # Multiple spikes - remove this spike and recalculate derivative skew
            # Simply remove the spike window and concatenate the trace
            spike_window = 5  # Remove +/- 5 frames around spike
            spike_start = max(0, idx - spike_window)
            spike_end = min(len(f_trace), idx + spike_window + 1)
            
            # Concatenate trace without this spike region
            f_without_spike = np.concatenate([
                f_trace[:spike_start],
                f_trace[spike_end:]
            ])
            
            # Recompute derivative skewness without this spike
            if len(f_without_spike) > 10:  # Need minimum length for meaningful skew
                f_smooth_modified = gaussian_filter1d(f_without_spike, sigma=4.0)
                derivative_modified = np.diff(f_smooth_modified)
                new_skew = skew(derivative_modified) if len(derivative_modified) > 0 else 0.0
                
                # Calculate contribution as change in skewness
                skew_change = abs(neuron_deriv_skew - new_skew)
                # Normalize to [0, 1] by dividing by absolute skew
                total_skew = abs(neuron_deriv_skew) + 1e-10
                features[i, 0] = min(skew_change / total_skew, 1.0)
            else:
                # Trace too short after removal, assign moderate contribution
                features[i, 0] = 0.5
        
        # Feature 2: spike_prob_value
        # Cascade probability value at spike
        features[i, 1] = spike.prob_height
        
        # Feature 3: max_second_derivative_raw
        # Maximum 2nd derivative in pre-spike window of raw fluorescence
        pre_window_start = max(0, idx - 10)
        pre_window_end = idx
        if pre_window_end > pre_window_start + 2:
            pre_window = f_trace[pre_window_start:pre_window_end]
            first_deriv = np.diff(pre_window)
            if len(first_deriv) > 1:
                second_deriv = np.diff(first_deriv)
                features[i, 2] = np.max(np.abs(second_deriv)) if len(second_deriv) > 0 else 0.0
            else:
                features[i, 2] = 0.0
        else:
            features[i, 2] = 0.0
    
    # Final safety check: replace any remaining NaN values with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features

def _minmax_scale_array(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise MinMax scale array to [0,1]; returns scaled, mins, maxs."""
    if arr.size == 0:
        return arr, np.array([]), np.array([])
    mins = np.nanmin(arr, axis=0)
    maxs = np.nanmax(arr, axis=0)
    denom = np.where(maxs > mins, (maxs - mins), 1.0)
    scaled = (arr - mins) / denom
    # constant columns -> 0
    scaled = np.where(denom == 1.0, 0.0, scaled)
    return scaled, mins, maxs


def filter_spikes(features: np.ndarray, classifier_model, *, per_video_scale: bool = False) -> np.ndarray:
    """
    Filter spikes using trained classifier.
    
    Parameters:
        features: Spike feature array (n_spikes x 3)
        classifier_model: Trained classifier or model dict
        
    Returns:
        Boolean mask for valid spikes
    """
    if classifier_model is None:
        logger.debug("No spike classifier, keeping all spikes")
        return np.ones(len(features), dtype=bool)
    
    if len(features) == 0:
        return np.array([], dtype=bool)
    
    try:
        # Extract classifier from model dict (support both old pipeline and new classifier)
        if isinstance(classifier_model, dict):
            classifier = classifier_model.get('classifier') or classifier_model.get('pipeline')
            expects_scale = classifier_model.get('expects_per_video_minmax', False)
        else:
            classifier = classifier_model
            expects_scale = False
        
        X = features
        if per_video_scale or expects_scale:
            X, _, _ = _minmax_scale_array(X)
        
        # Predict
        predictions = classifier.predict(X)
        
        # Return boolean mask (1 = valid spike)
        return predictions == 1
        
    except Exception as e:
        logger.error(f"Spike classification failed: {e}")
        logger.warning("Keeping all spikes due to error")
        return np.ones(len(features), dtype=bool)

def compute_spike_metrics(spike: Spike, 
                         f_trace: np.ndarray,
                         window_size: int = 30) -> Dict:
    """
    Compute detailed metrics for a spike.
    
    Parameters:
        spike: Spike object
        f_trace: Fluorescence trace
        window_size: Analysis window
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    idx = spike.frame_index
    
    # Get window
    start = max(0, idx - window_size//2)
    end = min(len(f_trace), idx + window_size//2)
    window = f_trace[start:end]
    
    # Baseline
    baseline_start = max(0, idx - window_size)
    baseline = np.median(f_trace[baseline_start:idx])
    
    # Amplitude metrics
    metrics['amplitude'] = spike.f_value - baseline
    metrics['relative_amplitude'] = metrics['amplitude'] / baseline if baseline > 0 else 0
    
    # Timing metrics
    metrics['rise_time'] = compute_rise_time(f_trace, idx, baseline)
    metrics['decay_time'] = compute_decay_time(f_trace, idx, baseline)
    metrics['fwhm'] = compute_fwhm(window, baseline)
    
    # Area metrics
    metrics['auc'] = np.trapz(window - baseline)
    
    return metrics

def compute_rise_time(trace: np.ndarray, peak_idx: int, baseline: float) -> float:
    """Compute 10-90% rise time."""
    if peak_idx < 2:
        return np.nan
    
    peak_val = trace[peak_idx]
    rise_10 = baseline + 0.1 * (peak_val - baseline)
    rise_90 = baseline + 0.9 * (peak_val - baseline)
    
    # Search backwards for crossing points
    rise_start = peak_idx
    rise_end = peak_idx
    
    for i in range(peak_idx, max(0, peak_idx-20), -1):
        if trace[i] <= rise_90 and rise_end == peak_idx:
            rise_end = i
        if trace[i] <= rise_10:
            rise_start = i
            break
    
    return rise_end - rise_start

def compute_decay_time(trace: np.ndarray, peak_idx: int, baseline: float) -> float:
    """Compute decay to baseline."""
    if peak_idx >= len(trace) - 2:
        return np.nan
    
    peak_val = trace[peak_idx]
    threshold = baseline + 0.1 * (peak_val - baseline)
    
    # Search forward for decay
    for i in range(peak_idx, min(len(trace), peak_idx + 50)):
        if trace[i] <= threshold:
            return i - peak_idx
    
    return np.nan

def compute_fwhm(window: np.ndarray, baseline: float) -> float:
    """Compute full width at half maximum."""
    peak_idx = np.argmax(window)
    peak_val = window[peak_idx]
    half_max = baseline + (peak_val - baseline) / 2
    
    above_half = window > half_max
    if np.any(above_half):
        return np.sum(above_half)
    return 0