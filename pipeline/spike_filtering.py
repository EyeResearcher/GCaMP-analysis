"""Spike feature extraction and filtering with top 3 features."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, TYPE_CHECKING
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import skew
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
    
    Top 3 Features (97% accuracy):
    1. skew_contribution: Change in prominence distribution skewness when spike removed
    2. spike_prob_value: Cascade probability at spike
    3. max_second_derivative_raw: Maximum 2nd derivative in pre-spike window (raw F)
    
    Parameters:
        spikes: List of Spike objects
        f_trace: Fluorescence trace
        cascade_prob: Cascade probability trace
        window_size: Window around spike for features
        
    Returns:
        Feature array (n_spikes x 3)
    """
    if len(spikes) == 0:
        return np.array([])
    
    features = np.zeros((len(spikes), 3))
    
    # Compute all prominences for skewness calculation
    # Suppress PeakPropertyWarning about zero prominences
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*prominence.*')
        spike_indices = [s.cascade_peak_idx for s in spikes]
        all_prominences = peak_prominences(cascade_prob, spike_indices)[0]
    neuron_prom_skew = skew(all_prominences) if len(all_prominences) > 1 else 0.0
    
    for i, spike in enumerate(spikes):
        idx = spike.frame_index
        
        # Feature 1: skew_contribution (as proportion of total skew)
        # Proportion of total skewness contributed by this spike (0-1)
        if len(all_prominences) == 1:
            # Only one spike - it contributes 100% of the skew
            features[i, 0] = 1.0
        elif len(all_prominences) == 2:
            # Two spikes - each contributes roughly 50%
            features[i, 0] = 0.5
        else:
            # Multiple spikes - calculate actual contribution
            proms_without = np.delete(all_prominences, i)
            new_skew = skew(proms_without)
            skew_change = abs(neuron_prom_skew - new_skew)
            # Normalize to [0, 1] by dividing by absolute skew
            # Add small epsilon to avoid division by zero
            total_skew = abs(neuron_prom_skew) + 1e-10
            features[i, 0] = min(skew_change / total_skew, 1.0)
        
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

def filter_spikes(features: np.ndarray, classifier_model) -> np.ndarray:
    """
    Filter spikes using trained top-3-feature classifier.
    
    Parameters:
        features: Spike feature array (n_spikes x 3)
        classifier_model: Trained classifier (expects 3 features)
        
    Returns:
        Boolean mask for valid spikes
    """
    if classifier_model is None:
        logger.debug("No spike classifier, keeping all spikes")
        return np.ones(len(features), dtype=bool)
    
    if len(features) == 0:
        return np.array([], dtype=bool)
    
    try:
        # Extract pipeline from model dict
        if isinstance(classifier_model, dict):
            pipeline = classifier_model['pipeline']
        else:
            pipeline = classifier_model
        
        # Predict
        predictions = pipeline.predict(features)
        
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