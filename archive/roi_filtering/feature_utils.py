import sys


import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import skew, zscore, kurtosis

def zscore_features(features):
    """
    Z-score the features for each spike.
    """
    if not isinstance(features, np.ndarray) and not isinstance(features, list):
        raise ValueError("Features must be a numpy array or a list of arrays.")
    features = np.array(features) if isinstance(features, list) else features
    return zscore(features, axis=0) if features.ndim > 1 else zscore(features)

def four_primary_roi_features(
    raw_trace: np.ndarray,
    spike_prob_trace: np.ndarray,
    sigma_fluo: float     = 4.0,
    sigma_spike: float    = 2.0,
    prominence: float     = 0.05,
    distance: int         = 10
) -> dict:
    """
    Compute 4 primary features:
      1) skewness of first‐derivative of the smoothed fluorescence trace
      2) skewness of the left-base prominences of the smoothed spike‐probability trace
      3) mean peak height of that smoothed spike‐probability trace
      4) mean left‐base prominence of that smoothed spike‐probability trace

    Parameters
    ----------
    raw_trace : 1D array of fluorescence values
    spike_prob_trace : 1D array of deconvolved / spike‐probability values
    sigma_fluo : gaussian σ for smoothing raw_trace
    sigma_spike: gaussian σ for smoothing spike_prob_trace
    prominence : min prominence for peak finding on spike_prob_trace
    distance   : min inter‐peak distance

    Returns
    -------
    feats : list of 4 features as described above
    """
    # 1) smooth the fluorescence and compute derivative skew
    fl_smooth = gaussian_filter1d(raw_trace, sigma=sigma_fluo)
    deriv = np.diff(fl_smooth)
    derivative_skew = float(skew(deriv)) if deriv.size > 1 else 0.0

    # 2) smooth the spike‐probability trace & find peaks
    sp_smooth = gaussian_filter1d(spike_prob_trace, sigma=sigma_spike)
    peaks, _ = find_peaks(sp_smooth)

    if peaks.size:
        # get full prominences + base indices
        proms, left_bases, right_bases = peak_prominences(sp_smooth, peaks)

        # compute left‐base prominence = peak_value ‐ signal[left_base]
        peak_vals    = sp_smooth[peaks]
        left_vals    = sp_smooth[left_bases]
        left_proms   = peak_vals - left_vals

        spike_prom_skew  = float(skew(left_proms)) if left_proms.size > 1 else 0.0
        spike_peak_mean  = float(np.mean(peak_vals))
        spike_prom_mean  = float(np.mean(left_proms))
    else:
        spike_prom_skew  = 0.0
        spike_peak_mean  = 0.0
        spike_prom_mean  = 0.0

    return [derivative_skew, spike_prom_skew, spike_peak_mean, spike_prom_mean ]

def get_windowed_trace(raw_trace, trace, i_peak, edge = 32):
    """
    Get a windowed trace around a peak and its previous local minimum.
    Args:
        trace (np.ndarray): The trace from which to extract the window.
        i_peak (int): The index of the peak in the trace.
    Returns:
        window (np.ndarray): The windowed trace.
    """
    start = max(find_local_minimum(trace, i_peak, left = True), edge)
    end = min(find_local_minimum(trace, i_peak, right=True), len(trace) - edge)
    left_window = trace[start:i_peak]
    window = trace[start:end]
    right_window = trace[i_peak:end]
    left_window_raw, window_raw, right_window_raw = raw_trace[start:i_peak], raw_trace[start:end], raw_trace[i_peak:end]
    return (left_window, window, right_window), (left_window_raw, window_raw, right_window_raw)
    
def find_local_minimum(trace, i_peak, left = False, right = False):
    """ Find the previous local minimum in a trace before a given peak index.
    Args:
        trace (np.ndarray): The trace in which to find the local minimum.
        i_peak (int): The index of the peak in the trace.
    Returns:
        j (int): The index of the previous local minimum, or None if not found."""
    start, end, step = (i_peak - 1, 0, -1) if left else (i_peak + 1, len(trace), 1)
    for j in range(start, end, step):
        if trace[j] < trace[j-1] and trace[j] < trace[j+1]:
            return j
    return 0  # no local minimum found, default to start of trace

def find_max_second_derivative(window):
    d1 = np.diff(window) if len(window) > 1 else np.array([0])
    d2 = np.diff(d1) if len(d1) > 1 else np.array([0])
   
    max_second_derivative = np.max(d2)
    return max_second_derivative



def compute_spike_features(i, raw_trace, spike_prob_trace, all_left_base_proms, spike_idx_prob, neuron_prom_skew, edge = 32):
    """
    Compute features for a single spike:
      1. Left-base prominence at spike index in spike_prob_trace.
      2. Value in spike_prob_trace at spike index.
      3. Change in skewness of prominence distribution when this spike is removed.
    """
    

    
    # Find the index of this spike in the peaks array
    """peak_idx = np.where(peaks == spike_idx_prob)[0][0]
    try:
        left_base_prom = spike_prob_trace[peaks[peak_idx]] - spike_prob_trace[left_bases[peak_idx]]
    except IndexError:
        raise IndexError(Possible reasons inclued: 
                         {peak_idx} is out of bounds for peaks array of length {len(peaks)}. 
                         {peak_idx} is out of bounds for left_bases array of length {len(left_bases)}.
                         {peaks} or {left_bases} contain unexpected values.) """
    #Step 0: Find window around the spike
    left_window, window, right_window = get_windowed_trace(raw_trace, spike_prob_trace, spike_idx_prob)[0]
    left_window_raw, window_raw, right_window_raw = get_windowed_trace(raw_trace, spike_prob_trace, spike_idx_prob)[1]

    #Step 1: Find derivative and integral features
    max_d2 = find_max_second_derivative(left_window)
    max_d2_raw = find_max_second_derivative(left_window_raw)
    auc = np.trapz(window)

    #Step 2: Value in spike_prob_trace at spike index
    spike_prob_value = spike_prob_trace[spike_idx_prob]

    #Step 3: Change in skewness of prominence distribution if this spike is removed
    if all_left_base_proms.size > 1:
        proms_wo = np.delete(all_left_base_proms, i)
        new_skew = skew(proms_wo) if proms_wo.size > 1 else 0.0
        delta_skew = neuron_prom_skew - new_skew
        
    else:
        delta_skew = 0.0
    
    #Step 4: Compute window features
    (window_skew, window_kurtosis) = (skew(window), kurtosis(window)) if (len(window) > 4 and np.nanvar(window) > 0) else (0.0, 0.0)
    (window_raw_skew, window_raw_kurtosis) = (skew(window_raw), kurtosis(window_raw)) if (len(window_raw) > 4 and np.nanvar(window_raw) > 0) else (0.0, 0.0)

    return [all_left_base_proms[i], spike_prob_value, delta_skew, auc, max_d2, max_d2_raw, window_kurtosis, window_skew, window_raw_kurtosis, window_raw_skew]