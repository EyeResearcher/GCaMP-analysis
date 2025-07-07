import sys


import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import skew

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
    feats : dict with keys
       "derivative_skew",
       "spike_prom_skew",
       "spike_peak_mean",
       "spike_prom_mean"
    """
    # 1) smooth the fluorescence and compute derivative skew
    fl_smooth = gaussian_filter1d(raw_trace, sigma=sigma_fluo)
    deriv = np.diff(fl_smooth)
    derivative_skew = float(skew(deriv)) if deriv.size > 1 else 0.0

    # 2) smooth the spike‐probability trace & find peaks
    sp_smooth = gaussian_filter1d(spike_prob_trace, sigma=sigma_spike)
    peaks, _ = find_peaks(sp_smooth, prominence=prominence, distance=distance)

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

    return {
        "derivative_skew":   derivative_skew,
        "spike_prom_skew":   spike_prom_skew,
       "spike_peak_mean":   spike_peak_mean,
        "spike_prom_mean":   spike_prom_mean
    }