import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import skew


def left_based_prominence(spike_prob: np.ndarray) -> tuple:
    """Compute mean and skew of left-based prominences."""
    peaks, _ = find_peaks(spike_prob)
    if len(peaks) == 0:
        return (0.0, 0.0, False)
    proms, left_bases, _ = peak_prominences(spike_prob, peaks)
    peak_vals = spike_prob[peaks]
    left_vals = spike_prob[left_bases]
    left_base_prominences = peak_vals - left_vals
    prom_mean = float(np.mean(left_base_prominences))
    if len(left_base_prominences) > 1 and np.ptp(left_base_prominences) > 0:
        prom_skew = float(skew(left_base_prominences))
        if not np.isfinite(prom_skew):
            prom_skew = 0.0
    else:
        prom_skew = 0.0
    return (prom_mean, prom_skew, True)

def derivative_skewness(smoothed_scaled_f: np.ndarray) -> tuple:
    """Compute skewness of the derivative."""
    derivative = np.diff(smoothed_scaled_f)
    if len(derivative) == 0:
        return (0.0, False, derivative)
    if np.any(np.isnan(derivative)) or np.any(np.isinf(derivative)):
        return (0.0, False, derivative)
    if np.ptp(derivative) == 0:          # constant → skew undefined
        return (0.0, False, derivative)
    s = float(skew(derivative))
    if not np.isfinite(s):
        return (0.0, False, derivative)
    return (s, True, derivative)


def derivative_asymmetry(smoothed_scaled_f: np.ndarray) -> tuple:
    """
    Energy asymmetry between positive and negative derivatives.

    Real ROIs with upward transients tend to have pos_energy > neg_energy.
    Noise-like ROIs tend to have pos_energy ≈ neg_energy → asymmetry ≈ 1.
    """
    d = np.diff(smoothed_scaled_f)
    if d.size == 0:
        return (0.0, False)
    if np.any(~np.isfinite(d)):
        return (0.0, False)

    pos = np.abs(d[d > 0]).sum()
    neg = np.abs(d[d < 0]).sum()
    if pos == 0 and neg == 0:
        return (0.0, False)

    asym = pos / (neg + 1e-9)
    return (float(asym), True)


def rolling_variance_of_variance(x: np.ndarray, window: int = 30) -> tuple:
    """
    Variance of a rolling variance over the trace.

    Real ROIs: baseline + bursts → rolling variance changes over time → higher var_of_var.
    Noise ROIs: more stationary → lower var_of_var.
    """
    x = np.asarray(x, float)
    n = x.size
    if n < window * 2:
        return (0.0, False)

    w = np.ones(window, float) / window
    mean = np.convolve(x, w, mode="valid")
    mean_sq = np.convolve(x * x, w, mode="valid")
    roll_var = np.maximum(mean_sq - mean * mean, 0.0)

    var_of_var = float(np.var(roll_var))
    if not np.isfinite(var_of_var):
        return (0.0, False)
    return (var_of_var, True)


def autocorr_decay(smoothed_scaled_f: np.ndarray, lag1: int = 1, lag2: int = 5) -> tuple:
    """
    Simple autocorrelation decay metric: rho(lag1) - rho(lag2).

    Real ROIs with slow kinetics keep correlation over multiple lags.
    Noise ROIs decorrelate quickly → smaller difference.
    """
    x = np.asarray(smoothed_scaled_f, float)
    n = x.size
    max_lag = max(lag1, lag2)
    if n <= max_lag:
        return (0.0, False)

    x = x - np.nanmean(x)
    if not np.all(np.isfinite(x)):
        return (0.0, False)

    def _rho(L):
        a = x[:-L]
        b = x[L:]
        num = np.dot(a, b)
        den = np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-12)
        if den == 0:
            return 0.0
        return float(num / den)

    rho1 = _rho(lag1)
    rho2 = _rho(lag2)
    ac_decay = rho1 - rho2
    if not np.isfinite(ac_decay):
        return (0.0, False)
    return (ac_decay, True)


def snr_estimate(smoothed_f_trace: np.ndarray) -> tuple:
    """
    Robust SNR estimate using MAD for noise and percentile spread for signal.

    Good ROIs: snr >> 1
    Bad/noise ROIs: snr ≈ 1
    """
    x = np.asarray(smoothed_f_trace, float)
    if x.size == 0:
        return (0.0, False)
    if not np.all(np.isfinite(x)):
        return (0.0, False)

    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    noise_level = 1.4826 * mad  # approx sigma for Gaussian

    p95 = np.nanpercentile(x, 95)
    p20 = np.nanpercentile(x, 20)
    signal_level = p95 - p20

    if noise_level <= 0:
        return (0.0, False)

    snr = signal_level / (noise_level + 1e-9)
    if not np.isfinite(snr):
        return (0.0, False)
    return (float(snr), True)


def peak_density_and_prominence(smoothed_f_trace: np.ndarray) -> tuple:
    """
    Trace-level peak density and median prominence of spike-probability peaks.

    Noise ROIs: lots of tiny peaks → high density, tiny median prominence.
    Good ROIs: fewer, stronger peaks.
    """
    x = np.asarray(smoothed_f_trace, float)
    if x.size == 0:
        return (0.0, 0.0, False)

    dyn_range = float(np.nanmax(x) - np.nanmin(x))
    if not np.isfinite(dyn_range) or dyn_range <= 0:
        return (0.0, 0.0, False)

    prom_thresh = 0.05 * dyn_range
    peaks, _ = find_peaks(x, prominence=prom_thresh)

    if len(peaks) == 0:
        return (0.0, 0.0, False)

    proms, _, _ = peak_prominences(x, peaks)
    peak_density = len(peaks) / float(x.size)
    median_prom = float(np.median(proms))
    return (peak_density, median_prom, True)


# =============================================================================
# Comprehensive ROI Feature Extraction
# =============================================================================

def compute_roi_features(smoothed_f_trace: np.ndarray) -> tuple[dict, dict]:
    """
    Extract comprehensive ROI-level features from traces.

    Parameters
    ----------
    smoothed_f_trace : np.ndarray
        Smoothed, min-max normalized fluorescence trace (1D).

    Returns
    -------
    features : dict
        Scalar features per ROI (one row per ROI for classifier).
    validity : dict
        Flags indicating which feature groups were computed cleanly.
    """
    assert smoothed_f_trace.ndim == 1
    
    # Derivative-based metrics
    deriv_skew, valid_deriv_skew, derivative = derivative_skewness(smoothed_f_trace)
    deriv_asym, valid_deriv_asym = derivative_asymmetry(smoothed_f_trace)
    spike_prom_mean, spike_prom_skew, valid_prom = left_based_prominence(smoothed_f_trace)
    var_of_var, valid_vov = rolling_variance_of_variance(smoothed_f_trace, window=30)

    ac_decay, valid_ac = autocorr_decay(smoothed_f_trace, lag1=1, lag2=5)

    snr, valid_snr = snr_estimate(smoothed_f_trace)

    peak_density, median_spike_prom, valid_peak = peak_density_and_prominence(smoothed_f_trace)

    trace_range = float(np.nanmax(smoothed_f_trace) - np.nanmin(smoothed_f_trace))

    features = {
                "derivative_skew": float(deriv_skew),
                "derivative_asymmetry": float(deriv_asym),
                "spike_prom_mean": float(spike_prom_mean),
                "spike_prom_skew": float(spike_prom_skew),
                "range_trace": trace_range,
                "var_of_var": float(var_of_var),
                "ac_decay": float(ac_decay),
                "snr_estimate": float(snr),
                "peak_density": float(peak_density),
                "median_spike_prom": float(median_spike_prom),
                }

    validity = {
        "valid_deriv_skew": bool(valid_deriv_skew),
        "valid_deriv_asym": bool(valid_deriv_asym),
        "valid_prom": bool(valid_prom),
        "valid_vov": bool(valid_vov),
        "valid_ac": bool(valid_ac),
        "valid_snr": bool(valid_snr),
        "valid_peak": bool(valid_peak),
    }

    return features, validity