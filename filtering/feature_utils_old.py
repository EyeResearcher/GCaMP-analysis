import numpy as np

from scipy.signal import find_peaks, peak_prominences
from numpy.polynomial.polynomial import polyfit, Polynomial
from scipy.ndimage import gaussian_filter1d

import numpy as np
print(np.__file__)
from scipy.ndimage import gaussian_filter1d
from scipy.signal     import find_peaks, peak_prominences
from scipy.stats      import skew
def roughness(trace: np.ndarray) -> float:
    """Average absolute second derivative."""
    second_diffs = np.diff(np.diff(trace))
    return float(np.mean(np.abs(second_diffs)))


def normalize_z(trace: np.ndarray) -> np.ndarray:
    """Z-score normalize a 1D array."""
    t = np.asarray(trace, dtype=np.float64).flatten()
    return (t - t.mean()) / (t.std() + 1e-6)


def dtw_score(trace: np.ndarray, template: np.ndarray) -> float:
    """Negative normalized DTW distance between two z-scored sequences."""
    t1 = normalize_z(trace)
    t2 = normalize_z(template)
    dist, _ = fastdtw(t1, t2, dist=lambda x, y: abs(x - y))
    return -dist / max(len(t1), len(t2))


def multi_template_dtw_scores(trace: np.ndarray, templates: tuple) -> tuple:
    """DTW scores against few/med/many templates."""
    return (
        dtw_score(trace, templates[0]),
        dtw_score(trace, templates[1]),
        dtw_score(trace, templates[2]),
    )


def detrend_trace(trace: np.ndarray, degree: int = 2) -> tuple:
    """Remove a low-order polynomial trend."""
    x = np.arange(len(trace))
    p = Polynomial.fit(x, trace, deg=degree)
    trend = p(x)
    return trace - trend, trend


def linear_fit_mse(trace: np.ndarray) -> float:
    """MSE vs. straight line from mean(first10) to mean(last10)."""
    start, end = trace[:10].mean(), trace[-10:].mean()
    x = np.linspace(0, 1, len(trace))
    fit = start + (end - start) * x
    return float(np.mean((trace - fit) ** 2))


def spike_dtw_feature(
    sm: np.ndarray,
    spike_template: np.ndarray,
    peaks_valid: np.ndarray,
    cutoff: float
) -> float:
    """Avg DTW cost per spike-aligned segment."""
    
    tpl = np.ravel(spike_template)

    costs = []
    for i in range(len(peaks_valid) - 1):
        seg = sm[max(0, peaks_valid[i] - 5): np.argmin(sm[peaks_valid[i] : peaks_valid[i + 1]])]
        #if len(seg) == len(tpl):
        mn, mx = seg.min(), seg.max()
        if mx > mn:
            seg_n = ((seg - mn) / (mx - mn + 1e-6)).ravel()
            d, _ = fastdtw(seg_n, tpl, dist=lambda x, y: abs(x - y))
            costs.append(float(d))

    return float(np.mean(costs)) if costs else 999.0


def adjusted_peak_prominences(
    signal: np.ndarray,
    proms: np.ndarray,
    left: np.ndarray,
    right: np.ndarray
) -> np.ndarray:
    """prominence + abs(base_left - base_right)."""
    base_diff = np.abs(signal[left] - signal[right])
    return proms + base_diff


def get_prominence_stats(
    raw_sm: np.ndarray,
    norm_sm: np.ndarray,
    scaled_sm: np.ndarray,
    peaks_raw: np.ndarray,
    peaks_norm: np.ndarray,
    peaks_scaled: np.ndarray,
    cutoff_s: float = 0.15,
    cutoff_n: float = .1,
    cutoff_m: float = 15
) -> tuple:
    """Compute various adjusted & scaled prominence stats."""
    if peaks_raw.size == 0 or peaks_norm.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # raw
    pr, lr, rr = peak_prominences(raw_sm, peaks_raw)
    adj_pr = adjusted_peak_prominences(raw_sm, pr, lr, rr)
    # scale raw by peak height
    peaks_h = raw_sm[peaks_raw]
    scaled = adj_pr / (peaks_h + 1e-6)
    valid_s = scaled[scaled > cutoff_s]
    sum_s = float(valid_s.sum()) if valid_s.size else 0.0
    avg_s = float(valid_s.mean()) if valid_s.size else 0.0
    max_s = float(valid_s.max()) if valid_s.size else 0.0

    # normalized
    pn, ln, rn = peak_prominences(norm_sm, peaks_norm)
    adj_pn = adjusted_peak_prominences(norm_sm, pn, ln, rn)
    valid_n = adj_pn[adj_pn > cutoff_n]
    sum_n = float(valid_n.sum()) if valid_n.size else 0.0
    avg_n = float(valid_n.mean()) if valid_n.size else 0.0
    max_n = float(valid_n.max()) if valid_n.size else 0.0

    #scaled by mean 

    pm, lm, rm = peak_prominences(scaled_sm, peaks_scaled)
    adj_ps = adjusted_peak_prominences(scaled_sm, pm, lm, rm)
    valid_m = adj_ps[adj_ps > cutoff_m]
    valid_peaks = peaks_scaled[adj_ps > cutoff_m]
    sum_m = float(valid_m.sum()) if valid_m.size else 0.0
    avg_m = float(valid_m.mean()) if valid_m.size else 0.0
    max_m = float(valid_m.max()) if valid_m.size else 0.0

    return sum_s, avg_s, max_s, sum_n, avg_n, max_n, sum_m, avg_m, max_m, valid_peaks

def peak_linear_mse(peaks: np.ndarray, trace: np.ndarray) -> np.ndarray:
    """
    For each pair of consecutive peaks, fit a straight line between their fluorescence values
    and compute the Mean Squared Error (MSE) of the fit relative to the actual trace.

    Parameters
    ----------
    peaks : array-like of int
        Sorted indices of peaks in the fluorescence trace.
    trace : array-like of float
        1D fluorescence signal.

    Returns
    -------
    mses : np.ndarray
        Array of MSE values, one for each segment between consecutive peaks.
    """
    peaks = np.asarray(peaks, dtype=int)
    trace = np.asarray(trace, dtype=float)

    if peaks.size < 2:
        return np.array([0])

    mses = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        # actual signal segment
        y_true = trace[start:end + 1]
        # linear prediction between start and end peaks
        y0, y1 = trace[start], trace[end]
        x = np.arange(start, end + 1)
        y_pred = y0 + (y1 - y0) * (x - start) / (end - start)
        mse = np.mean((y_true - y_pred) ** 2)
        mses.append(mse)

    return np.array(mses)


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
def extract_features(
    summary_files : dict,
    template_traces: tuple = None,
    spike_template: np.ndarray = None
) -> dict:
    raw_trace = summary_files['raw_fluorescence']
    norm_trace = (raw_trace - raw_trace.min())/(raw_trace.max() - raw_trace.min())
    scaled_trace = raw_trace/np.mean(raw_trace) * 100
    # 0) Smoothing 
    # 1) smoothing
    sm_norm = gaussian_filter1d(norm_trace, sigma=4)
    sm_raw = gaussian_filter1d(raw_trace, sigma=4)
    sm_scaled = gaussian_filter1d(scaled_trace, sigma = 4)
    # 1) Find rough peaks
    peaks_raw, _  = find_peaks(sm_raw)
    peaks_norm, _ = find_peaks(sm_norm)
    peaks_mean_sclaed, _ = find_peaks(sm_scaled)

    # 2) roughness
    rough = roughness(sm_norm)

    # 3) DTW vs. few/med/many
    #dtw_f, dtw_m, dtw_M = multi_template_dtw_scores(sm_norm, template_traces)
    #dtw_best = min(dtw_f, dtw_m, dtw_M)

    # 4) per-spike DTW
   #spike_template_norm = (spike_template - spike_template.min( keepdims=True)) / (spike_template.max( keepdims=True) - spike_template.min( keepdims=True) + 1e-6)
   # spike_template_scaled = spike_template/np.mean(spike_template) * 100
    #spike_norm_dtw = spike_dtw_feature(sm_norm, spike_template_norm, peaks_norm, cutoff=.1)
    
    # 5) detrended range
    det, _ = detrend_trace(sm_scaled)
    rng_det = float(np.percentile(det, 98) - np.percentile(det, 2))

    # 6) slope
    slope = float(polyfit(np.arange(len(sm_scaled)), sm_scaled, 1)[1])

    # 7) mean first diff
    mean_dy = float(np.mean(np.diff(norm_trace)))

    # 8) prominence stats
    sum_s, avg_s, max_s, sum_n, avg_n, max_n, sum_m, avg_m, max_m, valid_peaks_m = get_prominence_stats(sm_raw, sm_norm, sm_scaled, peaks_raw, peaks_norm, peaks_mean_sclaed)
    #spike_scaled_dtw = spike_dtw_feature(sm_scaled, spike_template_scaled, valid_peaks_m, cutoff=.1)

    # 9) spikiness
    spikiness = (rough + avg_n) * (abs(slope) + 1e-3)

    # 10) linear MSE
    #lin_mse = linear_fit_mse(sm_scaled)

    lin_mse_peaks = np.mean(peak_linear_mse(valid_peaks_m, sm_scaled))

    return {
        "roughness": rough,
        #"dtw_few": dtw_f,
        #"dtw_med": dtw_m,
        #"dtw_many": dtw_M,
        #"dtw_best": dtw_best,
        #"spike_norm_dtw" :spike_norm_dtw,
        #"spike_scaled_dtw" : spike_scaled_dtw,
        "range_detrended": rng_det,
        #"slope": slope,
        #"mean_diff": mean_dy,
        "prom_sum_scaled": sum_s,
        "prom_avg_scaled": avg_s,
        #prom_max_scaled": max_s,
        #"prom_sum_norm": sum_n,
        #"prom_avg_norm": avg_n,
        #"prom_max_norm": max_n,
        #"prom_sum_mean_scaled" : sum_m,
        #"prom_avg_mean_scaled" : avg_m,
        "prom_max_mean_scaled" : max_m,
        "spikiness_index": spikiness,
        #"linear_mse": np.log10(lin_mse),
        "linear_mse_per_peak" : lin_mse_peaks
    }
