import sys


import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import curve_fit

def find_spikes(
    spike_prob_trace,
    raw_fluorescence,
    sigma=2.0,
    window_radius=None,
    window_size=None,
    edge=32,
    **_ignored,
):
    """Detect spike candidates from Cascade probabilities.

    Parameters
    ----------
    spike_prob_trace : 1D array-like
        Raw cascade spike probability trace (may contain NaNs at edges).
    raw_fluorescence : 1D array-like
        Raw fluorescence trace aligned with the probability trace.
    sigma : float, optional
        Gaussian kernel sigma used to smooth the probability trace. Defaults to 2.
    window_radius : int, optional
        Half-width (in frames) of the fluorescence search window around each
        probability peak. Defaults to 5 frames.
    edge : int, optional
        Number of leading/trailing samples to ignore (designed for 32-frame NaN
        padding). Defaults to 32.

    Returns
    -------
    ((np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray), np.ndarray)
        A tuple containing spike metadata and the smoothed probability trace.
    """

    prob = np.asarray(spike_prob_trace, dtype=float)
    fluo = np.asarray(raw_fluorescence, dtype=float)

    if prob.shape != fluo.shape:
        raise ValueError("Spike probability and fluorescence traces must have matching shapes")

    n_frames = prob.size
    if n_frames == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=float),
        ), prob

    finite_mask = np.isfinite(prob)
    prob_filled = np.where(finite_mask, prob, 0.0)

    smoothed = gaussian_filter1d(prob_filled, sigma=sigma) if sigma and sigma > 0 else prob_filled
    smoothed_detection = smoothed.copy()

    if edge > 0:
        smoothed_detection[:edge] = 0.0
        smoothed_detection[-edge:] = 0.0

    peaks, _ = find_peaks(smoothed_detection)
    if peaks.size == 0:
        smoothed_output = smoothed.copy()
        smoothed_output[~finite_mask] = np.nan
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=float),
        ), smoothed_output

    valid_mask = np.ones(peaks.shape[0], dtype=bool)
    if edge > 0:
        valid_mask &= (peaks >= edge) & (peaks < (n_frames - edge))

    peaks = peaks[valid_mask]
    if peaks.size == 0:
        smoothed_output = smoothed.copy()
        smoothed_output[~finite_mask] = np.nan
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=float),
        ), smoothed_output

    _, left_bases, _ = peak_prominences(smoothed_detection, peaks)

    selected_prob_idx = []
    selected_prob_vals = []
    selected_proms = []
    selected_fluo_idx = []
    selected_fluo_vals = []

    if window_radius is None:
        window_radius = window_size if window_size is not None else 5

    half_window = int(max(1, window_radius))

    for peak_idx, left_base in zip(peaks, left_bases):
        start = max(0, int(peak_idx) - half_window)
        end = min(n_frames, int(peak_idx) + half_window + 1)
        window = fluo[start:end]
        if window.size == 0:
            continue
        window_finite = np.isfinite(window)
        if not np.any(window_finite):
            continue
        local_idx = int(np.nanargmax(np.where(window_finite, window, -np.inf)))
        fluo_idx = start + local_idx
        fluo_val = float(fluo[fluo_idx])

        selected_prob_idx.append(int(peak_idx))
        selected_prob_vals.append(float(smoothed_detection[peak_idx]))
        left_value = float(smoothed_detection[left_base]) if left_base >= 0 else 0.0
        selected_proms.append(float(smoothed_detection[peak_idx] - left_value))
        selected_fluo_idx.append(int(fluo_idx))
        selected_fluo_vals.append(fluo_val)

    smoothed_output = smoothed.copy()
    smoothed_output[~finite_mask] = np.nan

    return (
        np.asarray(selected_prob_idx, dtype=int),
        np.asarray(selected_prob_vals, dtype=float),
        np.asarray(selected_proms, dtype=float),
        np.asarray(selected_fluo_idx, dtype=int),
        np.asarray(selected_fluo_vals, dtype=float),
    ), smoothed_output

# --- Windowing ---
def window_spike_transients(fluorescence, spike_indices):
    """
    Partition a fluorescence trace into non-overlapping spike windows.

    Each window spans from the lowest point *after* the previous peak to the
    lowest point before the next peak, ensuring the sequence of windows tiles
    the trace without gaps.

    Args:
        fluorescence (array-like): Fluorescence trace (1D).
        spike_indices (array-like): Indices of spike peaks.

    Returns:
        list[tuple[int, int, int]]: (start_idx, peak_idx, end_idx) for each
        spike in the same order as `spike_indices`.
    """

    if fluorescence is None:
        return []

    peaks = np.asarray(spike_indices, dtype=int)
    if peaks.size == 0:
        return []

    n = len(fluorescence)
    if n == 0:
        return []

    peaks = np.clip(peaks, 0, n - 1)
    unique_peaks = np.unique(peaks)

    windows_unique = []

    first_peak = int(unique_peaks[0])
    if first_peak > 0:
        seg = fluorescence[: first_peak + 1]
        start = int(np.argmin(seg))
    else:
        start = 0
    start = max(0, min(start, first_peak))

    for idx, peak_val in enumerate(unique_peaks):
        peak = int(peak_val)
        if idx < unique_peaks.size - 1:
            next_peak = int(unique_peaks[idx + 1])
            if next_peak <= peak:
                end = peak
            else:
                seg = fluorescence[peak : next_peak + 1]
                end = peak + int(np.argmin(seg))
                end = max(peak, min(end, next_peak))
        else:
            if peak < n - 1:
                seg = fluorescence[peak:]
                end = peak + int(np.argmin(seg))
            else:
                end = peak
            end = max(peak, min(end, n - 1))

        windows_unique.append((int(start), peak, int(end)))
        start = int(end)

    windows_by_peak = {int(pk): win for pk, win in zip(unique_peaks, windows_unique)}
    return [windows_by_peak[int(pk)] for pk in peaks]

# --- Models ---
def exponential_decay(t, A, tau, offset):
    """Exponential decay: A * exp(-t/tau) + offset"""
    return A * np.exp(-t / tau) + offset

def linear_rise(t, m, b):
    """Linear rise: m * t + b"""
    return m * t + b

# --- Fitting functions ---
def fit_rise_constants(normed_segments, fs=30, rise_fraction=0.1):
    """
    Fit linear rise on normalized segments.
    Returns list of (m, b) tuples.
    """
    params = []
    for seg in normed_segments:
        peak_norm = seg.max()
        thresh = rise_fraction * peak_norm
        idxs = np.where(seg >= thresh)[0]
        seg2 = seg[idxs[0]:] if idxs.size else seg
        t = np.arange(len(seg2)) / fs

        m0 = (seg2[-1] - seg2[0]) / (t[-1] if t[-1] != 0 else 1)
        b0 = float(seg2[0])
        try:
            popt, _ = curve_fit(linear_rise, t, seg2, p0=[m0, b0], maxfev=10000)
        except Exception:
            popt = (np.nan, np.nan)
        params.append(tuple(popt))
    return params


def fit_decay_constants(normed_segments, fs=30, decay_fraction=0.9):
    """
    Fit exponential decay on normalized segments.
    Returns list of (A, tau, offset) tuples.
    """
    params = []
    for seg in normed_segments:
        peak_norm = seg[0]
        end_norm = seg[-1]
        thresh = peak_norm - decay_fraction * (peak_norm - end_norm)
        idxs = np.where(seg <= thresh)[0]
        seg2 = seg[:idxs[0] + 1] if idxs.size else seg
        t = np.arange(len(seg2)) / fs

        A0 = float(seg2[0] - seg2[-1])
        tau0 = (len(seg2) / fs) / 2
        off0 = float(seg2[-1])
        try:
            popt, _ = curve_fit(exponential_decay, t, seg2, p0=[A0, tau0, off0], maxfev=10000)
        except Exception:
            popt = (np.nan, np.nan, np.nan)
        params.append(tuple(popt))
    return params


def compute_spike_constants(
    fluo,
    peak_idx,
    fs=30,
    rise_fraction=0.1,
    decay_fraction=0.9,
):
    """
    Estimate rise slope and decay time constant for a single spike transient.

    The implementation avoids curve-fitting instability by
    (1) normalizing the transient using the local baseline and peak amplitude,
    (2) fitting a simple linear model to the rising segment, and
    (3) computing the time to decay to 1/e of the peak using linear interpolation.
    """

    windows = window_spike_transients(fluo, [peak_idx])
    if not windows:
        return np.nan, np.nan

    start, peak, end = windows[0]
    if end <= start:
        return np.nan, np.nan

    segment = np.asarray(fluo[start : end + 1], dtype=float)
    if segment.size < 3 or not np.isfinite(segment).all():
        return np.nan, np.nan

    baseline = float(np.nanmin(segment))
    peak_value = float(np.nanmax(segment))
    amplitude = peak_value - baseline
    if not np.isfinite(amplitude) or amplitude <= 1e-8:
        return np.nan, np.nan

    normed = (segment - baseline) / amplitude
    peak_rel = int(peak - start)
    peak_rel = max(0, min(peak_rel, normed.size - 1))

    # Rising slope via linear regression on samples up to the peak
    rise_segment = normed[: peak_rel + 1]
    if rise_segment.size < 2:
        rise_slope = np.nan
    else:
        t_rise = np.arange(rise_segment.size, dtype=float) / float(fs)
        try:
            slope, _ = np.polyfit(t_rise, rise_segment, 1)
        except Exception:
            slope = np.nan
        rise_slope = slope

    # Decay constant estimated from time to reach 1/e of peak amplitude
    decay_segment = normed[peak_rel:]
    if decay_segment.size < 2:
        decay_tau = np.nan
    else:
        t_decay = np.arange(decay_segment.size, dtype=float) / float(fs)
        target = np.exp(-1.0)
        below = np.where(decay_segment <= target)[0]
        if below.size == 0:
            positive = decay_segment > 1e-6
            if np.count_nonzero(positive) >= 2:
                t_fit = t_decay[positive]
                y_fit = np.log(decay_segment[positive])
                try:
                    slope, _ = np.polyfit(t_fit, y_fit, 1)
                except Exception:
                    slope = np.nan
                if np.isfinite(slope) and slope < 0:
                    decay_tau = -1.0 / slope
                else:
                    decay_tau = np.nan
            else:
                decay_tau = np.nan
        else:
            idx = int(below[0])
            if idx == 0:
                decay_tau = 0.0
            else:
                y0 = decay_segment[idx - 1]
                y1 = decay_segment[idx]
                x0 = t_decay[idx - 1]
                x1 = t_decay[idx]
                if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
                    decay_tau = t_decay[idx]
                else:
                    frac = (target - y0) / (y1 - y0)
                    frac = np.clip(frac, 0.0, 1.0)
                    decay_tau = x0 + frac * (x1 - x0)

    if not np.isfinite(decay_tau):
        if decay_segment.size >= 2:
            decay_tau = t_decay[-1]
        else:
            decay_tau = 0.0

    if not np.isfinite(rise_slope):
        rise_slope = np.nan

    return rise_slope, decay_tau

# --- Area under curve ---
def compute_area_under_curve(fluo, fs=30, baseline=None):
    """
    Compute total area under the entire fluorescence trace minus baseline.

    Args:
        fluo (1D array): raw fluorescence trace.
        fs (float): sampling frequency (Hz).
        baseline (float or None): baseline to subtract. If None, use global min.

    Returns:
        total_area (float): AUC of full trace minus baseline.
    """
    if baseline is None:
        baseline = np.min(fluo)
    total_area = np.trapz(fluo - baseline, dx=1.0/fs)
    return total_area

def find_valleys(cascade): 
    valley_idx = find_peaks(-cascade)[0]
    return valley_idx

