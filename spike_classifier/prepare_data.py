import argparse
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks, peak_prominences, savgol_filter
from scipy.stats import skew
import pandas as pd

def load_roi_data(path: str) -> Dict:
    """Load ROI dictionary saved as a .npy file."""
    data = np.load(path, allow_pickle=True)
    # np.load with allow_pickle returns an array with a single dict item in this layout.
    return data.item()  # type: ignore[no-any-return]

def _compute_min_between(
    trace: np.ndarray, start: int, end: int
) -> int:
    """Return index of minimum value between start and end (exclusive of end)."""
    if start >= end:
        return start
    local_min = int(np.argmin(trace[start:end]))
    return start + local_min

def area_asymmetry(window: np.ndarray, zero_index: int) -> float:
    """
    Compute the Area-Asymmetry Index (AAI) of a 1D signal relative to a given zero index.
    
    AAI = (A_pos - A_neg) / (A_pos + A_neg)
    
    where:
        A_neg = sum(|signal[i]| for i < zero_index)
        A_pos = sum(|signal[i]| for i > zero_index)

    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values.
    zero_index : int
        Index representing the zero-reference boundary.

    Returns
    -------
    float
        Asymmetry index in [-1, 1].
    """
    left = np.sum(np.abs(window[:zero_index]))
    right = np.sum(np.abs(window[zero_index+1:]))

    if left + right == 0:
        return 0.0

    return (right - left) / (right + left)

def area_asymmetry_trapz(
    signal: np.ndarray,
    x: Optional[np.ndarray] = None,
    zero_value: float = 0.0
) -> float:
    """
    Compute the Area-Asymmetry Index (AAI) using trapezoidal integration
    of |signal| on each side of a zero-reference x-value.

        AAI = (A_pos - A_neg) / (A_pos + A_neg)

    where:
        A_neg = ∫_{x < zero_value} |signal(x)| dx
        A_pos = ∫_{x > zero_value} |signal(x)| dx

    Parameters
    ----------
    signal : np.ndarray
        1D array of y-values of the signal.
    x : np.ndarray, optional
        1D array of x-values (same length as signal). If None, uses
        x = np.arange(len(signal)).
    zero_value : float, optional
        The x-coordinate representing the zero boundary.

    Returns
    -------
    float
        Asymmetry index in [-1, 1]. Returns 0.0 if total area is zero.
    """
    signal = np.asarray(signal)
    if x is None:
        x = np.arange(signal.shape[0])
    else:
        x = np.asarray(x)
        assert x.shape == signal.shape, "x and signal must have same shape"

    abs_sig = np.abs(signal)

    left_mask = x < zero_value
    right_mask = x > zero_value

    if np.any(left_mask):
        A_left = np.trapz(abs_sig[left_mask], x[left_mask])
    else:
        A_left = 0.0

    if np.any(right_mask):
        A_right = np.trapz(abs_sig[right_mask], x[right_mask])
    else:
        A_right = 0.0

    total = A_left + A_right
    if total == 0:
        return 0.0

    return (A_right - A_left) / total
def _create_large_window(
    valid_spike_prob: np.ndarray,
    peak_idx: int,
    left_base_idx: int,
    right_base_idx: int,
    start_idx: int
) -> Tuple[np.ndarray, int, int, float]:
    """
    Create the large window (prominence-based) around a spike peak.
    
    Args:
        valid_spike_prob: Spike probability trace (valid region only)
        peak_idx: Peak index in valid region coordinates
        left_base_idx: Left base index in valid region coordinates
        right_base_idx: Right base index in valid region coordinates
        start_idx: Starting index of valid region in original array
    
    Returns:
        Tuple of (large_window, absolute_left_base, absolute_right_base, spike_prominence)
    """
    # Extract window in valid region coordinates
    large_window = valid_spike_prob[left_base_idx:right_base_idx]
    
    # Convert to absolute coordinates
    absolute_left_base = int(left_base_idx + start_idx)
    absolute_right_base = int(right_base_idx + start_idx)
    absolute_peak = int(peak_idx + start_idx)
    
    # Calculate spike prominence
    spike_prom = valid_spike_prob[peak_idx] - valid_spike_prob[left_base_idx]
    
    return large_window, absolute_left_base, absolute_right_base, float(spike_prom)


def _create_small_window(
    valid_spike_prob: np.ndarray,
    peak_idx: int,
    prev_peak_idx: int,
    next_peak_idx: int,
    start_idx: int
) -> Tuple[np.ndarray, int, int]:
    """
    Create the small window (inter-peak distance) around a spike peak.
    
    Args:
        valid_spike_prob: Spike probability trace (valid region only)
        peak_idx: Current peak index in valid region coordinates
        prev_peak_idx: Previous peak index (or 0 if first peak)
        next_peak_idx: Next peak index (or len(trace) if last peak)
        start_idx: Starting index of valid region in original array
    
    Returns:
        Tuple of (small_window, absolute_prev_min, absolute_next_min)
    """
    # Find local minima between peaks
    prev_min = _compute_min_between(valid_spike_prob, prev_peak_idx, peak_idx)
    next_min = _compute_min_between(valid_spike_prob, peak_idx, next_peak_idx)
    

    # Ensure the small window is non-empty and ordered
    if next_min <= prev_min:
        next_min = prev_min + 1 if prev_min + 1 < len(valid_spike_prob) else len(valid_spike_prob)
    
    # Extract window in valid region coordinates
    small_window = valid_spike_prob[prev_min:next_min]
    
    # Convert to absolute coordinates
    absolute_prev_min = int(prev_min + start_idx)
    absolute_next_min = int(next_min + start_idx)
    
    return small_window, absolute_prev_min, absolute_next_min
def compute_spike_constants(
    window: np.ndarray,
    peak_idx_in_window: int,
    fs: float = 30.0,
    rise_fraction: float = 0.1,
    decay_fraction: float = 0.9,
    ) -> Tuple[float, float]:
    """
    Estimate rise slope and decay time constant for a single spike transient.
    
    The implementation avoids curve-fitting instability by:
    (1) normalizing the transient using the local baseline and peak amplitude,
    (2) fitting a simple linear model to the rising segment, and
    (3) computing the time to decay to 1/e of the peak using linear interpolation.
    
    Args:
        window: Spike probability window containing the spike transient
        peak_idx_in_window: Index of the peak within the window (relative to window start)
        fs: Sampling frequency in Hz (default: 30.0)
        rise_fraction: Fraction of peak amplitude to start rise fitting (default: 0.1)
        decay_fraction: Fraction of peak amplitude to start decay fitting (default: 0.9)
    
    Returns:
        Tuple of (rise_slope, decay_tau):
            - rise_slope: Linear slope of the rising phase (normalized units/second)
            - decay_tau: Decay time constant in seconds (time to reach 1/e of peak)
    """
    segment = np.asarray(window, dtype=float)
    
    # Validate input
    if segment.size < 3 or not np.isfinite(segment).all():
        return np.nan, np.nan
    
    # Normalize the segment
    baseline = float(np.nanmin(segment))
    peak_value = float(np.nanmax(segment))
    amplitude = peak_value - baseline
    
    if not np.isfinite(amplitude) or amplitude <= 1e-8:
        return np.nan, np.nan
    
    normed = (segment - baseline) / amplitude
    peak_rel = int(peak_idx_in_window)
    peak_rel = max(0, min(peak_rel, normed.size - 1))
    
    # ===== Rising slope via linear regression =====
    rise_segment = normed[:peak_rel + 1]
    if rise_segment.size < 2:
        rise_slope = np.nan
    else:
        t_rise = np.arange(rise_segment.size, dtype=float) / float(fs)
        try:
            slope, _ = np.polyfit(t_rise, rise_segment, 1)
        except Exception:
            slope = np.nan
        rise_slope = slope
    
    # ===== Decay constant estimated from time to reach 1/e of peak =====
    decay_segment = normed[peak_rel:]
    if decay_segment.size < 2:
        decay_tau = np.nan
    else:
        t_decay = np.arange(decay_segment.size, dtype=float) / float(fs)
        target = np.exp(-1.0)  # 1/e ≈ 0.368
        below = np.where(decay_segment <= target)[0]
        
        if below.size == 0:
            # Extrapolate using exponential fit if target not reached
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
            # Interpolate to find exact crossing point
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
        
        # Fallback if decay_tau is still invalid
        if not np.isfinite(decay_tau):
            if decay_segment.size >= 2:
                decay_tau = t_decay[-1]
            else:
                decay_tau = 0.0
    
    # Final validation
    if not np.isfinite(rise_slope):
        rise_slope = np.nan
    
    return rise_slope, decay_tau

def compute_decay_shape_features(
    window: np.ndarray,
    peak_idx_in_window: int,
    fs: float = 30.0,
) -> dict:
    """
    Compute decay shape features beyond the time constant.
    
    Returns:
        dict with keys:
            - decay_r2: R² of exponential fit (goodness of fit)
            - decay_residual_std: Std of residuals (fit quality)
            - decay_curvature: Second derivative at decay midpoint
            - decay_biphasic: Evidence of two-phase decay
    """
    segment = np.asarray(window, dtype=float)
    baseline = float(np.nanmin(segment))
    peak_value = float(np.nanmax(segment))
    amplitude = peak_value - baseline
    
    if not np.isfinite(amplitude) or amplitude <= 1e-8:
        return {k: np.nan for k in ['decay_r2', 'decay_residual_std', 
                                      'decay_curvature', 'decay_biphasic_ratio']}
    
    normed = (segment - baseline) / amplitude
    peak_rel = max(0, min(peak_idx_in_window, normed.size - 1))
    
    # Decay segment
    decay_segment = normed[peak_rel:]
    if decay_segment.size < 3:
        return {k: np.nan for k in ['decay_r2', 'decay_residual_std', 
                                      'decay_curvature', 'decay_biphasic_ratio']}
    
    t_decay = np.arange(decay_segment.size, dtype=float) / float(fs)
    
    # 1. Exponential fit quality (R²)
    positive = decay_segment > 1e-6
    if np.count_nonzero(positive) >= 2:
        t_fit = t_decay[positive]
        y_fit = np.log(decay_segment[positive])
        try:
            slope, intercept = np.polyfit(t_fit, y_fit, 1)
            y_pred = slope * t_fit + intercept
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            residual_std = np.std(y_fit - y_pred)
        except:
            r2 = np.nan
            residual_std = np.nan
    else:
        r2 = np.nan
        residual_std = np.nan
    
    # 2. Curvature at decay midpoint (second derivative)
    if decay_segment.size >= 5:
        mid_idx = len(decay_segment) // 2
        # Use finite differences for second derivative
        if mid_idx > 1 and mid_idx < len(decay_segment) - 2:
            second_deriv = (decay_segment[mid_idx + 1] - 2 * decay_segment[mid_idx] + 
                           decay_segment[mid_idx - 1]) * (fs ** 2)
            curvature = float(second_deriv)
        else:
            curvature = np.nan
    else:
        curvature = np.nan
    
    # 3. Biphasic decay detection
    # Fit two exponentials and compare to single exponential
    if decay_segment.size >= 6 and np.count_nonzero(positive) >= 4:
        # Split decay into two halves
        mid = len(decay_segment) // 2
        decay_first = decay_segment[:mid][decay_segment[:mid] > 1e-6]
        decay_second = decay_segment[mid:][decay_segment[mid:] > 1e-6]
        
        if len(decay_first) >= 2 and len(decay_second) >= 2:
            try:
                # Fit each half
                t_first = np.arange(len(decay_first)) / fs
                t_second = np.arange(len(decay_second)) / fs
                
                slope1, _ = np.polyfit(t_first, np.log(decay_first), 1)
                slope2, _ = np.polyfit(t_second, np.log(decay_second), 1)
                
                # Ratio of decay rates (>2 suggests biphasic)
                tau1 = -1.0 / slope1 if slope1 < 0 else np.nan
                tau2 = -1.0 / slope2 if slope2 < 0 else np.nan
                
                if np.isfinite(tau1) and np.isfinite(tau2) and tau2 > 0:
                    biphasic_score = tau2 / tau1  # Slower phase / faster phase
                else:
                    biphasic_score = 1.0
            except:
                biphasic_score = np.nan
        else:
            biphasic_score = np.nan
    else:
        biphasic_score = np.nan
    
    return {
        'decay_r2': float(r2),
        'decay_residual_std': float(residual_std),
        'decay_curvature': float(curvature),
        'decay_biphasic_ratio': float(biphasic_score)
    }

def compute_additional_decay_features(
    window: np.ndarray,
    peak_idx_in_window: int,
) -> dict:
    """More decay shape characteristics."""
    segment = np.asarray(window, dtype=float)
    baseline = float(np.nanmin(segment))
    peak_value = float(np.nanmax(segment))
    amplitude = peak_value - baseline
    
    if not np.isfinite(amplitude) or amplitude <= 1e-8:
        return {k: np.nan for k in ['decay_skew', 'decay_kurtosis', 
                                      'decay_linearity']}
    
    normed = (segment - baseline) / amplitude
    peak_rel = max(0, min(peak_idx_in_window, normed.size - 1))
    decay_segment = normed[peak_rel:]
    
    if decay_segment.size < 3:
        return {k: np.nan for k in ['decay_skew', 'decay_kurtosis', 
                                      'decay_linearity']}
    
    from scipy.stats import skew, kurtosis
    
    # Skewness of decay (symmetry)
    decay_skew = float(skew(decay_segment))
    
    # Kurtosis (tail heaviness)
    decay_kurt = float(kurtosis(decay_segment))
    
    # Linearity in log space (deviation from exponential)
    positive = decay_segment > 1e-6
    if np.count_nonzero(positive) >= 3:
        log_decay = np.log(decay_segment[positive])
        # Fit line and measure deviation
        x = np.arange(len(log_decay))
        try:
            coeffs = np.polyfit(x, log_decay, 1)
            line_fit = np.polyval(coeffs, x)
            deviation = np.std(log_decay - line_fit)
            linearity = 1.0 / (1.0 + deviation)  # 1 = perfect line, 0 = nonlinear
        except:
            linearity = np.nan
    else:
        linearity = np.nan
    
    return {
        'decay_skew': float(decay_skew),
        'decay_kurtosis': float(decay_kurt),
        'decay_linearity': float(linearity)
    }
def build_peak_clusters(
    peaks: np.ndarray,
    widths: np.ndarray,
    width_factor: float = 1.5,
) -> list[np.ndarray]:
    """
    Group 1D peaks into local clusters based on time proximity.

    Peaks that are closer than (width_factor * median(width)) in index units
    will be placed in the same cluster.

    Parameters
    ----------
    peaks : np.ndarray
        1D array of peak indices (e.g. from scipy.signal.find_peaks).
    widths : np.ndarray
        1D array of peak widths (same length/order as peaks).
        If you don't have true widths, you can approximate with
        right_base - left_base.
    width_factor : float, optional
        Multiplier on the typical (median) width to define the clustering
        radius. Default is 1.5.

    Returns
    -------
    clusters : list[np.ndarray]
        Each element is a 1D np.ndarray of integer indices *into the peaks
        array* indicating which peaks belong to that cluster.
    """
    if peaks.size == 0:
        return []

    # Sort peaks by location in time
    order = np.argsort(peaks)
    peaks_sorted = peaks[order]

    # Robust typical width
    if widths.size == 0:
        typical_width = 1.0
    else:
        typical_width = float(np.median(widths))
        if not np.isfinite(typical_width) or typical_width <= 0:
            typical_width = 1.0

    radius = width_factor * typical_width

    clusters: list[np.ndarray] = []
    current_cluster = [order[0]]  # store indices into original peaks array

    for prev_idx, cur_idx in zip(order[:-1], order[1:]):
        # distance between successive peaks in time
        if (peaks[cur_idx] - peaks[prev_idx]) <= radius:
            current_cluster.append(cur_idx)
        else:
            clusters.append(np.array(current_cluster, dtype=int))
            current_cluster = [cur_idx]

    # add last cluster
    clusters.append(np.array(current_cluster, dtype=int))

    return clusters


def compute_peak_hierarchy_features(
    peaks: np.ndarray,
    prominences: np.ndarray,
    widths: np.ndarray,
    width_factor: float = 1.5,
) -> dict:
    """
    Compute local hierarchy features (parent/child-like metrics) for each peak.

    Features are computed per peak, based on prominence and local clustering:

        - dominance_score: prominence / max prominence in cluster
        - local_rank: rank within cluster (0 = largest prominence)
        - local_rank_norm: normalized rank in [0, 1]
        - cluster_size: number of peaks in cluster
        - prom_gap: (parent_prom - prom) / parent_prom in [0, 1]
        - time_to_parent: |peak_index - parent_peak_index| in samples

    Parameters
    ----------
    peaks : np.ndarray
        1D array of peak indices in the trace.
    prominences : np.ndarray
        1D array of prominences (same length/order as peaks).
    widths : np.ndarray
        1D array of widths (same length/order as peaks).
    width_factor : float, optional
        Passed to build_peak_clusters to control cluster radius.

    Returns
    -------
    features : dict[str, np.ndarray]
        Dict with keys:
            'dominance_score', 'local_rank', 'local_rank_norm',
            'cluster_size', 'prom_gap', 'time_to_parent'.
        All arrays have shape (n_peaks,).
    """
    n = peaks.size
    if n == 0:
        return {
            "dominance_score": np.array([], dtype=float),
            "local_rank": np.array([], dtype=int),
            "local_rank_norm": np.array([], dtype=float),
            "cluster_size": np.array([], dtype=int),
            "prom_gap": np.array([], dtype=float),
            "time_to_parent": np.array([], dtype=float),
        }

    clusters = build_peak_clusters(peaks, widths, width_factor=width_factor)

    dominance_score = np.zeros(n, dtype=float)
    local_rank = np.zeros(n, dtype=int)
    local_rank_norm = np.zeros(n, dtype=float)
    cluster_size = np.zeros(n, dtype=int)
    prom_gap = np.zeros(n, dtype=float)
    time_to_parent = np.zeros(n, dtype=float)

    eps = 1e-9

    for cl in clusters:
        # cl is an array of indices into peaks/prominences/widths
        cl_prom = prominences[cl]
        cl_peaks = peaks[cl]

        # parent: largest prominence in this cluster
        parent_idx_in_cl = int(np.argmax(cl_prom))
        parent_global_idx = int(cl[parent_idx_in_cl])
        parent_prom = float(cl_prom[parent_idx_in_cl])
        parent_pos = int(cl_peaks[parent_idx_in_cl])

        # rank within cluster by prominence (0 = largest)
        # sort in descending order
        rank_order = np.argsort(-cl_prom)
        rank_of = np.empty_like(rank_order)
        rank_of[rank_order] = np.arange(len(cl))

        for j, global_idx in enumerate(cl):
            r = int(rank_of[j])
            p = float(prominences[global_idx])

            cluster_size[global_idx] = len(cl)
            local_rank[global_idx] = r

            if len(cl) > 1:
                local_rank_norm[global_idx] = r / float(len(cl) - 1)
            else:
                local_rank_norm[global_idx] = 0.0

            # dominance_score: own prominence vs cluster parent
            dominance_score[global_idx] = p / (parent_prom + eps)

            # prom_gap: 0 for parent, >0 for children (up to 1)
            prom_gap[global_idx] = (parent_prom - p) / (parent_prom + eps)

            # time_to_parent: distance in samples to parent peak
            time_to_parent[global_idx] = abs(int(peaks[global_idx]) - parent_pos)

    return {
        "dominance_score": dominance_score,
        "local_rank": local_rank,
        "local_rank_norm": local_rank_norm,
        "cluster_size": cluster_size,
        "prom_gap": prom_gap,
        "time_to_parent": time_to_parent,
    }
def _compute_spike_features(
    large_window_sp: np.ndarray,
    small_window_sp: np.ndarray,
    large_window_f: np.ndarray,
    small_window_f: np.ndarray,
    spike_prom: float,
    peak_idx: int,
    left_base_idx: int,
    absolute_prev_min: int,
    hierarchy: dict,
    i: int
) -> dict:
    """
    Compute all features for a detected spike.
    
    Args:
        large_window: Prominence-based window around spike
        small_window: Inter-peak window around spike
        spike_prom: Spike prominence value
        peak_idx: Peak index in valid region coordinates
        left_base_idx: Left base index in valid region coordinates
        absolute_prev_min: Absolute index of previous minimum
    
    Returns:
        Dictionary of spike features
    """
    if len(large_window_sp) >= 7:
        smoothed_small_f = savgol_filter(small_window_f, window_length=7, polyorder=3, mode='nearest')
    else:
        smoothed_small_f = small_window_f
    
    # Compute peak position in large window
    peak_in_large_window = peak_idx - left_base_idx
    
    # Kinetic features from SMOOTHED trace
    rise_slope, decay_tau = compute_spike_constants(
        smoothed_small_f, 
        peak_in_large_window, 
        fs=30.0
    )
    
    # Decay shape features from SMOOTHED trace
    decay_shape = compute_decay_shape_features(
        smoothed_small_f, 
        peak_in_large_window, 
        fs=30.0
    )
    
    additional_decay = compute_additional_decay_features(
        smoothed_small_f, 
        peak_in_large_window
    )

    return {
        "spike_prom": float(spike_prom),
        "isolation": int(len(large_window_sp)),
        "distance": int(len(small_window_sp)),
        "iso_skew": float(skew(large_window_sp)) if large_window_sp.size else 0.0,
        "dist_skew": float(skew(small_window_sp)) if small_window_sp.size else 0.0,
        "iso_aai_sum": float(area_asymmetry(large_window_sp, peak_idx - left_base_idx)),
        "dist_aai_sum": float(area_asymmetry(small_window_sp, peak_idx - absolute_prev_min)),
        "iso_aai_trapz": float(area_asymmetry_trapz(large_window_sp, zero_value=peak_idx - left_base_idx)),
        "dist_aai_trapz": float(area_asymmetry_trapz(small_window_sp, zero_value=peak_idx - absolute_prev_min)),
        "rise_slope": float(rise_slope),
        "decay_tau": float(decay_tau),
        
        # Decay shape features (from smoothed trace)
        **decay_shape,
        **additional_decay,
        
        "dominance_score": float(hierarchy["dominance_score"][i]),
        "local_rank": int(hierarchy["local_rank"][i]),
        "local_rank_norm": float(hierarchy["local_rank_norm"][i]),
        "cluster_size": int(hierarchy["cluster_size"][i]),
        "prom_gap": float(hierarchy["prom_gap"][i]),
        "time_to_parent": float(hierarchy["time_to_parent"][i]),
    
    }

def smooth_for_kinetics(trace: np.ndarray, window_length: int = 5, polyorder: int = 3) -> np.ndarray:
    """
    Smooth trace using Savitzky-Golay filter.
    
    Args:
        trace: Raw fluorescence trace
        window_length: Must be odd, typically 5-11 frames (default: 7 ≈ 0.23s at 30Hz)
        polyorder: Polynomial order, typically 2-3 (default: 3)
    
    Returns:
        Smoothed trace
    """
    if len(trace) < window_length:
        return trace
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    from scipy.signal import savgol_filter
    return savgol_filter(trace, window_length, polyorder, mode='nearest')

def detect_spikes(smoothed_spike_prob: np.ndarray = None, f_trace: np.ndarray = None,
                   peaks : np.ndarray | None = None, roi_idx = None,
                     mode = "train") -> Tuple[Dict[int, Dict], list[int|str]]:
    """
    Detect spikes and compute windows/features per spike.
    Args:
        smoothed_spike_prob (np.ndarray): 1D array of smoothed spike probabilities
        peaks (np.ndarray | None): Optional precomputed peak indices
        roi_idx: Optional ROI index for key construction
        mode: "train" to return detailed spike data, "inference" for feature list only
    Returns:
        Tuple containing for training mode:
            - spike_data (Dict[int, Dict]): Mapping of spike index to its data (windows, features, label)
            - spike_keys (list[int|str]): List of spike indices detected
        or for inference mode:
            - features_list (list[Dict]): List of feature dictionaries for each spike
            - spike_keys (list[int|str]): List of spike keys

    """
    # Find the valid region (non-NaN) values)
    valid_mask = ~np.isnan(smoothed_spike_prob)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return {}, []
    
    # Get the valid region bounds
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1
    
    # Extract the valid portion
    valid_spike_prob = smoothed_spike_prob[start_idx:end_idx]
    
    # Detect peaks in the valid region
    peaks, _ = find_peaks(valid_spike_prob) if peaks is None else (peaks, None)
    if peaks.size == 0:
        return {}, []

    prominences, left_bases, right_bases = peak_prominences(
        valid_spike_prob, peaks
    )

    widths = right_bases - left_bases

# Peak hierarchy / child-peakness features (per peak)
    hierarchy = compute_peak_hierarchy_features(
    peaks=peaks,
    prominences=prominences,
    widths=widths,
    width_factor=1.5,
)

    spike_data: Dict[int, Dict] = {}
    num_peaks = len(peaks)
    spike_keys = []
    inference_list = []
    for i, peak in enumerate(peaks):
        large_window_sp, absolute_left_base, absolute_right_base, spike_prom = _create_large_window(
            valid_spike_prob, peak, left_bases[i], right_bases[i], start_idx
        )
        # Adjust indices back to original array coordinates
        prev_peak = peaks[i - 1] if i > 0 else 0
        next_peak = peaks[i + 1] if i < num_peaks - 1 else len(valid_spike_prob)

        small_window_sp, absolute_prev_min, absolute_next_min = _create_small_window(
            valid_spike_prob, peak, prev_peak, next_peak, start_idx
        )
        large_window_f, small_window_f = f_trace[absolute_left_base:absolute_right_base], f_trace[absolute_prev_min:absolute_next_min]
        features = _compute_spike_features(
            large_window_sp, small_window_sp, large_window_f, small_window_f, spike_prom,
            peak, left_bases[i], absolute_prev_min, hierarchy, i
        )
       
        # Ensure the small window is non-empty and ordered.
        absolute_peak = int(peak + start_idx)
        spike_key = absolute_peak if roi_idx is None else f"{roi_idx}_{absolute_peak}"
        spike_keys.append(spike_key)
        if mode == "inference":
            inference_list.append(features)
            continue
            
        spike_data[absolute_peak] = {
            "windows": {
                'large_window': {
                    'window_values': large_window_sp, 
                    'bounds': (absolute_left_base, absolute_right_base)
                },
                'small_window': {
                    'window_values': small_window_sp, 
                    'bounds': (absolute_prev_min, absolute_next_min)
                }
            },
            "features": features,
            "label": -1,  
        }
    
      
    
    if mode == "inference":
        return inference_list, spike_keys

    
    return spike_data, spike_keys

def parallel_detect_spikes(args):
    smoothed_spike_prob, f_trace, roi_idx = args
    spike_data, spike_keys = detect_spikes(smoothed_spike_prob, f_trace=f_trace, roi_idx=roi_idx)
    return spike_data, spike_keys
def extract_spike_features(roi_dict: Dict, max_rois: Optional[int] = None) -> Tuple[Dict, list]:
    """
    Iterate through ROI data and collect spike features for labeled-good ROIs (label == 1).
    
    Args:
        roi_dict: Dictionary of ROI data
        max_rois: Maximum number of good ROIs to process (None = all)
    
    Returns: (roi_dict_with_spikes, all_roi_spike_keys)
    """
    all_roi_spike_keys = []
    processed_count = 0
    bad_roi = 0
    for roi_key, roi_data in roi_dict.items():
        if roi_data.get("label", -1) != 1:
            bad_roi +=1
            continue
        
        # Check if we've reached the max ROI limit
        if max_rois is not None and processed_count >= max_rois:
            print(f"✓ Reached max_rois limit ({max_rois}), stopping spike extraction")
            break

        smoothed_traces = roi_data.get("smoothed_traces", [])
        raw_f_traces = roi_data.get("raw_traces", [])
        if len(smoothed_traces) < 2:
            continue

        smoothed_spike_prob = np.asarray(smoothed_traces[0])
        roi_spike_data, spike_keys = detect_spikes(smoothed_spike_prob, raw_f_traces[0])
        
        # Preserve existing labels if spikes already exist
        existing_spikes = roi_data.get('spikes', {})
        for spike_idx in spike_keys:
            if spike_idx in existing_spikes:
                # Preserve the existing label
                roi_spike_data[spike_idx]['label'] = existing_spikes[spike_idx]['label']
        
        roi_spike_keys = [(f"{roi_key}-{spike_idx}", roi_spike_data[spike_idx]['label']) for spike_idx in spike_keys]
        all_roi_spike_keys.extend(roi_spike_keys)
        roi_dict[roi_key]['spikes'] = roi_spike_data
        processed_count += 1

    print(f"✓ Processed {processed_count} good ROIs, found {len(all_roi_spike_keys)} spikes")
    print(f"✗ Skipped {bad_roi} bad ROIs")
    return roi_dict, all_roi_spike_keys


def main(input_path: str, output_path: Optional[str] = None, max_rois: Optional[int] = None) -> Dict[str, Dict[int, Dict]]:
    roi_dict = load_roi_data(input_path)
    
    
    roi_dict_spikes, all_roi_spike_keys = extract_spike_features(roi_dict, max_rois=max_rois)
    
    if output_path:
        np.save(output_path, roi_dict_spikes)
        all_roi_spike_keys_path = output_path.replace('.npy', '_spike_keys.csv')
        df = pd.DataFrame(all_roi_spike_keys, columns=['spike_key', 'label'])
        df.to_csv(all_roi_spike_keys_path, index=False)
    if not output_path:
        np.save(input_path, roi_dict_spikes)
        all_roi_spike_keys_path = input_path.replace('.npy', '_spike_keys.csv')
        df = pd.DataFrame(all_roi_spike_keys, columns=['spike_key', 'label'])
        df.to_csv(all_roi_spike_keys_path, index=False)
    return roi_dict_spikes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare spike feature data from fluorescence traces."
    )
    parser.add_argument("--input_path", help="Path to .npy file containing ROI dictionary.", default="training_data/roi_filtering/all_roi_features.npy")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="Optional path to save extracted spike features (.npy).",
    )
    parser.add_argument(
        "--max_rois",
        type=int,
        default=None,
        help="Maximum number of good ROIs to process (default: all ROIs)",
    )

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.max_rois)
