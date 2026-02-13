
import numpy as np
from scipy.stats import skew
from typing import Dict, Optional
from utils.feature_utils import compute_spike_constants, _create_large_window, _create_small_window
from scipy.signal import peak_prominences, peak_widths
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
    large_window: np.ndarray,
    small_window: np.ndarray,
    spike_prom: float,
    peak_idx: int,
    left_base_idx: int,
    absolute_prev_min: int,
    hierarchy: dict,
    i: int,
    top3 = False,
    trace_range: float = 1.0,
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


    peak_in_large_window = peak_idx - left_base_idx
    rise_slope, decay_tau = compute_spike_constants(
        small_window, 
        peak_in_large_window, 
        fs=15.0
    )
    decay_shape = compute_decay_shape_features(
        small_window, 
        peak_in_large_window, 
        fs=15.0
    )
    additional_decay = compute_additional_decay_features(
        small_window, 
        peak_in_large_window
    )

    mini_prom = large_window[peak_in_large_window] - small_window[0]
    if top3:
        return {
            "spike_prom": float(spike_prom)/trace_range,
            "dominance_score": float(hierarchy["dominance_score"][i]),
            #"prom_gap": float(hierarchy["prom_gap"][i]),
            "mini_prom": float(mini_prom)/trace_range,
            "distance": int(len(small_window)),}
    return {
        "spike_prom": float(spike_prom)/trace_range,
        "isolation": int(len(large_window)),
        "distance": int(len(small_window)),
        "iso_skew": float(skew(large_window)) if large_window.size else 0.0,
        "dist_skew": float(skew(small_window)) if small_window.size else 0.0,
        "iso_aai_sum": float(area_asymmetry(large_window, peak_idx - left_base_idx)),
        "dist_aai_sum": float(area_asymmetry(small_window, peak_idx - absolute_prev_min)),
        "iso_aai_trapz": float(area_asymmetry_trapz(large_window, zero_value=peak_idx - left_base_idx)),
        "dist_aai_trapz": float(area_asymmetry_trapz(small_window , zero_value=peak_idx - absolute_prev_min)),
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

def get_all_spike_features(smoothed_f, peaks, props : dict, mode = "train", roi_idx = None) -> list[str]:
    """Return list of all spike feature names."""
    trace_range = np.ptp(smoothed_f) if smoothed_f.size > 0 else 1.0
    prominences, left_bases, right_bases = peak_prominences(
        smoothed_f, peaks
    )

    widths = peak_widths(
        smoothed_f, peaks)
    widths = widths[0]  # extract widths array
    hierarchy = compute_peak_hierarchy_features(
    peaks=peaks,
    prominences=prominences,
    widths=widths,
    width_factor=1.5,
)
    
    from utils.label_utils import create_label_dict

    spike_data: Dict[int, Dict] = {}
    num_peaks = len(peaks)
    spike_keys = []
    windows_list = []
    features_list = []
    labels_list = []
  
    
    for i, peak in enumerate(peaks):
        large_window_f, absolute_left_base, absolute_right_base, spike_prom = _create_large_window(
            smoothed_f, peak, left_bases[i], right_bases[i]
        )
        # Adjust indices back to original array coordinates
        prev_peak = peaks[i - 1] if i > 0 else 0
        next_peak = peaks[i + 1] if i < num_peaks - 1 else len(smoothed_f)
        small_window_f, absolute_prev_min, absolute_next_min = _create_small_window(
            smoothed_f, peak, prev_peak, next_peak
        )
        
        features = _compute_spike_features(
            large_window_f, small_window_f, spike_prom,
            peak, left_bases[i], absolute_prev_min, hierarchy, i,
            top3=True, trace_range=trace_range
        )

        # Ensure the small window is non-empty and ordered.
        
        spike_key = peak if roi_idx is None else f"{roi_idx}_{peak}"
        spike_keys.append(spike_key)
        features_list.append(features)
        labels_list.append(create_label_dict(-1, 'unlabeled'))
        windows_list.append({
            'large_window': {
                'window_values': large_window_f, 
                'bounds': (absolute_left_base, absolute_right_base)
            },
            'small_window': {
                'window_values': small_window_f, 
                'bounds': (absolute_prev_min, absolute_next_min)
            }
        })
    if mode == "inference":
        return features_list, spike_keys
    spike_data = {
        peak: {"windows": windows, "features": features, "label": label}
        for peak, windows, features, label in zip(peaks, windows_list, features_list, labels_list)
    }
    return spike_data, spike_keys
    