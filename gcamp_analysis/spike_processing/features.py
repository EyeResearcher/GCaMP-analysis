"""Spike-level feature computation.

All spike feature math lives here.  Both the training pipeline
(spike_classifier.prepare_data) and the inference pipeline
(spike_processing.filtering → SpikeService)
ultimately call describe_spikes / get_spike_feats from this module.

Peak-hierarchy helpers (formerly hierarchy.py) are included at the top.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .kinetics import _create_small_window


# =====================================================================
#  PEAK HIERARCHY (formerly hierarchy.py)
# =====================================================================


def build_peak_clusters(
    peaks: np.ndarray,
    widths: np.ndarray,
    width_factor: float = 1.5,
) -> list[np.ndarray]:
    """Group peaks into local clusters based on time proximity."""
    if peaks.size == 0:
        return []

    order = np.argsort(peaks)
    peaks_sorted = peaks[order]

    typical_width = float(np.median(widths)) if widths.size > 0 else 1.0
    if not np.isfinite(typical_width) or typical_width <= 0:
        typical_width = 1.0

    radius = width_factor * typical_width
    clusters: list[np.ndarray] = []
    current_cluster = [order[0]]

    for prev_idx, cur_idx in zip(order[:-1], order[1:]):
        if (peaks[cur_idx] - peaks[prev_idx]) <= radius:
            current_cluster.append(cur_idx)
        else:
            clusters.append(np.array(current_cluster, dtype=int))
            current_cluster = [cur_idx]

    clusters.append(np.array(current_cluster, dtype=int))
    return clusters

def assign_p_prom(peaks : np.ndarray, 
                  clusters : list[np.ndarray], 
                  prominences : np.ndarray)-> np.ndarray:
    
    """
    This function takes clusters, peaks, and prominences as inputs. 

    It iterates through clusters (arrays of indices of peaks in peak array) and assignes a parent prominence to each. 
    
    Parameters 
    ----------
    peaks : np.ndarrray
        array of indices of peaks
    clusters : list[np.ndarray]
        list of arrays where each array is a set of integers specifying index of peak within peaks
    prominences : np.ndarray
        list of prominences as computed by scipys ``peak_prominences`` 
    
    Returns
    -------
    cluster_parent_prom : np.ndarray
        array where each entry corresponds to a peak's parent's prominence
        """
    cluster_parent_prom = np.empty(len(peaks), dtype=float)
    for cl in clusters:
        parent_prom = float(np.max(prominences[cl]))
        cluster_parent_prom[cl] = parent_prom
    return cluster_parent_prom

def compute_peak_hierarchy_features(
    peaks: np.ndarray,
    prominences: np.ndarray,
    widths: np.ndarray,
    width_factor: float = 1.5,
) -> dict:
    """Compute local hierarchy features for each peak.

    Currently returns only ``dominance_score``.  Additional per-peak
    features (local_rank, cluster_size, prom_gap, time_to_parent) can
    be derived from the same cluster structure — see commented block below.
    """
    n = peaks.size
    if n == 0:
        return {"dominance_score": np.array([], dtype=float)}

    clusters = build_peak_clusters(peaks, widths, width_factor=width_factor)

    dominance_score = np.zeros(n, dtype=float)
    eps = 1e-9

    for cl in clusters:
        parent_prom = float(np.max(prominences[cl]))
        for idx in cl:
            dominance_score[idx] = float(prominences[idx]) / (parent_prom + eps)

    # To re-enable additional hierarchy features, uncomment and add to return:
    #
    # local_rank = np.zeros(n, dtype=int)
    # local_rank_norm = np.zeros(n, dtype=float)
    # cluster_size = np.zeros(n, dtype=int)
    # prom_gap = np.zeros(n, dtype=float)
    # time_to_parent = np.zeros(n, dtype=float)
    #
    # for cl in clusters:
    #     cl_prom = prominences[cl]
    #     parent_idx = int(np.argmax(cl_prom))
    #     parent_prom = float(cl_prom[parent_idx])
    #     parent_pos = int(peaks[cl[parent_idx]])
    #     rank_order = np.argsort(-cl_prom)
    #     rank_of = np.empty_like(rank_order)
    #     rank_of[rank_order] = np.arange(len(cl))
    #     for j, gidx in enumerate(cl):
    #         cluster_size[gidx] = len(cl)
    #         local_rank[gidx] = int(rank_of[j])
    #         local_rank_norm[gidx] = rank_of[j] / (len(cl) - 1) if len(cl) > 1 else 0.0
    #         prom_gap[gidx] = (parent_prom - prominences[gidx]) / (parent_prom + eps)
    #         time_to_parent[gidx] = abs(int(peaks[gidx]) - parent_pos)

    return {"dominance_score": dominance_score}


# =====================================================================
#  ASYMMETRY
# =====================================================================


def area_asymmetry(window: np.ndarray, zero_index: int) -> float:
    """AAI using summation."""
    left = np.sum(np.abs(window[:zero_index]))
    right = np.sum(np.abs(window[zero_index + 1:]))
    if left + right == 0:
        return 0.0
    return (right - left) / (right + left)


def area_asymmetry_trapz(
    signal: np.ndarray,
    x: Optional[np.ndarray] = None,
    zero_value: float = 0.0,
) -> float:
    """AAI using trapezoidal integration."""
    signal = np.asarray(signal)
    if x is None:
        x = np.arange(signal.shape[0])
    else:
        x = np.asarray(x)

    abs_sig = np.abs(signal)
    left_mask = x < zero_value
    right_mask = x > zero_value

    A_left = np.trapz(abs_sig[left_mask], x[left_mask]) if np.any(left_mask) else 0.0
    A_right = np.trapz(abs_sig[right_mask], x[right_mask]) if np.any(right_mask) else 0.0

    total = A_left + A_right
    if total == 0:
        return 0.0
    return (A_right - A_left) / total


# =====================================================================
#  DECAY SHAPE
# =====================================================================


def compute_decay_shape_features(
    window: np.ndarray,
    peak_idx_in_window: int,
    fs: float = 30.0,
) -> dict:
    """Decay R², residual std, curvature, biphasic ratio."""
    segment = np.asarray(window, dtype=float)
    baseline = float(np.nanmin(segment))
    peak_value = float(np.nanmax(segment))
    amplitude = peak_value - baseline

    nan_result = {k: np.nan for k in [
        'decay_r2', 'decay_residual_std', 'decay_curvature', 'decay_biphasic_ratio'
    ]}
    if not np.isfinite(amplitude) or amplitude <= 1e-8:
        return nan_result

    normed = (segment - baseline) / amplitude
    peak_rel = max(0, min(peak_idx_in_window, normed.size - 1))
    decay_segment = normed[peak_rel:]
    if decay_segment.size < 3:
        return nan_result

    t_decay = np.arange(decay_segment.size, dtype=float) / float(fs)

    # Exponential fit quality
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
        except Exception:
            r2, residual_std = np.nan, np.nan
    else:
        r2, residual_std = np.nan, np.nan

    # Curvature at decay midpoint
    if decay_segment.size >= 5:
        mid_idx = len(decay_segment) // 2
        if 1 < mid_idx < len(decay_segment) - 2:
            second_deriv = (decay_segment[mid_idx + 1] - 2 * decay_segment[mid_idx] +
                            decay_segment[mid_idx - 1]) * (fs ** 2)
            curvature = float(second_deriv)
        else:
            curvature = np.nan
    else:
        curvature = np.nan

    # Biphasic decay
    if decay_segment.size >= 6 and np.count_nonzero(positive) >= 4:
        mid = len(decay_segment) // 2
        decay_first = decay_segment[:mid][decay_segment[:mid] > 1e-6]
        decay_second = decay_segment[mid:][decay_segment[mid:] > 1e-6]
        if len(decay_first) >= 2 and len(decay_second) >= 2:
            try:
                t1 = np.arange(len(decay_first)) / fs
                t2 = np.arange(len(decay_second)) / fs
                slope1, _ = np.polyfit(t1, np.log(decay_first), 1)
                slope2, _ = np.polyfit(t2, np.log(decay_second), 1)
                tau1 = -1.0 / slope1 if slope1 < 0 else np.nan
                tau2 = -1.0 / slope2 if slope2 < 0 else np.nan
                biphasic_score = tau2 / tau1 if np.isfinite(tau1) and np.isfinite(tau2) and tau2 > 0 else 1.0
            except Exception:
                biphasic_score = np.nan
        else:
            biphasic_score = np.nan
    else:
        biphasic_score = np.nan

    return {
        'decay_r2': float(r2),
        'decay_residual_std': float(residual_std),
        'decay_curvature': float(curvature),
        'decay_biphasic_ratio': float(biphasic_score),
    }


def compute_additional_decay_features(
    window: np.ndarray,
    peak_idx_in_window: int,
) -> dict:
    """Decay skew, kurtosis, linearity."""
    segment = np.asarray(window, dtype=float)
    baseline = float(np.nanmin(segment))
    peak_value = float(np.nanmax(segment))
    amplitude = peak_value - baseline

    nan_result = {k: np.nan for k in ['decay_skew', 'decay_kurtosis', 'decay_linearity']}
    if not np.isfinite(amplitude) or amplitude <= 1e-8:
        return nan_result

    normed = (segment - baseline) / amplitude
    peak_rel = max(0, min(peak_idx_in_window, normed.size - 1))
    decay_segment = normed[peak_rel:]
    if decay_segment.size < 3:
        return nan_result

    decay_skew = float(skew(decay_segment))
    decay_kurt = float(kurtosis(decay_segment))

    positive = decay_segment > 1e-6
    if np.count_nonzero(positive) >= 3:
        log_decay = np.log(decay_segment[positive])
        x = np.arange(len(log_decay))
        try:
            coeffs = np.polyfit(x, log_decay, 1)
            line_fit = np.polyval(coeffs, x)
            deviation = np.std(log_decay - line_fit)
            linearity = 1.0 / (1.0 + deviation)
        except Exception:
            linearity = np.nan
    else:
        linearity = np.nan

    return {
        'decay_skew': float(decay_skew),
        'decay_kurtosis': float(decay_kurt),
        'decay_linearity': float(linearity),
    }


# =====================================================================
#  PER-SPIKE FEATURE ASSEMBLY
# =====================================================================


def agg_spike_feats(
    spike_prom: float, parent_prom : float, 
    small_window: np.ndarray,
    trace_range: float = 1.0,
    eps : float = 1e-9
) -> dict:
    """Compute all features for a single detected spike.

    This is the extension point for adding new per-spike features.

    Parameters
    ----------
    spike_prom : float
        Raw spike prominence.
    parent_prom : float
        Raw prominence of the dominant peak in this peak's cluster.
    small_window : np.ndarray
        Window that spans trace from left to right valley.
    trace_range : float
        Peak-to-peak range of the full trace, used for normalisation.
    eps : float
        Small constant to avoid division by zero.
    """
    norm_prom = float(spike_prom / trace_range)
    dom_score = float(spike_prom/(parent_prom + eps))
    norm_mini_prom = float((np.max(small_window) - small_window[0]) / trace_range)
    distance = int(len(small_window))
    return {
        "spike_prom": norm_prom,
        "dominance_score": dom_score,
        "mini_prom": norm_mini_prom,
        "distance": distance,
    }

# =====================================================================
#  ORCHESTRATOR
# =====================================================================


def get_spike_feats(
    smoothed_f: np.ndarray,
    peaks: np.ndarray,
    mode: str = "train",
    roi_idx=None,
) -> tuple[dict[int, dict[str,dict]] | list[dict[str,Any]], list[int|str]]:
    """
    Compute features for every detected spike in a trace.

    Parameters
    ----------
    smoothed_f : np.ndarray
        1-D smoothed fluorescence trace.
    peaks : np.ndarray
        Detected peak indices.
    mode : str
        ``"train"`` returns spike_data dict with windows/labels.
        ``"inference"`` returns a flat features list.
    roi_idx : optional
        ROI identifier used to construct spike keys.

    Returns
    -------
    result : dict[int, dict] or list[dict]
        If *mode* is ``"train"``, a dict mapping each peak index to a
        record with keys ``"features"``, ``"windows"``, and ``"label"``.
        If *mode* is ``"inference"``, a list of per-spike feature dicts.
    spike_keys : list[int | str]
        Identifiers for each spike — plain peak indices when *roi_idx*
        is None, or ``"{roi_idx}_{peak}"`` strings otherwise.
    """
    from utils.label_utils import create_label_dict   # lazy to avoid circular

    trace_range = np.ptp(smoothed_f) if smoothed_f.size > 0 else 1.0
    prominences, left_bases, right_bases = peak_prominences(smoothed_f, peaks)
    widths = peak_widths(smoothed_f, peaks)[0]

    clusters = build_peak_clusters(peaks, widths, width_factor=1.5)
    cluster_parent_prom = np.empty(len(peaks), dtype=float)
    for cl in clusters:
        parent_prom = float(np.max(prominences[cl]))
        cluster_parent_prom[cl] = parent_prom

    num_peaks = len(peaks)
    spike_keys = []
    spike_records = []

    for i, peak in enumerate(peaks):
        large_window_f = smoothed_f[left_bases[i]:right_bases[i]]
        small_window_f, prev_min, next_min = _create_small_window(
            smoothed_f, peaks, i
        )

        features = agg_spike_feats(
            prominences[i], cluster_parent_prom[i], small_window_f,
            trace_range=trace_range,
        )

        spike_key = peak if roi_idx is None else f"{roi_idx}_{peak}"
        spike_keys.append(spike_key)
        spike_records.append({
            "features": features,
            "windows": {
                'large_window': {
                    'window_values': large_window_f,
                    'bounds': (int(left_bases[i]), int(right_bases[i])),
                },
                'small_window': {
                    'window_values': small_window_f,
                    'bounds': (prev_min, next_min),
                },
            },
            "label": create_label_dict(-1, 'unlabeled'),
        })

    if mode == "inference":
        return [r["features"] for r in spike_records], spike_keys

    spike_data = {
        peak: record
        for peak, record in zip(peaks, spike_records)
    }
    return spike_data, spike_keys


# =====================================================================
#  SPIKE FEATURE EXTRACTION (inference entry point)
# =====================================================================


def _min_peak_distance_frames(fs: float = 15.0) -> int:
    """Convert a frame rate to a minimum inter-peak distance in frames.

    Inversely proportional to ``fs`` so the minimum refractory period in
    *seconds* stays constant.  Anchored at ``20 frames @ 30 Hz`` (\u2248 0.67 s).
    """
    return max(3, int(round(20 * fs / 15)))


def describe_spikes(
    smoothed_f: np.ndarray,
    roi_idx=None,
    mode: str = "inference",
    fs: float = 15.0,
) -> Tuple[Any, list, np.ndarray]:
    """Detect peaks in a trace and compute per-peak features.

    Unified entry point for both training and inference pipelines.

    Parameters
    ----------
    smoothed_f : ndarray
        1-D smoothed fluorescence trace.
    roi_idx : int or None
        ROI index (passed through to feature keys).
    mode : {'inference', 'train'}
        ``"inference"`` returns a flat list of feature dicts.
        ``"train"`` returns a dict keyed by peak index with
        windows / features / labels.
    fs : float
        Frame rate in Hz.

    Returns
    -------
    (result, spike_keys, peaks)
        *result* shape depends on *mode*.  *spike_keys* is a list of
        identifiers.  *peaks* is the ndarray of detected peak indices.
    """
    x = np.asarray(smoothed_f, dtype=float)
    if x.ndim != 1 or x.size < 3 or not np.isfinite(x).all():
        empty = np.asarray([], dtype=int)
        return ([] if mode == "inference" else {}), [], empty

    dist = _min_peak_distance_frames(fs)
    peaks, _ = find_peaks(x, distance=dist)
    peaks = np.asarray(peaks, dtype=int)

    if peaks.size == 0:
        return ([] if mode == "inference" else {}), [], peaks

    result, spike_keys = get_spike_feats(
        x, peaks, mode=mode, roi_idx=roi_idx,
    )
    if mode == "inference":
        result = list(result or [])
    return result, spike_keys, peaks
