"""Spike-level feature computation.

All spike feature math lives here.  Both the training pipeline
(spike_classifier.prepare_data) and the inference pipeline
(spike_processing.detector → pipeline.services.spike_service)
ultimately call get_all_spike_features from this module.
"""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import peak_prominences, peak_widths

from .kinetics import compute_spike_constants, _create_large_window, _create_small_window
from .hierarchy import compute_peak_hierarchy_features


# ---------------------------------------------------------------------------
# Asymmetry
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Decay shape
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-spike feature assembly
# ---------------------------------------------------------------------------

def _compute_spike_features(
    large_window: np.ndarray,
    small_window: np.ndarray,
    spike_prom: float,
    peak_idx: int,
    left_base_idx: int,
    absolute_prev_min: int,
    hierarchy: dict,
    i: int,
    top3: bool = False,
    trace_range: float = 1.0,
) -> dict:
    """Compute all features for a single detected spike."""
    peak_in_large_window = peak_idx - left_base_idx
    rise_slope, decay_tau = compute_spike_constants(
        small_window, peak_in_large_window, fs=15.0
    )
    decay_shape = compute_decay_shape_features(small_window, peak_in_large_window, fs=15.0)
    additional_decay = compute_additional_decay_features(small_window, peak_in_large_window)

    mini_prom = large_window[peak_in_large_window] - small_window[0]

    if top3:
        return {
            "spike_prom": float(spike_prom) / trace_range,
            "dominance_score": float(hierarchy["dominance_score"][i]),
            "mini_prom": float(mini_prom) / trace_range,
            "distance": int(len(small_window)),
        }

    return {
        "spike_prom": float(spike_prom) / trace_range,
        "isolation": int(len(large_window)),
        "distance": int(len(small_window)),
        "iso_skew": float(skew(large_window)) if large_window.size else 0.0,
        "dist_skew": float(skew(small_window)) if small_window.size else 0.0,
        "iso_aai_sum": float(area_asymmetry(large_window, peak_idx - left_base_idx)),
        "dist_aai_sum": float(area_asymmetry(small_window, peak_idx - absolute_prev_min)),
        "iso_aai_trapz": float(area_asymmetry_trapz(large_window, zero_value=peak_idx - left_base_idx)),
        "dist_aai_trapz": float(area_asymmetry_trapz(small_window, zero_value=peak_idx - absolute_prev_min)),
        "rise_slope": float(rise_slope),
        "decay_tau": float(decay_tau),
        **decay_shape,
        **additional_decay,
        "dominance_score": float(hierarchy["dominance_score"][i]),
        "local_rank": int(hierarchy["local_rank"][i]),
        "local_rank_norm": float(hierarchy["local_rank_norm"][i]),
        "cluster_size": int(hierarchy["cluster_size"][i]),
        "prom_gap": float(hierarchy["prom_gap"][i]),
        "time_to_parent": float(hierarchy["time_to_parent"][i]),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def get_all_spike_features(
    smoothed_f: np.ndarray,
    peaks: np.ndarray,
    props: Optional[dict],
    mode: str = "train",
    roi_idx=None,
) -> tuple:
    """
    Compute features for every detected spike in a trace.

    Parameters
    ----------
    smoothed_f : np.ndarray
        1-D smoothed fluorescence trace.
    peaks : np.ndarray
        Detected peak indices.
    props : dict or None
        Unused (kept for API compat).
    mode : str
        ``"train"`` returns spike_data dict with windows/labels.
        ``"inference"`` returns a flat features list.
    roi_idx : optional
        ROI identifier used to construct spike keys.

    Returns
    -------
    (spike_data, spike_keys) or (features_list, spike_keys)
    """
    from utils.label_utils import create_label_dict   # lazy to avoid circular

    trace_range = np.ptp(smoothed_f) if smoothed_f.size > 0 else 1.0
    prominences, left_bases, right_bases = peak_prominences(smoothed_f, peaks)
    widths = peak_widths(smoothed_f, peaks)[0]
    hierarchy = compute_peak_hierarchy_features(
        peaks=peaks, prominences=prominences, widths=widths, width_factor=1.5,
    )

    num_peaks = len(peaks)
    spike_keys = []
    windows_list = []
    features_list = []
    labels_list = []

    for i, peak in enumerate(peaks):
        large_window_f, absolute_left_base, absolute_right_base, spike_prom = _create_large_window(
            smoothed_f, peak, left_bases[i], right_bases[i]
        )
        prev_peak = peaks[i - 1] if i > 0 else 0
        next_peak = peaks[i + 1] if i < num_peaks - 1 else len(smoothed_f)
        small_window_f, absolute_prev_min, absolute_next_min = _create_small_window(
            smoothed_f, peak, prev_peak, next_peak
        )

        features = _compute_spike_features(
            large_window_f, small_window_f, spike_prom,
            peak, left_bases[i], absolute_prev_min, hierarchy, i,
            top3=True, trace_range=trace_range,
        )

        spike_key = peak if roi_idx is None else f"{roi_idx}_{peak}"
        spike_keys.append(spike_key)
        features_list.append(features)
        labels_list.append(create_label_dict(-1, 'unlabeled'))
        windows_list.append({
            'large_window': {
                'window_values': large_window_f,
                'bounds': (absolute_left_base, absolute_right_base),
            },
            'small_window': {
                'window_values': small_window_f,
                'bounds': (absolute_prev_min, absolute_next_min),
            },
        })

    if mode == "inference":
        return features_list, spike_keys

    spike_data = {
        peak: {"windows": windows, "features": features, "label": label}
        for peak, windows, features, label in zip(peaks, windows_list, features_list, labels_list)
    }
    return spike_data, spike_keys