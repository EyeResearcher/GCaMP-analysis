"""Spike kinetics: window creation, transient normalization, rise/decay computation,
decay estimators, and per-spike kinetics interface (SpikeKinetics).

Decay estimators (formerly decay_estimators.py) are included here, eliminating
the circular import between the two modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple
import numpy as np
from scipy.optimize import curve_fit

def _create_small_window(
    trace: np.ndarray,
    peaks: np.ndarray,
    i: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Create the small window (valley to valley) around a spike peak.

    Parameters
    ----------
    trace : np.ndarray
        1-D array of fluorescence values.
    peaks : np.ndarray
        1-D array of all detected peak indices.
    i : int
        Index into *peaks* for the current spike.

    Returns
    -------
    small_window : np.ndarray
        1-D array of fluorescence values in the small window.
    prev_min_idx : int
        Index of the valley before the peak.
    next_min_idx : int
        Index of the valley after the peak.

    Raises
    ------
    ValueError
        If the computed window is invalid (e.g. next_min <= prev_min).
    """
    peak_idx = int(peaks[i])
    prev_peak_idx = int(peaks[i - 1]) if i > 0 else 0
    next_peak_idx = int(peaks[i + 1]) if i < len(peaks) - 1 else len(trace)

    prev_min = prev_peak_idx + int(np.argmin(trace[prev_peak_idx : peak_idx])) if prev_peak_idx >= 0 else 0
    next_min = peak_idx + int(np.argmin(trace[peak_idx : next_peak_idx])) if next_peak_idx <= trace.size else trace.size - 1

    if next_min <= prev_min:
        raise ValueError(f"Invalid peak indices: prev_min={prev_min}, next_min={next_min} for peak_idx={peak_idx}")

    small_window = trace[prev_min:next_min]

    return small_window, int(prev_min), int(next_min)


@dataclass(frozen=True)
class TransientNormalization:
    baseline: float
    peak_value: float
    amplitude: float
    normed: np.ndarray
    peak_rel: int


def _validate_segment(segment: np.ndarray) -> bool:
    return segment.size >= 3 and np.isfinite(segment).all()


def normalize_transient(
    window: np.ndarray,
    peak_idx_in_window: int,
    *,
    baseline: Optional[float] = None,
    peak_value: Optional[float] = None,
    eps: float = 1e-8,
) -> Optional[TransientNormalization]:
    """
    Normalize a spike window to [0, 1] using baseline and peak.
    baseline defaults to min(window); peak_value defaults to max(window).

    Returns None if invalid (too short, non-finite, or ~zero amplitude).
    """
    segment = np.asarray(window, dtype=float)
    if not _validate_segment(segment):
        return None

    b = float(np.nanmin(segment) if baseline is None else baseline)
    p = float(np.nanmax(segment) if peak_value is None else peak_value)
    amp = p - b

    if not np.isfinite(amp) or amp <= eps:
        return None

    normed = (segment - b) / amp
    peak_rel = int(np.clip(int(peak_idx_in_window), 0, normed.size - 1))

    return TransientNormalization(
        baseline=b, peak_value=p, amplitude=amp, normed=normed, peak_rel=peak_rel
    )

def compute_rise_slope_linear(
    normed: np.ndarray,
    peak_rel: int,
    fs: float,
) -> float:
    """
    Linear regression slope of normalized rise from window start to peak.

    Units: normalized units / second.
    """
    rise_segment = normed[: peak_rel + 1]
    if rise_segment.size < 2 or not np.isfinite(rise_segment).all():
        return np.nan

    t = np.arange(rise_segment.size, dtype=float) / float(fs)
    try:
        slope, _ = np.polyfit(t, rise_segment, 1)
        return float(slope)
    except Exception:
        return np.nan
    
def compute_rise_slope(
    window: np.ndarray,
    peak_idx_in_window: int,
    fs: float = 15.0,
) -> float:
    """Rise slope of a spike transient (normalized units / second).

    Normalizes the window to [0, 1] and fits a linear regression on the
    rise phase (window start → peak).
    """
    norm = normalize_transient(window, peak_idx_in_window)
    if norm is None:
        return np.nan
    return compute_rise_slope_linear(norm.normed, norm.peak_rel, fs=fs)


def compute_decay_tau(
    window: np.ndarray,
    peak_idx_in_window: int,
    fs: float = 30.0,
) -> float:
    """Model-free decay tau: time from peak to 1/e of amplitude.

    Normalizes the window to [0, 1] and finds the first post-peak
    crossing of the 1/e (approx 0.368) threshold via linear interpolation.
    """
    norm = normalize_transient(window, peak_idx_in_window)
    if norm is None:
        return np.nan

    decay_seg = norm.normed[norm.peak_rel:]
    if decay_seg.size < 2:
        return np.nan

    threshold = np.exp(-1.0)  # ≈ 0.3679
    hits = np.where(decay_seg <= threshold)[0]
    if hits.size == 0:
        return np.nan

    i1 = int(hits[0])
    if i1 == 0:
        return 0.0

    i0 = i1 - 1
    y0, y1 = float(decay_seg[i0]), float(decay_seg[i1])
    t0, t1 = i0 / float(fs), i1 / float(fs)

    if np.isfinite(y0) and np.isfinite(y1) and y1 != y0:
        frac = np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0)
        return float(t0 + frac * (t1 - t0))
    return t1

def half_max_width(window: np.ndarray, peak_idx_in_window: int, fs: float = 30.0) -> float:
    """Half-maximum width of a spike transient, in seconds.

    The half-max level is defined as ``baseline + amplitude / 2`` where
    *baseline* is the window minimum and *amplitude* is ``peak - baseline``.
    Linear interpolation is used at the crossing points.
    """
    segment = np.asarray(window, dtype=float)
    if segment.size < 3 or not np.isfinite(segment).all():
        return np.nan

    baseline = float(np.min(segment))
    peak_value = float(np.max(segment))
    amplitude = peak_value - baseline
    if amplitude <= 1e-8:
        return np.nan

    half_level = baseline + amplitude / 2.0
    peak_idx = int(np.clip(int(peak_idx_in_window), 0, segment.size - 1))

    # --- left crossing ---
    left_time = np.nan
    for j in range(peak_idx, 0, -1):
        if segment[j - 1] <= half_level:
            denom = segment[j] - segment[j - 1]
            if abs(denom) > 1e-12:
                frac = (half_level - segment[j - 1]) / denom
                left_time = (j - 1 + frac) / float(fs)
            else:
                left_time = (j - 1) / float(fs)
            break

    # --- right crossing ---
    right_time = np.nan
    for j in range(peak_idx, segment.size - 1):
        if segment[j + 1] <= half_level:
            denom = segment[j] - segment[j + 1]
            if abs(denom) > 1e-12:
                frac = (half_level - segment[j + 1]) / denom
                right_time = (j + 1 - frac) / float(fs)
            else:
                right_time = (j + 1) / float(fs)
            break

    if np.isfinite(left_time) and np.isfinite(right_time):
        return float(right_time - left_time)
    return np.nan


@dataclass
class SpikeKinetics:
    """Compute per-spike kinetics (rise slope, decay tau, half-max width)
    for a single spike window."""

    fs: float = 15.0

    def compute(self, window: np.ndarray) -> Dict[str, float]:
        segment = np.asarray(window, dtype=float)
        if segment.size < 3 or not np.isfinite(segment).all():
            return {"rise_slope": np.nan, "decay_tau": np.nan, "half_max_width": np.nan}

        peak_idx = int(np.argmax(segment))

        rise = compute_rise_slope(segment, peak_idx, fs=float(self.fs))
        tau = compute_decay_tau(segment, peak_idx, fs=float(self.fs))
        hmw = half_max_width(segment, peak_idx, fs=float(self.fs))

        return {
            "rise_slope": float(rise) if np.isfinite(rise) else np.nan,
            "decay_tau": float(tau) if np.isfinite(tau) else np.nan,
            "half_max_width": float(hmw) if np.isfinite(hmw) else np.nan,
        }
