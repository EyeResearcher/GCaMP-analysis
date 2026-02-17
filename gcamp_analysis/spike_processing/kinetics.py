"""
Spike kinetics: window creation, transient normalization, rise/decay computation,
and per-spike kinetics interface (SpikeKinetics).

Formerly split between utils/feature_utils.py and spike_processing/kinetics.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from .decay_estimators import DecayEstimator, ExpOffsetDecayEstimator, LegacyTimeTo1eDecayEstimator


# ---------------------------------------------------------------------------
# Low-level helpers (window creation)
# ---------------------------------------------------------------------------

def _compute_min_between(
    trace: np.ndarray, start: int, end: int
) -> int:
    """Return index of minimum value between start and end (exclusive of end)."""
    if start >= end:
        return start
    local_min = int(np.argmin(trace[start:end]))
    return start + local_min


def _create_large_window(
    valid_spike_prob: np.ndarray,
    peak_idx: int,
    left_base_idx: int,
    right_base_idx: int,
    start_idx: int = 0
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
    large_window = valid_spike_prob[left_base_idx:right_base_idx]
    absolute_left_base = int(left_base_idx + start_idx)
    absolute_right_base = int(right_base_idx + start_idx)
    spike_prom = valid_spike_prob[peak_idx] - valid_spike_prob[left_base_idx]
    return large_window, absolute_left_base, absolute_right_base, float(spike_prom)


def _create_small_window(
    valid_spike_prob: np.ndarray,
    peak_idx: int,
    prev_peak_idx: int,
    next_peak_idx: int,
    start_idx: int = 0
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
    prev_min = _compute_min_between(valid_spike_prob, prev_peak_idx, peak_idx)
    next_min = _compute_min_between(valid_spike_prob, peak_idx, next_peak_idx)

    if next_min <= prev_min:
        next_min = prev_min + 1 if prev_min + 1 < len(valid_spike_prob) else len(valid_spike_prob)

    small_window = valid_spike_prob[prev_min:next_min]
    absolute_prev_min = int(prev_min + start_idx)
    absolute_next_min = int(next_min + start_idx)
    return small_window, absolute_prev_min, absolute_next_min


# ---------------------------------------------------------------------------
# Transient normalization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Rise / decay computations
# ---------------------------------------------------------------------------

def compute_rise_slope_linear(
    normed: np.ndarray,
    peak_rel: int,
    *,
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


def time_to_reach_fraction_of_peak(
    normed: np.ndarray,
    peak_rel: int,
    *,
    fs: float,
    fraction: float = np.exp(-1.0),
    direction: str = "decay",
    interpolation: str = "linear",
) -> float:
    """
    Model-free time-to-threshold relative to peak, measured from the peak time.

    For decay: finds first time AFTER peak where normed <= fraction.
    For rise:  finds last time BEFORE peak where normed <= fraction.

    Returns time in seconds from the peak to the crossing point.
    """
    if normed.size < 2 or not np.isfinite(normed).all():
        return np.nan

    frac = float(fraction)
    if not (0.0 < frac < 1.0):
        raise ValueError("fraction must be between 0 and 1 (exclusive).")

    if direction not in {"decay", "rise"}:
        raise ValueError("direction must be 'decay' or 'rise'.")

    if direction == "decay":
        seg = normed[peak_rel:]
        if seg.size < 2:
            return np.nan

        hits = np.where(seg <= frac)[0]
        if hits.size == 0:
            return np.nan

        i1 = int(hits[0])
        if i1 == 0:
            return 0.0

        i0 = i1 - 1
        y0, y1 = float(seg[i0]), float(seg[i1])
        t0, t1 = i0 / float(fs), i1 / float(fs)

        if interpolation == "linear" and np.isfinite(y0) and np.isfinite(y1) and y1 != y0:
            a = (frac - y0) / (y1 - y0)
            a = float(np.clip(a, 0.0, 1.0))
            return t0 + a * (t1 - t0)

        return t1

    # direction == "rise"
    seg = normed[: peak_rel + 1]
    if seg.size < 2:
        return np.nan
    hits = np.where(seg <= frac)[0]
    if hits.size == 0:
        return np.nan
    i = int(hits[-1])
    return (peak_rel - i) / float(fs)


def compute_decay_tau_time_to_37(
    normed: np.ndarray,
    peak_rel: int,
    *,
    fs: float,
) -> float:
    """
    Decay tau proxy: time from peak to reach 1/e (~37%) of peak above baseline.
    Returns NaN if 37% is never reached within the window.
    """
    return time_to_reach_fraction_of_peak(
        normed, peak_rel, fs=fs, fraction=np.exp(-1.0), direction="decay"
    )


def compute_spike_constants(
    window: np.ndarray,
    peak_idx_in_window: int,
    fs: float = 30.0,
    *,
    compute_rise: bool = True,
    compute_decay: bool = True,
) -> Tuple[float, float]:
    """
    Modular wrapper:
      - rise_slope: linear slope on normalized rise (optional)
      - decay_tau:  time to reach 37% of peak (no fitting)

    Returns (rise_slope, decay_tau).
    """
    norm = normalize_transient(window, peak_idx_in_window)
    if norm is None:
        return np.nan, np.nan

    rise_slope = (
        compute_rise_slope_linear(norm.normed, norm.peak_rel, fs=fs)
        if compute_rise
        else np.nan
    )

    decay_tau = (
        compute_decay_tau_time_to_37(norm.normed, norm.peak_rel, fs=fs)
        if compute_decay
        else np.nan
    )

    return rise_slope, decay_tau


# ---------------------------------------------------------------------------
# High-level per-spike kinetics interface
# ---------------------------------------------------------------------------


def half_max_width_legacy(window: np.ndarray, peak_idx_in_window: int, fs: float = 30.0) -> float:
    """
    Legacy-style half-max width (kept intentionally similar to existing behavior).
    NOTE: This can still produce non-finite values; we will harden it later.
    """
    segment = np.asarray(window, dtype=float)
    if segment.size < 3 or not np.isfinite(segment).all():
        return np.nan

    peak_value = float(np.nanmax(segment))
    half_max = peak_value / 2.0

    peak_idx = int(np.clip(int(peak_idx_in_window), 0, segment.size - 1))

    left_idx = peak_idx
    while left_idx > 0 and segment[left_idx] >= half_max:
        left_idx -= 1

    # interpolate (legacy)
    if left_idx < peak_idx:
        denom = (segment[left_idx + 1] - segment[left_idx])
        left_time = left_idx + (half_max - segment[left_idx]) / denom
    else:
        left_time = left_idx

    right_idx = peak_idx
    while right_idx < segment.size - 1 and segment[right_idx] >= half_max:
        right_idx += 1

    if right_idx > peak_idx:
        denom = (segment[right_idx - 1] - segment[right_idx])
        right_time = right_idx - (half_max - segment[right_idx]) / denom
    else:
        right_time = right_idx

    width_frames = right_time - left_time
    return float(width_frames / float(fs))


@dataclass
class SpikeKinetics:
    """
    Compute per-spike kinetics for a spike window.

    By default this preserves your existing decay_tau behavior via LegacyTimeTo1eDecayEstimator.
    Later you can swap decay=ExpOffsetDecayEstimator(...) without touching service code.
    """

    fs: float = 30.0
    decay: Optional[DecayEstimator] = None

    def __post_init__(self) -> None:
        if self.decay is None:
            self.decay = LegacyTimeTo1eDecayEstimator()

    def compute(self, window: np.ndarray) -> Dict[str, float]:
        segment = np.asarray(window, dtype=float)
        if segment.size < 3 or not np.isfinite(segment).all():
            return {"rise_slope": np.nan, "decay_tau": np.nan, "half_max_width": np.nan}

        peak_idx = int(np.argmax(segment))

        # Rise slope + decay (legacy function returns both; keep for compatibility)
        # We use the estimator for tau to keep the strategy interface consistent.
        rise_slope, _tau_unused = compute_spike_constants(segment, peak_idx, fs=float(self.fs))
        tau, _diag = self.decay.estimate(segment, peak_idx, fs=float(self.fs))

        hmw = half_max_width_legacy(segment, peak_idx, fs=float(self.fs))

        return {
            "rise_slope": float(rise_slope) if np.isfinite(rise_slope) else np.nan,
            "decay_tau": float(tau) if np.isfinite(tau) else np.nan,
            "half_max_width": float(hmw) if np.isfinite(hmw) else np.nan,
        }
