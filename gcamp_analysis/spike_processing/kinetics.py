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
    """Baseline-to-peak normalization of one transient window."""

    baseline: float
    peak_value: float
    amplitude: float
    normed: np.ndarray
    peak_rel: int


def _validate_segment(segment: np.ndarray) -> bool:
    """Check that a trace segment is long enough and contains only finite values.

    A segment is considered valid when it has at least 3 samples (the minimum
    needed to define a rise phase, peak, and decay phase) and every element
    is finite (no NaN or Inf).

    Parameters
    ----------
    segment : np.ndarray
        1-D array of fluorescence values to validate.

    Returns
    -------
    bool
        True if the segment has >= 3 elements and all values are finite,
        False otherwise.
    """
    return segment.size >= 3 and np.isfinite(segment).all()


def normalize_transient(
    window: np.ndarray,
    peak_idx_in_window: int,
    *,
    baseline: Optional[float] = None,
    peak_value: Optional[float] = None,
    eps: float = 1e-8,
) -> Optional[TransientNormalization]:
    """Normalize a spike transient window to the range [0, 1].

    The normalization is computed as::

        normed = (window - baseline) / (peak_value - baseline)

    where *baseline* defaults to ``min(window)`` and *peak_value* defaults to
    ``max(window)`` when not explicitly provided.  The function returns
    ``None`` when the segment is too short (< 3 samples), contains
    non-finite values, or has an amplitude (``peak_value - baseline``)
    that is effectively zero (below *eps*).

    Parameters
    ----------
    window : np.ndarray
        1-D array of fluorescence values for the spike transient.
    peak_idx_in_window : int
        Index of the peak within *window*. Clamped to valid bounds
        internally.
    baseline : float, optional
        Explicit baseline value. If None, the minimum of *window* is used.
    peak_value : float, optional
        Explicit peak value. If None, the maximum of *window* is used.
    eps : float, optional
        Minimum amplitude threshold; amplitudes <= *eps* are treated as
        invalid. Default is 1e-8.

    Returns
    -------
    TransientNormalization or None
        A frozen dataclass containing ``baseline``, ``peak_value``,
        ``amplitude``, the normalized trace ``normed``, and the relative
        peak index ``peak_rel``.  Returns None on invalid input.
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
    """Compute the rise slope of a normalized transient via linear regression.

    Extracts the rise segment ``normed[0 : peak_rel + 1]`` and fits a
    first-degree polynomial (``numpy.polyfit``) against a time axis derived
    from the sampling rate *fs*.  The returned slope has units of
    *normalized fluorescence units per second*.

    Returns ``np.nan`` if the rise segment has fewer than 2 samples or
    contains non-finite values.

    Parameters
    ----------
    normed : np.ndarray
        1-D normalized (0–1) fluorescence trace for the transient window.
    peak_rel : int
        Index of the peak within *normed*.
    fs : float
        Sampling rate in Hz, used to convert sample indices to seconds.

    Returns
    -------
    float
        Slope of the linear fit to the rise phase (normalized units / s),
        or ``np.nan`` if the fit cannot be computed.
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
    """Compute the rise slope of a raw spike transient.

    This is a convenience wrapper that first normalizes *window* to [0, 1]
    via `normalize_transient` and then delegates to
    `compute_rise_slope_linear` to perform a linear regression on the rise
    phase (from window start to peak).  The result is expressed in
    *normalized fluorescence units per second*.

    Parameters
    ----------
    window : np.ndarray
        1-D array of raw fluorescence values for the spike transient.
    peak_idx_in_window : int
        Index of the peak within *window*.
    fs : float, optional
        Sampling rate in Hz. Default is 15.0.

    Returns
    -------
    float
        Rise slope in normalized units / s, or ``np.nan`` if the window
        cannot be normalized or the fit fails.
    """
    norm = normalize_transient(window, peak_idx_in_window)
    if norm is None:
        return np.nan
    return compute_rise_slope_linear(norm.normed, norm.peak_rel, fs=fs)


def compute_decay_tau(
    window: np.ndarray,
    peak_idx_in_window: int,
    fs: float = 15.0,
) -> float:
    """Estimate a model-free decay time constant (tau) for a spike transient.

    The window is first normalized to [0, 1] via `normalize_transient`.
    The decay segment (peak onward) is then searched for the first sample
    that falls at or below ``1/e ≈ 0.3679`` of the peak amplitude.  Linear
    interpolation between the two flanking samples is used to refine the
    crossing time.  The returned value is the elapsed time (in seconds)
    from the peak to this crossing point, analogous to the time constant
    of an exponential decay without fitting an explicit exponential model.

    Returns ``np.nan`` if the transient cannot be normalized or the decay
    never crosses the 1/e threshold within the window.

    Parameters
    ----------
    window : np.ndarray
        1-D array of raw fluorescence values for the spike transient.
    peak_idx_in_window : int
        Index of the peak within *window*.
    fs : float, optional
        Sampling rate in Hz. Default is 15.0.

    Returns
    -------
    float
        Decay tau in seconds, or ``np.nan`` if it cannot be determined.
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

def half_max_width(window: np.ndarray, peak_idx_in_window: int, fs: float = 15.0) -> float:
    """Compute the full width at half maximum (FWHM) of a spike transient.

    The half-maximum level is calculated as::

        half_level = baseline + amplitude / 2

    where ``baseline = min(window)`` and ``amplitude = max(window) - baseline``.
    Starting from the peak, the function walks leftward to find the first
    crossing below *half_level* (left crossing) and rightward for the
    corresponding right crossing.  Linear interpolation between the two
    flanking samples refines each crossing time.  The width is the
    difference ``right_time - left_time``, expressed in seconds.

    Returns ``np.nan`` if the window is too short (< 3 samples), contains
    non-finite values, has near-zero amplitude, or if either crossing
    cannot be found.

    Parameters
    ----------
    window : np.ndarray
        1-D array of raw fluorescence values for the spike transient.
    peak_idx_in_window : int
        Index of the peak within *window*.
    fs : float, optional
        Sampling rate in Hz. Default is 30.0.

    Returns
    -------
    float
        Full width at half maximum in seconds, or ``np.nan`` if it cannot
        be determined.
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
    """Callable kinetics calculator for individual spike transients.

    Wraps `compute_rise_slope`, `compute_decay_tau`, and `half_max_width`
    behind a single `compute` method that returns all three metrics in a
    dictionary.  The peak is located automatically as the argmax of the
    input window.

    Parameters
    ----------
    fs : float, optional
        Sampling rate in Hz, forwarded to all kinetics functions.
        Default is 15.0.
    """

    fs: float = 15.0

    def compute(self, window: np.ndarray) -> Dict[str, float]:
        """Compute rise slope, decay tau, and half-max width for a spike.

        The peak index is determined automatically via ``np.argmax``.
        Each metric is computed independently; if one fails (e.g., the
        decay never crosses the 1/e threshold), it is returned as
        ``np.nan`` while the others are still reported.

        Parameters
        ----------
        window : np.ndarray
            1-D array of fluorescence values for a single spike
            transient.

        Returns
        -------
        dict of {str: float}
            Dictionary with keys ``'rise_slope'`` (normalized units / s),
            ``'decay_tau'`` (s), and ``'half_max_width'`` (s).  Any
            metric that cannot be computed is set to ``np.nan``.
        """
        segment = np.asarray(window, dtype=float)
        if segment.size < 3 or not np.isfinite(segment).all():
            return {"rise_slope": np.nan, "decay_tau": np.nan, "half_max_width": np.nan}

        peak_idx = int(np.argmax(segment))

        rise = compute_rise_slope(segment, peak_idx, fs=float(self.fs))
        tau = compute_decay_tau(segment, peak_idx, fs=float(self.fs))
        hmw = half_max_width(segment, peak_idx, fs=float(self.fs))

        return {
            "rise_slope_hz": float(rise) if np.isfinite(rise) else np.nan,
            "decay_tau_seconds": float(tau) if np.isfinite(tau) else np.nan,
            "half_max_width_seconds": float(hmw) if np.isfinite(hmw) else np.nan,
        }
