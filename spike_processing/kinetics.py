from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from .decay_estimators import DecayEstimator, ExpOffsetDecayEstimator, LegacyTimeTo1eDecayEstimator
from utils.feature_utils import compute_spike_constants as compute_spike_constants_legacy


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
        rise_slope, _tau_unused = compute_spike_constants_legacy(segment, peak_idx, fs=float(self.fs))
        tau, _diag = self.decay.estimate(segment, peak_idx, fs=float(self.fs))

        hmw = half_max_width_legacy(segment, peak_idx, fs=float(self.fs))

        return {
            "rise_slope": float(rise_slope) if np.isfinite(rise_slope) else np.nan,
            "decay_tau": float(tau) if np.isfinite(tau) else np.nan,
            "half_max_width": float(hmw) if np.isfinite(hmw) else np.nan,
        }
