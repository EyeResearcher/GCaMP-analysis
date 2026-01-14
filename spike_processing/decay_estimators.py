from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Tuple
import numpy as np

# Keep current behavior for now by using your existing implementation
from archive.utils.spike_utils import compute_spike_constants as compute_spike_constants_legacy


class DecayEstimator(Protocol):
    """
    Strategy interface for decay constant estimation.
    Must return (tau_seconds, diagnostics_dict).
    """
    name: str

    def estimate(self, window: np.ndarray, peak_idx_in_window: int, fs: float) -> Tuple[float, Dict[str, float]]:
        ...


@dataclass
class LegacyTimeTo1eDecayEstimator:
    """
    Current behavior: time to 1/e of normalized peak with fallback log-linear.
    Uses archive.utils.spike_utils.compute_spike_constants to preserve behavior.
    """
    name: str = "legacy_time_to_1e"

    def estimate(self, window: np.ndarray, peak_idx_in_window: int, fs: float) -> Tuple[float, Dict[str, float]]:
        rise_slope, tau = compute_spike_constants_legacy(window, peak_idx_in_window, fs=fs)
        return float(tau), {}


# Placeholder for later (you’ll implement the bounded exp+offset fit here)
@dataclass
class ExpOffsetDecayEstimator:
    name: str = "exp_offset"

    def estimate(self, window: np.ndarray, peak_idx_in_window: int, fs: float) -> Tuple[float, Dict[str, float]]:
        # Implement later (SciPy curve_fit with bounds).
        return np.nan, {}
