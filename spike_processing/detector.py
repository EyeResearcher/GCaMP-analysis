from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.signal import find_peaks

from .features import get_all_spike_features


# ---------------------------------------------------------------------------
# Core spike detection
# ---------------------------------------------------------------------------

def define_candidate_fluor_events(
    smoothed_f: np.ndarray = None,
    peaks: np.ndarray | None = None,
    roi_idx=None,
    mode="train",
) -> Tuple[Dict[int, Dict], list[int | str]]:
    """Detect spikes and compute windows/features per spike."""
    peaks, props = find_peaks(smoothed_f, distance=30) if peaks is None else (peaks, None)
    if peaks.size == 0:
        return {}, []

    spike_data, spike_keys = get_all_spike_features(
        smoothed_f, peaks, props, mode=mode, roi_idx=roi_idx
    )
    return spike_data, spike_keys


# ---------------------------------------------------------------------------
# SpikeDetector (inference entry point)
# ---------------------------------------------------------------------------

@dataclass
class SpikeDetector:
    """Detect candidate peaks and compute per-peak feature dicts."""

    def get_feats(
        self,
        sm_norm_f: np.ndarray,
        roi_idx: int,
        dist: int = 20,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        x = np.asarray(sm_norm_f, dtype=float)
        if x.ndim != 1 or x.size < 3 or not np.isfinite(x).all():
            return [], np.asarray([], dtype=int)

        peaks, _ = find_peaks(x, distance=dist)
        peaks = np.asarray(peaks, dtype=int)
        if peaks.size == 0:
            return [], peaks

        feats_list, _keys = define_candidate_fluor_events(
            x, peaks, roi_idx=roi_idx, mode="inference"
        )
        feats_list = list(feats_list or [])
        return feats_list, peaks