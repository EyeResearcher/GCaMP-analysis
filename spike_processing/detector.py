from __future__ import annotations

from dataclasses import dataclass
from turtle import distance
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.signal import find_peaks

# Your existing feature extractor used for spike classifier inference
from spike_classifier.prepare_data import define_candidate_fluor_events as detect_spikes


@dataclass
class SpikeDetector:
    """
    Detect candidate peaks and compute per-peak feature dicts (for spike classifier).
    """

    def get_feats(
        self,
        sm_norm_f: np.ndarray,
        roi_idx: int,
        dist: int = 20,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Args:
            sm_norm_f: Smoothed, normalized trace for a single ROI/neuron (1D).
            roi_idx: Original ROI index (used by your feature code).

        Returns:
            features_list: list of dicts (one per peak) for classifier input
            peaks: np.ndarray of peak indices (same ordering as features_list)
        """
        x = np.asarray(sm_norm_f, dtype=float)
        if x.ndim != 1 or x.size < 3 or not np.isfinite(x).all():
            return [], np.asarray([], dtype=int)

        peaks, _ = find_peaks(x, distance=dist)
        peaks = np.asarray(peaks, dtype=int)
        if peaks.size == 0:
            return [], peaks

        feats_list, _keys = detect_spikes(x, peaks, roi_idx=roi_idx, mode="inference")
        feats_list = list(feats_list or [])
        return feats_list, peaks
