from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.signal import find_peaks

from .features import get_all_spike_features


# ---------------------------------------------------------------------------
# Frame-rate → distance conversion
# ---------------------------------------------------------------------------




def min_peak_distance_frames(fs: float = 15.0) -> int:
    """Convert a frame rate to a minimum inter-peak distance in frames.

    The distance is inversely proportional to ``fs`` so the minimum
    refractory period in *seconds* stays constant across acquisition
    rates.  Anchored at ``20 frames @ 30 Hz`` (≈ 0.67 s).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    int
        Minimum distance in frames (≥ ``_MIN_DIST``).
    """
    return max(3, int(round(20 * fs / 15)))


# ---------------------------------------------------------------------------
# Core spike detection
# ---------------------------------------------------------------------------

def get_f_events(
    smoothed_f: np.ndarray = None,
    peaks: np.ndarray | None = None,
    roi_idx=None,
    mode="train",
    fs: float = 15.0,
) -> Tuple[Dict[int, Dict], list[int | str]]:
    """Detect spikes and compute windows/features per spike.

    Parameters
    ----------
    smoothed_f : ndarray
        Smoothed fluorescence trace.
    peaks : ndarray or None
        Pre-detected peak indices.  If *None*, peaks are detected here
        using ``find_peaks(distance=min_peak_distance_frames(fs))``.
    roi_idx : int or None
        ROI index (used as key prefix in train mode).
    mode : {'train', 'inference'}
        Determines key format for the returned dicts.
    fs : float
        Frame rate in Hz — used to compute ``distance`` when *peaks*
        is *None*.
    """
    distance = min_peak_distance_frames(fs)
    peaks, props = find_peaks(smoothed_f, distance=distance) if peaks is None else (peaks, None)
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
        fs: float = 15.0,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Detect peaks and return per-peak feature dicts.

        Parameters
        ----------
        sm_norm_f : ndarray
            Smoothed, normalised fluorescence trace for one ROI.
        roi_idx : int
            ROI index.
        fs : float
            Frame rate in Hz.
        """
        x = np.asarray(sm_norm_f, dtype=float)
        if x.ndim != 1 or x.size < 3 or not np.isfinite(x).all():
            return [], np.asarray([], dtype=int)

        dist = min_peak_distance_frames(fs)
        peaks, _ = find_peaks(x, distance=dist)
        peaks = np.asarray(peaks, dtype=int)
        if peaks.size == 0:
            return [], peaks

        feats_list, _keys = get_f_events(
            x, peaks, roi_idx=roi_idx, mode="inference", fs=fs,
        )
        feats_list = list(feats_list or [])
        return feats_list, peaks