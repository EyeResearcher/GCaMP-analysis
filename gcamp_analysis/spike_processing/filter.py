from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class SpikeFilter:
    """
    Convert per-peak predictions (bool/0-1) into a list of retained peak indices.
    """

    def apply(self, peaks: np.ndarray, predictions: np.ndarray) -> List[int]:
        peaks_arr = np.asarray(peaks, dtype=int).reshape(-1)
        preds_arr = np.asarray(predictions).reshape(-1)

        if peaks_arr.size == 0:
            return []

        # Interpret anything > 0.5 as True if not already boolean
        if preds_arr.dtype != bool:
            preds_arr = preds_arr.astype(float) > 0.5
        else:
            preds_arr = preds_arr.astype(bool)

        # Length safety
        if preds_arr.size < peaks_arr.size:
            pad = np.zeros(peaks_arr.size - preds_arr.size, dtype=bool)
            preds_arr = np.concatenate([preds_arr, pad])
        elif preds_arr.size > peaks_arr.size:
            preds_arr = preds_arr[: peaks_arr.size]

        return [int(p) for p, keep in zip(peaks_arr.tolist(), preds_arr.tolist()) if bool(keep)]
