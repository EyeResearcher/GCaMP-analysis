from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.signal import peak_prominences

from data_classes.spike import Spike
from utils.feature_utils import _create_small_window


@dataclass
class SpikeFactory:
    """
    Instantiate Spike objects and attach windows/bases needed for downstream kinetics.

    Notes:
    - Uses peak_prominences on sm_norm_f to get left/right bases + prominence.
    - Uses _create_small_window on sg_norm_f for the kinetics window.
    """

    def instantiate_spikes(
        self,
        *,
        sm_norm_f: np.ndarray,
        sg_norm_f: np.ndarray,
        peaks_filtered: List[int],
    ) -> List[Spike]:
        if not peaks_filtered:
            return []

        sm = np.asarray(sm_norm_f, dtype=float).reshape(-1)
        sg = np.asarray(sg_norm_f, dtype=float).reshape(-1)
        peaks = np.asarray(peaks_filtered, dtype=int).reshape(-1)

        if sm.size < 3 or sg.size < 3 or peaks.size == 0:
            return []

        prominences, left_bases, right_bases = peak_prominences(sm, peaks)

        spikes: List[Spike] = []
        for i, peak_idx in enumerate(peaks.tolist()):
            sp = Spike(sm_f_idx=int(peak_idx), position_idx=int(i))

            sp.left_base = int(left_bases[i])
            sp.right_base = int(right_bases[i])
            sp.prominence = float(prominences[i])

            # neighbor indices (in trace coordinates)
            sp.prev_position_idx = int(peaks[i - 1]) if i > 0 else 0
            sp.next_position_idx = int(peaks[i + 1]) if i < (peaks.size - 1) else int(sm.size - 1)

            # small window around spike for kinetics/statistics (on sg_norm_f)
            small_window, abs_prev_min, abs_next_min = _create_small_window(
                sg,
                sp.sm_f_idx,
                sp.prev_position_idx,
                sp.next_position_idx,
            )
            sp.f_small_window_sg = small_window

            # cached peak value in sg trace
            if 0 <= sp.sm_f_idx < sg.size:
                sp.f_value = float(sg[sp.sm_f_idx])

            spikes.append(sp)

        return spikes
