from __future__ import annotations
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from data_classes.neuron import Neuron

from pipeline.neuron_grouping import compute_sttc_matrix  # re-use existing for now

@dataclass
class STTCSimilarity:
    """Produces correlation matrix in [-1, 1]."""
    time_window: float = 0.033
    fs: float = 30.0

    def compute(self, neurons: List["Neuron"], n_frames: int) -> np.ndarray:
        fs = float(neurons[0].fs) if neurons else float(self.fs)
        return compute_sttc_matrix(neurons, n_frames, time_window=float(self.time_window), fs=fs)
