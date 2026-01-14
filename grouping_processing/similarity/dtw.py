from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from data_classes.neuron import Neuron

from pipeline.neuron_grouping import compute_dtw_matrix  # re-use existing for now

@dataclass
class DTWSimilarity:
    """Produces distance matrix (>=0). May return None if skipped."""
    downsample_factor: int = 3
    use_gpu: bool = True

    def compute(self, neurons: List["Neuron"]) -> Optional[np.ndarray]:
        return compute_dtw_matrix(neurons, downsample_factor=int(self.downsample_factor), use_gpu=bool(self.use_gpu))
