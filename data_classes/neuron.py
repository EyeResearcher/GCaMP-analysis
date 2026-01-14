

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from .roi import ROI
from .spike import Spike


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from .roi import ROI
from .spike import Spike

"""Neuron class for filtered ROIs."""
from data_classes.spike import Spike
import numpy as np
from typing import List, Optional
from .roi import ROI

@dataclass
class Neuron:
    """
    Validated neuron built from an ROI.

    IMPORTANT:
    - We keep *all* ROI info accessible via delegation to `roi`.
    - We do NOT copy roi.__dict__ into neuron.__dict__.
    """

    roi: ROI
    filtered_index: int
    fs: float = 30.0

    # pipeline-populated
    spikes: List[Spike] = field(default_factory=list)
    spk_features: List[Dict[str, Any]] = field(default_factory=list)

    peaks: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))
    n_peaks_raw: int = 0
    peaks_filtered: List[int] = field(default_factory=list)

    all_spk_stats: List[Dict[str, Any]] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    # Optional debugging payload
    raw_stats: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str):
        """
        Delegate missing attributes to the ROI.
        This preserves the convenience of your old __dict__.update approach,
        while keeping Neuron typed + safer.
        """
        return getattr(self.roi, name)

    def __repr__(self) -> str:
        idx = getattr(self.roi, "index", None)
        return f"Neuron(index={idx}, filtered_index={self.filtered_index}, spikes={len(self.spikes)})"
