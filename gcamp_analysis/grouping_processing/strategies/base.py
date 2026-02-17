
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.video import Video

class GroupingStrategy(Protocol):
    name: str
    def compute(self, video: "Video", config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns dict with at least:
          - groups: list[NeuronGroup]
          - matrix: np.ndarray | None
          - config_label: str
        """
        ...
