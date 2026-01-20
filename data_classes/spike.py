"""Spike class for detected events."""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from utils.feature_utils import (
    _create_large_window,
    _create_small_window
)



from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class Spike:
    """Pure data object for one detected spike event (no compute)."""

    sm_f_idx: int
    position_idx: int

    # Bases from prominence calc
    left_base: Optional[int] = None
    right_base: Optional[int] = None

    # Neighbors (indices in the trace)
    prev_position_idx: Optional[int] = None
    next_position_idx: Optional[int] = None

    # Scalar features
    prominence: float = 0.0
    f_value: Optional[float] = None

    # Windows used for kinetics/stats
    f_small_window_sg: Optional[np.ndarray] = None

    # Outputs
    stats: Dict[str, Any] = field(default_factory=dict)
    is_valid: Optional[bool] = None
