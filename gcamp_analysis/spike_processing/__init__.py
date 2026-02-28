"""
Spike processing sub-pipeline.

Three modules mirroring ``roi_processing``:
  - features.py   — per-spike feature computation
  - kinetics.py   — rise/decay time-constants, decay estimators
  - filtering.py  — detection, classification filtering, spike instantiation, orchestration
"""

from .features import get_spike_feats, describe_spikes
from .filtering import (
    SpikeFilter,
    SpikeService,
)
from .kinetics import SpikeKinetics

__all__ = [
    "get_spike_feats",
    "describe_spikes",
    "SpikeFilter",
    "SpikeKinetics",
    "SpikeService",
]