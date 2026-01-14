"""
Spike processing sub-pipeline.

This package contains spike-specific helpers used by pipeline/services/spike_service.py
(detector -> filter -> factory -> kinetics -> summary).

Nothing here should import from pipeline/services to avoid circular imports.
"""

from .detector import SpikeDetector
from .filter import SpikeFilter
from .factory import SpikeFactory
from .kinetics import SpikeKinetics
from .summary import NeuronSpikeSummary

__all__ = [
    "SpikeDetector",
    "SpikeFilter",
    "SpikeFactory",
    "SpikeKinetics",
    "NeuronSpikeSummary",
]
