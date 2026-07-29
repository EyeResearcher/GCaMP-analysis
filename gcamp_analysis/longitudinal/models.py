"""Data models shared by longitudinal registration and reporting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class RecordingRef:
    """One treatment/region/day recording discovered on disk."""

    treatment: str
    region: str
    day: int
    recording_name: str
    video_dir: Path
    plane0_dir: Path
    metrics_path: Path


@dataclass(frozen=True)
class RegistrationResult:
    """Integer translation that maps a moving day into anchor coordinates."""

    shift_y: int
    shift_x: int
    correlation: float
    method: str = "phase_correlation"


@dataclass(frozen=True)
class CellMatch:
    """One accepted anchor-ROI to moving-day ROI assignment."""

    anchor_roi: int
    moving_roi: int
    score: float
    iou: float
    centroid_distance: float
    ambiguous: bool
