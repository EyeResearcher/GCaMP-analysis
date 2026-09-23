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
    bundle_path: Path | None = None


@dataclass(frozen=True)
class RegistrationResult:
    """Integer translation that maps a moving day into anchor coordinates."""

    shift_y: int
    shift_x: int
    correlation: float
    method: str = "phase_correlation"


@dataclass(frozen=True)
class CellMatch:
    """One accepted reference-ROI to moving-ROI assignment within a day pair."""

    anchor_roi: int
    moving_roi: int
    score: float
    iou: float
    centroid_distance: float
    ambiguous: bool


@dataclass(frozen=True)
class TracedCellMatch:
    """An anchor ROI traced through zero or more adjacent day-pair matches."""

    anchor_roi: int
    day_roi: int
    edge_count: int
    minimum_score: float
    minimum_iou: float
    maximum_centroid_distance: float
    has_ambiguous_edge: bool
