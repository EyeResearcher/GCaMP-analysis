"""Mutable data and pipeline state for one Suite2p video.

``Video`` owns injected Suite2p inputs, path-derived metadata, concatenated
section metadata, and results populated by processing services. Its constructor
does not access the filesystem. Use ``Video.from_suite2p`` when loading a video
from disk, or inject ``suite2p_data`` directly in tests and alternate loaders.

Concatenation CSV discovery and parsing live in
``gcamp_analysis.concatenation.metadata``. Report serialization and figure
writing live in
``gcamp_analysis.reporting``:

* ``VideoStatistics`` creates an immutable snapshot of completed results.
* ``VideoStatisticsWriter`` writes tables, matrices, and section outputs.
* ``VideoFiguresWriter`` generates and saves reporting figures.

The reporting names are re-exported at the bottom of this module for backwards
compatibility. New code should import them from ``gcamp_analysis.reporting``.

When adding pipeline state, add the field to ``Video`` and have the responsible
service populate it. Add it to ``VideoStatistics`` only if it must be included
in reporting or persisted output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any

import numpy as np
import pandas as pd

from gcamp_analysis.concatenation.metadata import (
    ConcatMetadata,
    ConcatSection,
    load_concat_metadata,
    normalize_section_key,
    validate_section_kind,
)
from utils.io_utils import load_suite2p_data

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.roi import ROI
    from gcamp_analysis.data_classes.neuron import Neuron


def _empty_array() -> np.ndarray:
    """Default factory for dataclass array fields."""
    return np.asarray([])


@dataclass
class Video:
    """
    Thin, stateful context object for a single video folder.

    Responsibilities (keep these here):
      - hold paths + metadata
      - hold already-loaded Suite2p data
      - store pipeline outputs for services and reporting snapshots
      - avoid heavy computation methods (those live in services)

    Use ``from_suite2p`` for explicit filesystem loading.
    """

    # ---- Inputs
    path: Path
    suite2p_path: Path  # e.g. <video_dir>/suite2p/plane0
    suite2p_data: dict[str, Any] = field(repr=False)
    is_concatenated: bool = False
    concat_metadata: Optional[ConcatMetadata] = field(
        default=None,
        repr=False,
    )

    # ---- Metadata (derived)
    video_id: str = field(init=False)
    experiment_name: str = field(init=False, default="unknown")
    treatment: str = field(init=False, default="unknown")
    timepoint_name: str = field(init=False, default="unknown")

    n_rois: int = field(init=False, default=0)
    n_frames: int = field(init=False, default=0)
    fs: float = field(init=False, default=15.0)

    # ---- Concatenated-video metadata
    concat_summary_path: Optional[Path] = field(init=False, default=None)
    concat_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    concat_sections: list[ConcatSection] = field(default_factory=list, repr=False)
    sections_dict: dict[str, ConcatSection] = field(default_factory=dict, repr=False)
    section_traces: dict[str, dict[str, np.ndarray]] = field(default_factory=dict, repr=False)

    # ---- Traces (populated by TraceService)
    norm_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    norm_sm_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    norm_sg_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    z_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    savgol_z_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    sm_sp: np.ndarray = field(default_factory=_empty_array, repr=False)  # optional if/when used

    # ---- ROI filtering outputs (populated by ROIService)
    n_good_rois: int = 0
    n_bad_rois: int = 0
    bad_rois: list["ROI"] = field(default_factory=list, repr=False)
    bad_rois_features: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # ---- Neurons (populated by ROIService + SpikeService)
    neurons: list["Neuron"] = field(default_factory=list, repr=False)

    # ---- Spike statistics (populated by SpikeService)
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # ---- Grouping outputs (populated by GroupingService)
    grouping_results: dict = field(default_factory=dict, repr=False)  # {name: GroupingResult}
    grouping_stats: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # ---- Baseline-vs-section comparison outputs (concatenated mode)
    section_comparison_results: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.suite2p_path = Path(self.suite2p_path)
        self.video_id = self.path.name

        F = self.suite2p_data.get("F", None)
        if F is None:
            raise ValueError(f"Suite2p data at {self.suite2p_path} missing key 'F'.")
        self.n_rois, self.n_frames = F.shape
        self.fs = float(self.suite2p_data.get("fs", self.suite2p_data.get("ops", {}).get("fs", 15.0)))

        self._parse_metadata()
        self._apply_concat_metadata()

    @classmethod
    def from_suite2p(
        cls,
        *,
        path: Path,
        suite2p_path: Path,
        is_concatenated: bool = False,
    ) -> Video:
        """Load Suite2p and optional concat metadata, then construct a video."""
        path = Path(path)
        suite2p_path = Path(suite2p_path)
        suite2p_data = load_suite2p_data(suite2p_path)

        fluorescence = suite2p_data.get("F")
        if fluorescence is None:
            raise ValueError(
                f"Suite2p data at {suite2p_path} missing key 'F'."
            )

        concat_metadata = None
        if is_concatenated:
            concat_metadata = load_concat_metadata(
                path,
                n_frames=int(fluorescence.shape[1]),
            )

        return cls(
            path=path,
            suite2p_path=suite2p_path,
            suite2p_data=suite2p_data,
            is_concatenated=is_concatenated,
            concat_metadata=concat_metadata,
        )

    def _parse_metadata(self) -> None:
        """Parse experiment/treatment/timepoint from the folder path."""
        parts = list(self.path.parts)
        if len(parts) >= 4:
            self.timepoint_name = parts[-2]
            self.treatment = parts[-3]
            self.experiment_name = parts[-4]
        else:
            self.timepoint_name = "unknown"
            self.treatment = "unknown"
            self.experiment_name = "unknown"

    def get_section(self, section_name: str) -> Optional[ConcatSection]:
        """Return a section descriptor by normalized section key."""
        return self.sections_dict.get(self._section_key(section_name))

    @property
    def baseline_section(self) -> Optional[ConcatSection]:
        """Return the unique baseline section when present."""
        return self.get_section("baseline")

    def get_sections_by_kind(self, kind: str) -> list[ConcatSection]:
        """Return all sections matching the requested canonical kind."""
        normalized = self._section_kind(kind)
        return [section for section in self.concat_sections if section.section_kind == normalized]

    def iter_nonbaseline_sections(self) -> list[ConcatSection]:
        """Return all non-baseline sections in concat order."""
        return [section for section in self.concat_sections if section.section_kind != "baseline"]

    def clear_results(self) -> None:
        """Reset all computed fields to defaults. Useful for re-running in notebooks."""
        self.norm_f = _empty_array()
        self.norm_sm_f = _empty_array()
        self.norm_sg_f = _empty_array()
        self.z_f = _empty_array()
        self.savgol_z_f = _empty_array()
        self.sm_sp = _empty_array()

        self.section_traces = {}

        self.n_good_rois = 0
        self.n_bad_rois = 0
        self.bad_rois = []
        self.bad_rois_features = pd.DataFrame()

        self.neurons = []
        self.summary_df = pd.DataFrame()

        self.grouping_results = {}
        self.grouping_stats = pd.DataFrame()
        self.section_comparison_results = {}

    def __repr__(self) -> str:
        return (
            f"Video(video_id={self.video_id}, n_rois={self.n_rois}, n_frames={self.n_frames}, "
            f"treatment={self.treatment}, timepoint={self.timepoint_name})"
        )

    def _apply_concat_metadata(self) -> None:
        """Apply already-loaded concat metadata to compatibility fields."""
        if not self.is_concatenated:
            return
        if self.concat_metadata is None:
            raise ValueError(
                "Concatenated videos require concat_metadata. Use "
                "Video.from_suite2p(...) to load it from disk."
            )

        self.concat_summary_path = self.concat_metadata.summary_path
        self.concat_summary_df = self.concat_metadata.summary_df
        self.concat_sections = list(self.concat_metadata.sections)
        self.sections_dict = self.concat_metadata.sections_by_key

    @staticmethod
    def _section_key(section_type: str) -> str:
        """Normalize a section label into a stable dict key."""
        return normalize_section_key(section_type)

    @staticmethod
    def _section_kind(section_type: str) -> str:
        """Validate the canonical section type stored in the concat CSV."""
        return validate_section_kind(section_type)


_REPORTING_EXPORTS = {
    "VideoFiguresWriter",
    "VideoStatistics",
    "VideoStatisticsWriter",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve legacy reporting imports without loading matplotlib."""
    if name in _REPORTING_EXPORTS:
        from gcamp_analysis import reporting

        return getattr(reporting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
