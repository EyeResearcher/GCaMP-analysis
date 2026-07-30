"""Mutable data and pipeline state for one Suite2p video.

``Video`` owns injected Suite2p inputs, path-derived metadata, and results
populated by processing services. Its constructor does not access the
filesystem. Use ``Video.from_suite2p`` when loading a video from disk, or
inject ``suite2p_data`` directly in tests and alternate loaders.

Report serialization and figure writing live in ``gcamp_analysis.reporting``:

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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

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

    # ---- Metadata (derived)
    video_id: str = field(init=False)
    experiment_name: str = field(init=False, default="unknown")
    treatment: str = field(init=False, default="unknown")
    timepoint_name: str = field(init=False, default="unknown")

    n_rois: int = field(init=False, default=0)
    n_frames: int = field(init=False, default=0)
    fs: float = field(init=False, default=15.0)

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

    @classmethod
    def from_suite2p(
        cls,
        *,
        path: Path,
        suite2p_path: Path,
    ) -> Video:
        """Load Suite2p arrays from disk, then construct a video."""
        path = Path(path)
        suite2p_path = Path(suite2p_path)
        suite2p_data = load_suite2p_data(suite2p_path)

        return cls(
            path=path,
            suite2p_path=suite2p_path,
            suite2p_data=suite2p_data,
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

    def clear_results(self) -> None:
        """Reset all computed fields to defaults. Useful for re-running in notebooks."""
        self.norm_f = _empty_array()
        self.norm_sm_f = _empty_array()
        self.norm_sg_f = _empty_array()
        self.z_f = _empty_array()
        self.savgol_z_f = _empty_array()
        self.sm_sp = _empty_array()

        self.n_good_rois = 0
        self.n_bad_rois = 0
        self.bad_rois = []
        self.bad_rois_features = pd.DataFrame()

        self.neurons = []
        self.summary_df = pd.DataFrame()

        self.grouping_results = {}
        self.grouping_stats = pd.DataFrame()

    def __repr__(self) -> str:
        return (
            f"Video(video_id={self.video_id}, n_rois={self.n_rois}, n_frames={self.n_frames}, "
            f"treatment={self.treatment}, timepoint={self.timepoint_name})"
        )

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
