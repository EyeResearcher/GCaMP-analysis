from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any, Literal, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from utils.io_utils import load_suite2p_data
from grouping_processing.visualization import visualize_grouping
if TYPE_CHECKING:
    from data_classes.roi import ROI
    from data_classes.neuron import Neuron
    from data_classes.neuron_group import NeuronGroup


def _empty_array() -> np.ndarray:
    return np.asarray([])


@dataclass
class Video:
    """
    Thin, stateful context object for a single video folder.

    Responsibilities (keep these here):
      - hold paths + metadata
      - load Suite2p once
      - store pipeline outputs (so VideoStatistics can pull from this)
      - avoid heavy computation methods (those live in services)

    Everything that *computes* should live in pipeline/services/*.
    """

    # ---- Inputs
    path: Path
    suite2p_path: Path  # e.g. <video_dir>/suite2p/plane0

    # ---- Loaded data
    suite2p_data: dict[str, Any] = field(init=False, repr=False)

    # ---- Metadata (derived)
    video_id: str = field(init=False)
    experiment_name: str = field(init=False, default="unknown")
    treatment: str = field(init=False, default="unknown")
    timepoint_name: str = field(init=False, default="unknown")

    n_rois: int = field(init=False, default=0)
    n_frames: int = field(init=False, default=0)
    fs: float = field(init=False, default=30.0)

    # ---- Traces (populated by TraceService)
    norm_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    norm_sm_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    norm_sg_f: np.ndarray = field(default_factory=_empty_array, repr=False)
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
    sttc_matrix: np.ndarray = field(default_factory=_empty_array, repr=False)
    dtw_matrix: np.ndarray = field(default_factory=_empty_array, repr=False)

    sttc_groups: list["NeuronGroup"] = field(default_factory=list, repr=False)
    dtw_groups: list["NeuronGroup"] = field(default_factory=list, repr=False)

    agreement: float = 0.0
    grouping_stats: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # ---- Figures (created by GroupingService.visualize, optional)
    sttc_fig: Optional[Figure] = None
    dtw_fig: Optional[Figure] = None
    sttc_heatmap: Optional[Figure] = None
    dtw_heatmap: Optional[Figure] = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.suite2p_path = Path(self.suite2p_path)

        self.video_id = self.path.name

        # Load suite2p once
        self.suite2p_data = load_suite2p_data(self.suite2p_path)

        # Basic dimensions
        F = self.suite2p_data.get("F", None)
        if F is None:
            raise ValueError(f"Suite2p data at {self.suite2p_path} missing key 'F'.")
        self.n_rois, self.n_frames = F.shape

        # Sampling rate (best-effort)
        self.fs = float(self.suite2p_data.get("fs", self.suite2p_data.get("ops", {}).get("fs", 30.0)))

        # Parse experiment metadata from folder structure (best-effort)
        self._parse_metadata()

    # --------- Small helpers (ok to keep on Video) ---------

    def _parse_metadata(self) -> None:
        """
        Best-effort metadata parse from directory path.
        This stays in Video because it's "context", not pipeline computation.
        """
        parts = list(self.path.parts)

        # Example structures you mentioned:
        # Experiment337 / GABA / Week1 / vid001
        # We treat the last 3 ancestors as (timepoint, treatment, experiment) when available.
        if len(parts) >= 4:
            self.timepoint_name = parts[-2]
            self.treatment = parts[-3]
            self.experiment_name = parts[-4]
        else:
            self.timepoint_name = "unknown"
            self.treatment = "unknown"
            self.experiment_name = "unknown"

    @property
    def metrics_dir(self) -> Path:
        """Default per-video output folder."""
        return self.path / "metrics"

    def clear_results(self) -> None:
        """
        Optional convenience: reset all computed fields.
        Useful in notebooks if you re-run with different models.
        """
        self.norm_f = _empty_array()
        self.norm_sm_f = _empty_array()
        self.norm_sg_f = _empty_array()
        self.sm_sp = _empty_array()

        self.n_good_rois = 0
        self.n_bad_rois = 0
        self.bad_rois = []
        self.bad_rois_features = pd.DataFrame()

        self.neurons = []
        self.summary_df = pd.DataFrame()

        self.sttc_matrix = _empty_array()
        self.dtw_matrix = _empty_array()
        self.sttc_groups = []
        self.dtw_groups = []
        self.agreement = 0.0
        self.grouping_stats = pd.DataFrame()

        self.sttc_fig = None
        self.dtw_fig = None

    def __repr__(self) -> str:
        return (
            f"Video(video_id={self.video_id}, n_rois={self.n_rois}, n_frames={self.n_frames}, "
            f"treatment={self.treatment}, timepoint={self.timepoint_name})"
        )

@dataclass(frozen=True)
class VideoStatistics:
    """Pure in-memory container for per-video outputs."""
    video_name: str

    per_neuron_spike_summaries: pd.DataFrame
    grouping_stats: pd.DataFrame
    bad_rois_features: pd.DataFrame

    sttc_matrix: np.ndarray
    dtw_matrix: np.ndarray

    sttc_fig: Optional[Figure] = None 
    dtw_fig: Optional[Figure] = None

    sttc_heatmap: Optional[Figure] = None
    dtw_heatmap: Optional[Figure] = None

    @classmethod
    def from_video(cls, video : "Video") -> "VideoStatistics":
        """Convenience constructor; keeps Video dependency out of __init__."""
        return cls(
            video_name=video.path.name,
            per_neuron_spike_summaries=video.summary_df,
            grouping_stats=video.grouping_stats,
            bad_rois_features=video.bad_rois_features,
            sttc_matrix=video.sttc_matrix,
            dtw_matrix=video.dtw_matrix,
            sttc_fig=getattr(video, "sttc_fig", None),
            dtw_fig=getattr(video, "dtw_fig", None),
            sttc_heatmap=getattr(video, "sttc_heatmap", None),
            dtw_heatmap=getattr(video, "dtw_heatmap", None),
            )
    
@dataclass
class VideoStatisticsWriter:
    """
    Responsible for writing VideoStatistics to disk.

    Directory scheme:
      <output_root>/<video_name>/metrics/...
    or if you pass output_dir directly, it writes into that folder.
    """
    save_fig_dpi: int = 300
    save_fig_bbox_inches: str = "tight"

    def metrics_dir(self, output_root: Path, video_name: str) -> Path:
        # You can change this scheme later without touching VideoStatistics
        if output_root.name == video_name:
            return output_root / "metrics"
        return output_root / video_name / "metrics"

    def write(self, stats: "VideoStatistics", output_root: Path) -> dict[str, str]:
        out_dir = self.metrics_dir(output_root, stats.video_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        base = stats.video_name
        manifest: dict[str, str] = {}

        # Tables
        excel_path = out_dir / f"{base}_metrics.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            if not stats.per_neuron_spike_summaries.empty:
                stats.per_neuron_spike_summaries.to_excel(
                    writer, sheet_name='spike_summary', index=False
                )
            
            if not stats.grouping_stats.empty:
                stats.grouping_stats.to_excel(
                    writer, sheet_name='grouping_stats', index=False
                )
            
            if not stats.bad_rois_features.empty:
                stats.bad_rois_features.to_excel(
                    writer, sheet_name='bad_rois_features', index=True
                )
        
        manifest["metrics_excel"] = str(excel_path)
        # Matrices
        sttc_npy = out_dir / f"{base}_sttc_matrix.npy"
        np.save(sttc_npy, stats.sttc_matrix)
        manifest["sttc_matrix_npy"] = str(sttc_npy)

        dtw_npy = out_dir / f"{base}_dtw_matrix.npy"
        np.save(dtw_npy, stats.dtw_matrix)
        manifest["dtw_matrix_npy"] = str(dtw_npy)


        return manifest
    
    
Which = Literal["sttc", "dtw"]


@dataclass
class VideoFiguresWriter:
    dpi: int = 200
    close_figs: bool = True

    def save_fig(self, fig: Optional[Figure], path: Path) -> None:
        if fig is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        if self.close_figs:
            plt.close(fig)

    def write_grouping_figures(
        self,
        video: "Video",
        *,
        out_dir: Path,
        which: Which = "sttc",
        config_label: str | None = None,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        overlay_fig, heatmap_fig = visualize_grouping(video, which=which, config_label=config_label)

        base = video.path.name
        overlay_path = out_dir / f"{base}_{which}_groups.png"
        heatmap_path = out_dir / f"{base}_{which}_heatmap.png"

        self.save_fig(overlay_fig, overlay_path)
        self.save_fig(heatmap_fig, heatmap_path)

        return (overlay_path if overlay_fig else None, heatmap_path if heatmap_fig else None)

    def write(self, video: "Video") -> dict:
        out_dir = video.path / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict = {}

        # STTC
        sttc_overlay, sttc_heat = self.write_grouping_figures(video, out_dir=out_dir, which="sttc")
        if sttc_overlay: manifest["sttc_overlay_png"] = str(sttc_overlay)
        if sttc_heat: manifest["sttc_heatmap_png"] = str(sttc_heat)

        # DTW (only if enabled / exists)
        if getattr(video, "dtw_matrix", None) is not None and np.asarray(getattr(video, "dtw_matrix")).size > 0:
            dtw_overlay, dtw_heat = self.write_grouping_figures(video, out_dir=out_dir, which="dtw")
            if dtw_overlay: manifest["dtw_overlay_png"] = str(dtw_overlay)
            if dtw_heat: manifest["dtw_heatmap_png"] = str(dtw_heat)

        return manifest