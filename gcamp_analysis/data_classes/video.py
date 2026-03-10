from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
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
    fs: float = field(init=False, default=15.0)

    # ---- Concatenated-video support
    is_concatenated: bool = False
    split_frame: Optional[int] = None

    # ---- Traces (populated by TraceService)
    norm_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    norm_sm_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    norm_sg_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    sm_sp: np.ndarray = field(default_factory=_empty_array, repr=False)  # optional if/when used

    # ---- Per-segment traces (populated by TraceService when is_concatenated)
    baseline_norm_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    baseline_norm_sm_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    baseline_norm_sg_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    treatment_norm_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    treatment_norm_sm_f: np.ndarray = field(default_factory=_empty_array, repr=False)
    treatment_norm_sg_f: np.ndarray = field(default_factory=_empty_array, repr=False)

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

    # ---- Treatment comparison outputs (populated by GroupingService when is_concatenated)
    treatment_comparison_results: dict = field(default_factory=dict, repr=False)

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
        self.fs = float(self.suite2p_data.get("fs", self.suite2p_data.get("ops", {}).get("fs", 15.0)))

        # Parse experiment metadata from folder structure (best-effort)
        self._parse_metadata()

    # --------- Segment properties ---------

    @property
    def baseline_slice(self) -> slice:
        """Frame slice for the baseline (first) half."""
        if not self.is_concatenated or self.split_frame is None:
            return slice(0, self.n_frames)
        return slice(0, self.split_frame)

    @property
    def treatment_slice(self) -> slice:
        """Frame slice for the treatment (second) half."""
        if not self.is_concatenated or self.split_frame is None:
            return slice(0, self.n_frames)
        return slice(self.split_frame, self.n_frames)

    @property
    def baseline_n_frames(self) -> int:
        if not self.is_concatenated or self.split_frame is None:
            return self.n_frames
        return self.split_frame

    @property
    def treatment_n_frames(self) -> int:
        if not self.is_concatenated or self.split_frame is None:
            return self.n_frames
        return self.n_frames - self.split_frame

    # --------- Small helpers (ok to keep on Video) ---------

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

    @property
    def metrics_dir(self) -> Path:
        """Default per-video output folder."""
        return self.path / "metrics"

    def clear_results(self) -> None:
        """Reset all computed fields to defaults. Useful for re-running in notebooks."""
        self.norm_f = _empty_array()
        self.norm_sm_f = _empty_array()
        self.norm_sg_f = _empty_array()
        self.sm_sp = _empty_array()

        # Per-segment traces
        self.baseline_norm_f = _empty_array()
        self.baseline_norm_sm_f = _empty_array()
        self.baseline_norm_sg_f = _empty_array()
        self.treatment_norm_f = _empty_array()
        self.treatment_norm_sm_f = _empty_array()
        self.treatment_norm_sg_f = _empty_array()

        self.n_good_rois = 0
        self.n_bad_rois = 0
        self.bad_rois = []
        self.bad_rois_features = pd.DataFrame()

        self.neurons = []
        self.summary_df = pd.DataFrame()

        self.grouping_results = {}
        self.grouping_stats = pd.DataFrame()
        self.treatment_comparison_results = {}

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

    # All strategy matrices keyed by strategy name
    matrices: dict = field(default_factory=dict)

    # Per-group light-evoked detail DataFrames keyed by group_id
    light_evoked_details: dict = field(default_factory=dict)

    # Treatment comparison results (concatenated mode only)
    treatment_comparison: dict = field(default_factory=dict)

    @classmethod
    def from_video(cls, video: "Video") -> "VideoStatistics":
        """Convenience constructor; keeps Video dependency out of __init__."""
        from gcamp_analysis.grouping_processing.light_evoked_detail import (
            build_light_evoked_detail,
        )

        matrices = {}
        for name, result in getattr(video, "grouping_results", {}).items():
            if result.matrix is not None:
                matrices[name] = result.matrix

        # Build light-evoked detail tables if the strategy was run
        light_evoked_details: dict = {}
        le_result = getattr(video, "grouping_results", {}).get("light-evoked")
        if le_result is not None:
            light_evoked_details = build_light_evoked_detail(
                le_result, fs=float(video.fs),
            )

        return cls(
            video_name=video.path.name,
            per_neuron_spike_summaries=video.summary_df,
            grouping_stats=video.grouping_stats,
            bad_rois_features=video.bad_rois_features,
            matrices=matrices,
            light_evoked_details=light_evoked_details,
            treatment_comparison=getattr(video, "treatment_comparison_results", {}),
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
        """Resolve the metrics directory, avoiding double-nesting if *output_root* already ends with *video_name*."""
        if output_root.name == video_name:
            return output_root / "metrics"
        return output_root / video_name / "metrics"

    def write(self, stats: "VideoStatistics", output_root: Path) -> dict[str, str]:
        """Write tables and matrices to disk. Returns a manifest of saved file paths."""
        out_dir = self.metrics_dir(output_root, stats.video_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        base = stats.video_name
        manifest: dict[str, str] = {}

        # Tables
        excel_path = out_dir / f"{base}_metrics.xlsx"
        sheets_written = False
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            if not stats.per_neuron_spike_summaries.empty:
                stats.per_neuron_spike_summaries.to_excel(
                    writer, sheet_name='spike_summary', index=False
                )
                sheets_written = True
            
            if not stats.grouping_stats.empty:
                stats.grouping_stats.to_excel(
                    writer, sheet_name='grouping_stats', index=False
                )
                sheets_written = True
            
            if not stats.bad_rois_features.empty:
                stats.bad_rois_features.to_excel(
                    writer, sheet_name='bad_rois_features', index=True
                )
                sheets_written = True

            # Light-evoked detail sheets
            for sheet_key, detail_df in stats.light_evoked_details.items():
                if detail_df is not None and not detail_df.empty:
                    # Excel sheet names are limited to 31 chars
                    sheet_name = sheet_key[:31]
                    detail_df.to_excel(
                        writer, sheet_name=sheet_name, index=False
                    )
                    sheets_written = True

            if not sheets_written:
                pd.DataFrame().to_excel(writer, sheet_name='empty', index=False)

        manifest["metrics_excel"] = str(excel_path)

        # Save all strategy matrices
        for mat_name, matrix in stats.matrices.items():
            npy_path = out_dir / f"{base}_{mat_name}_matrix.npy"
            np.save(npy_path, matrix)
            manifest[f"{mat_name}_matrix_npy"] = str(npy_path)

        # Save treatment comparison results (concatenated mode)
        if stats.treatment_comparison:
            for strategy_name, tc_result in stats.treatment_comparison.items():
                # Save per-group metrics as a DataFrame
                if hasattr(tc_result, "group_metrics") and tc_result.group_metrics:
                    tc_df = pd.DataFrame(tc_result.group_metrics)
                    tc_path = out_dir / f"{base}_{strategy_name}_treatment_comparison.csv"
                    tc_df.to_csv(tc_path, index=False)
                    manifest[f"{strategy_name}_treatment_comparison"] = str(tc_path)

                # Save treatment similarity matrix
                if hasattr(tc_result, "treatment_matrix") and tc_result.treatment_matrix is not None:
                    tc_mat_path = out_dir / f"{base}_{strategy_name}_treatment_matrix.npy"
                    np.save(tc_mat_path, tc_result.treatment_matrix)
                    manifest[f"{strategy_name}_treatment_matrix_npy"] = str(tc_mat_path)

        return manifest
    
    
@dataclass
class VideoFiguresWriter:
    """Saves grouping overlay and heatmap figures for a video."""
    dpi: int = 200
    close_figs: bool = True

    def save_fig(self, fig: Optional[Figure], path: Path) -> None:
        """Save a single figure to *path*. No-op if *fig* is None."""
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
        strategy_name: str = "corr",
        config_label: str | None = None,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Generate and save overlay + heatmap for one grouping strategy. Returns saved paths."""
        from gcamp_analysis.grouping_processing.service import visualize_grouping

        overlay_fig, heatmap_fig = visualize_grouping(
            video, strategy_name=strategy_name, config_label=config_label
        )

        base = video.path.name
        overlay_path = out_dir / f"{base}_{strategy_name}_groups.png"
        heatmap_path = out_dir / f"{base}_{strategy_name}_heatmap.png"

        self.save_fig(overlay_fig, overlay_path)
        self.save_fig(heatmap_fig, heatmap_path)

        return (overlay_path if overlay_fig else None, heatmap_path if heatmap_fig else None)

    def write_treatment_figures(
        self,
        video: "Video",
        *,
        out_dir: Path,
    ) -> dict:
        """Generate and save treatment-comparison spatial dispersion figures."""
        from gcamp_analysis.grouping_processing.service import visualize_treatment_comparison

        manifest: dict = {}
        tc_results = getattr(video, "treatment_comparison_results", {})
        base = video.path.name
        for strategy_name in tc_results:
            delta_fig, centroid_fig = visualize_treatment_comparison(
                video, strategy_name=strategy_name,
            )
            if delta_fig is not None:
                p = out_dir / f"{base}_{strategy_name}_delta_corr_vs_dispersion.png"
                self.save_fig(delta_fig, p)
                manifest[f"{strategy_name}_delta_corr_png"] = str(p)
            if centroid_fig is not None:
                p = out_dir / f"{base}_{strategy_name}_centroid_distances.png"
                self.save_fig(centroid_fig, p)
                manifest[f"{strategy_name}_centroid_dist_png"] = str(p)
        return manifest

    def write(self, video: "Video") -> dict:
        """Write all grouping figures for *video*. Returns a manifest of saved paths."""
        out_dir = video.path / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict = {}

        # Generate figures for every strategy that was run
        for name in getattr(video, "grouping_results", {}):
            overlay, heat = self.write_grouping_figures(
                video, out_dir=out_dir, strategy_name=name
            )
            if overlay:
                manifest[f"{name}_overlay_png"] = str(overlay)
            if heat:
                manifest[f"{name}_heatmap_png"] = str(heat)

        # Treatment comparison figures (concatenated mode)
        manifest.update(self.write_treatment_figures(video, out_dir=out_dir))

        return manifest