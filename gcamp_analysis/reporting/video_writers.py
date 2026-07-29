"""Filesystem and visualization writers for per-video reporting outputs.

All reporting side effects belong here rather than on ``Video``. Statistics
writers consume immutable ``VideoStatistics`` snapshots. Figure writers still
consume ``Video`` because visualization services require processed spatial and
grouping state.

When adding a reportable field, first add it to ``VideoStatistics`` and its
``from_video`` projection. Then update only the writer responsible for that
output format. Writers should not calculate scientific results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from gcamp_analysis.reporting.video_statistics import VideoStatistics

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.video import Video


@dataclass
class VideoStatisticsWriter:
    """Write a ``VideoStatistics`` snapshot to tabular and NumPy files."""

    save_fig_dpi: int = 300
    save_fig_bbox_inches: str = "tight"

    def metrics_dir(self, output_root: Path, video_name: str) -> Path:
        """Resolve the metrics directory without double-nesting the video."""
        if output_root.name == video_name:
            return output_root / "metrics"
        return output_root / video_name / "metrics"

    def write(
        self,
        stats: VideoStatistics,
        output_root: Path,
    ) -> dict[str, str]:
        """Write tables and matrices and return their saved paths."""
        out_dir = self.metrics_dir(output_root, stats.video_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        base = stats.video_name
        manifest: dict[str, str] = {}
        used_sheet_names: set[str] = set()

        def _unique_sheet_name(preferred: str) -> str:
            """Return an Excel-safe unique sheet name capped at 31 chars."""
            base_name = preferred[:31] or "sheet"
            candidate = base_name
            counter = 1
            while candidate in used_sheet_names:
                suffix = f"_{counter}"
                candidate = f"{base_name[: 31 - len(suffix)]}{suffix}"
                counter += 1
            used_sheet_names.add(candidate)
            return candidate

        excel_path = out_dir / f"{base}_metrics.xlsx"
        sheets_written = False
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            if not stats.per_neuron_spike_summaries.empty:
                stats.per_neuron_spike_summaries.to_excel(
                    writer,
                    sheet_name=_unique_sheet_name("spike_summary"),
                    index=False,
                )
                sheets_written = True

            if not stats.grouping_stats.empty:
                stats.grouping_stats.to_excel(
                    writer,
                    sheet_name=_unique_sheet_name("grouping_stats"),
                    index=False,
                )
                sheets_written = True

            if not stats.bad_rois_features.empty:
                stats.bad_rois_features.to_excel(
                    writer,
                    sheet_name=_unique_sheet_name("bad_rois_features"),
                    index=True,
                )
                sheets_written = True

            for sheet_key, detail_df in stats.light_evoked_details.items():
                if detail_df is not None and not detail_df.empty:
                    detail_df.to_excel(
                        writer,
                        sheet_name=_unique_sheet_name(sheet_key),
                        index=False,
                    )
                    sheets_written = True

            for section_results in stats.section_comparison.values():
                for section_key, comparison_result in section_results.items():
                    if not getattr(comparison_result, "group_metrics", None):
                        continue
                    section_df = pd.DataFrame(comparison_result.group_metrics)
                    if section_df.empty:
                        continue
                    section_df.to_excel(
                        writer,
                        sheet_name=_unique_sheet_name(
                            f"baseline-{section_key}"
                        ),
                        index=False,
                    )
                    sheets_written = True

            if not sheets_written:
                pd.DataFrame().to_excel(
                    writer,
                    sheet_name=_unique_sheet_name("empty"),
                    index=False,
                )

        manifest["metrics_excel"] = str(excel_path)

        for matrix_name, matrix in stats.matrices.items():
            matrix_path = out_dir / f"{base}_{matrix_name}_matrix.npy"
            np.save(matrix_path, matrix)
            manifest[f"{matrix_name}_matrix_npy"] = str(matrix_path)

        for strategy_name, section_results in stats.section_comparison.items():
            for section_key, comparison_result in section_results.items():
                if getattr(comparison_result, "group_metrics", None):
                    section_df = pd.DataFrame(comparison_result.group_metrics)
                    section_path = (
                        out_dir
                        / f"{base}_{strategy_name}_{section_key}_section_comparison.csv"
                    )
                    section_df.to_csv(section_path, index=False)
                    manifest[
                        f"{strategy_name}_{section_key}_section_comparison"
                    ] = str(section_path)

                section_matrix = getattr(
                    comparison_result,
                    "section_matrix",
                    None,
                )
                if section_matrix is not None:
                    matrix_path = (
                        out_dir
                        / f"{base}_{strategy_name}_{section_key}_section_matrix.npy"
                    )
                    np.save(matrix_path, section_matrix)
                    manifest[
                        f"{strategy_name}_{section_key}_section_matrix_npy"
                    ] = str(matrix_path)

        return manifest


@dataclass
class VideoFiguresWriter:
    """Generate and save grouping and section-comparison figures."""

    dpi: int = 200
    close_figs: bool = True

    def save_fig(self, fig: Optional[Figure], path: Path) -> None:
        """Save one figure, closing it when configured."""
        if fig is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        if self.close_figs:
            plt.close(fig)

    def write_grouping_figures(
        self,
        video: Video,
        *,
        out_dir: Path,
        strategy_name: str = "corr",
        config_label: str | None = None,
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Generate and save overlay and heatmap figures for one strategy."""
        from gcamp_analysis.grouping_processing.service import (
            visualize_grouping,
        )

        overlay_fig, heatmap_fig = visualize_grouping(
            video,
            strategy_name=strategy_name,
            config_label=config_label,
        )

        base = video.path.name
        overlay_path = out_dir / f"{base}_{strategy_name}_groups.png"
        heatmap_path = out_dir / f"{base}_{strategy_name}_heatmap.png"

        self.save_fig(overlay_fig, overlay_path)
        self.save_fig(heatmap_fig, heatmap_path)

        return (
            overlay_path if overlay_fig else None,
            heatmap_path if heatmap_fig else None,
        )

    def write_section_figures(
        self,
        video: Video,
        *,
        out_dir: Path,
    ) -> dict:
        """Generate and save section-comparison dispersion figures."""
        from gcamp_analysis.grouping_processing.service import (
            visualize_section_comparison,
        )

        manifest: dict = {}
        base = video.path.name
        for strategy_name, section_results in (
            video.section_comparison_results.items()
        ):
            for section_key in section_results:
                delta_fig, centroid_fig = visualize_section_comparison(
                    video,
                    strategy_name=strategy_name,
                    section_key=section_key,
                )
                if delta_fig is not None:
                    path = (
                        out_dir
                        / f"{base}_{strategy_name}_{section_key}_delta_corr_vs_dispersion.png"
                    )
                    self.save_fig(delta_fig, path)
                    manifest[
                        f"{strategy_name}_{section_key}_delta_corr_png"
                    ] = str(path)
                if centroid_fig is not None:
                    path = (
                        out_dir
                        / f"{base}_{strategy_name}_{section_key}_centroid_distances.png"
                    )
                    self.save_fig(centroid_fig, path)
                    manifest[
                        f"{strategy_name}_{section_key}_centroid_dist_png"
                    ] = str(path)
        return manifest

    def write(self, video: Video) -> dict:
        """Write all available grouping and section figures."""
        out_dir = video.path / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict = {}
        for strategy_name in video.grouping_results:
            overlay, heatmap = self.write_grouping_figures(
                video,
                out_dir=out_dir,
                strategy_name=strategy_name,
            )
            if overlay:
                manifest[f"{strategy_name}_overlay_png"] = str(overlay)
            if heatmap:
                manifest[f"{strategy_name}_heatmap_png"] = str(heatmap)

        manifest.update(self.write_section_figures(video, out_dir=out_dir))
        return manifest
