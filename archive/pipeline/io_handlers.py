"""I/O helpers for saving pipeline outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from utils.io_utils import save_timepoint_summary, save_sttc_heatmap

from .config import OutputConfig
from .neuron_grouping import GroupingResult


@dataclass(slots=True)
class VideoArtifacts:
    video_id: str
    excel_path: Path | None
    summary_row: Dict[str, float]


def _build_group_table(grouping: GroupingResult) -> pd.DataFrame:
    if grouping is None or not grouping.neuron_groups:
        return pd.DataFrame(columns=["group_id", "members", "avg_sttc"])
    records = []
    for idx, group in enumerate(grouping.neuron_groups):
        avg = grouping.average_group_sttc[idx] if idx < len(grouping.average_group_sttc) else np.nan
        records.append({
            "group_id": idx,
            "members": group,
            "avg_sttc": avg,
        })
    table = pd.DataFrame.from_records(records)
    table.index.name = "group_id"
    return table


def summarize_video(
    video_id: str,
    neuron_table: pd.DataFrame,
    grouping: GroupingResult,
) -> Dict[str, float]:
    num_cells = len(neuron_table)
    avg_spikes = neuron_table["num_spikes"].mean() if num_cells else np.nan
    avg_freq = neuron_table["spike_frequency"].mean() if num_cells else np.nan
    avg_amp = neuron_table["avg_peak"].mean() if num_cells else np.nan

    num_groups = len(grouping.neuron_groups) if grouping else 0
    cells_in_groups = sum(len(group) for group in grouping.neuron_groups) if grouping else 0
    avg_cells_per_group = (cells_in_groups / num_groups) if num_groups else np.nan
    percent_in_groups = (cells_in_groups / num_cells) if num_cells else np.nan

    return {
        "video_id": video_id,
        "num_cells": num_cells,
        "avg_spikes_per_cell": avg_spikes,
        "avg_spike_frequency": avg_freq,
        "avg_peak_amplitude": avg_amp,
        "num_groups": num_groups,
        "avg_cells_per_group": avg_cells_per_group,
        "percent_in_groups": percent_in_groups,
    }


def write_video_outputs(
    video_path: Path,
    roi_table: pd.DataFrame,
    spike_table: pd.DataFrame,
    neuron_table: pd.DataFrame,
    grouping: GroupingResult,
    config: OutputConfig,
) -> VideoArtifacts:
    metrics_dir = Path(video_path) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    has_groups = bool(grouping and grouping.neuron_groups)
    excel_path: Path | None
    if has_groups:
        excel_path = metrics_dir / config.video_summary_filename
        with pd.ExcelWriter(excel_path) as writer:
            roi_table.to_excel(writer, sheet_name="ROIs")
            spike_table.to_excel(writer, sheet_name="Spikes", index=False)
            neuron_table.to_excel(writer, sheet_name="Neurons", index=False)
            group_table = _build_group_table(grouping)
            group_table.to_excel(writer, sheet_name="Groups")
    else:
        excel_path = None
        print(f"no neuron groups for {video_path.name}")

    sttc_path = metrics_dir / "sttc_matrix.npy"
    if grouping is not None and grouping.sttc_matrix is not None:
        np.save(sttc_path, grouping.sttc_matrix)
        save_sttc_heatmap(grouping.sttc_matrix, sttc_path.with_suffix(".png"))

    summary_row = summarize_video(video_path.name, neuron_table, grouping)
    return VideoArtifacts(video_id=video_path.name, excel_path=excel_path, summary_row=summary_row)


def compile_timepoint_summary(video_artifacts: Iterable[VideoArtifacts]) -> pd.DataFrame:
    records = [artifact.summary_row for artifact in video_artifacts]
    if not records:
        return pd.DataFrame(columns=[
            "video_id",
            "num_cells",
            "avg_spikes_per_cell",
            "avg_spike_frequency",
            "avg_peak_amplitude",
            "num_groups",
            "avg_cells_per_group",
            "percent_in_groups",
        ])
    return pd.DataFrame.from_records(records).set_index("video_id")


def write_timepoint_outputs(
    experiment_name: str,
    timepoint_name: str,
    timepoint_path: Path,
    summary_df: pd.DataFrame,
    video_tables: Dict[str, pd.DataFrame],
) -> Path:
    return save_timepoint_summary(
        experiment_name=experiment_name,
        timepoint_name=timepoint_name,
        timepoint_df=summary_df,
        video_dfs=video_tables,
        output_dir=timepoint_path,
        filename=None,
    )
