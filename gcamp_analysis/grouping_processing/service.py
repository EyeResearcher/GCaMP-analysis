"""Grouping service: orchestration, comparison, summary, and visualization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from gcamp_analysis.data_classes.neuron_group import NeuronGroup
from gcamp_analysis.grouping_processing.strategies import STRATEGY_REGISTRY
from gcamp_analysis.reports import GroupingReport
from utils.visualization import (
    plot_matrix_heatmap,
    visualize_neuron_groups,
)

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.video import Video


@dataclass(frozen=True)
class GroupingResult:
    """Output of a single grouping strategy."""

    groups: list[NeuronGroup] | list[dict]
    matrix: np.ndarray | None
    config_label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def neuron_groups_from_dicts(
    group_dicts: List[dict],
    neurons: list,
    *,
    method: str = "combined",
) -> List[NeuronGroup]:
    """Build :class:`NeuronGroup` objects from strategy group dictionaries.

    Neuron indices absent from *neurons* are skipped, and empty groups are
    dropped.
    """
    idx_to_neuron = {neuron.index: neuron for neuron in neurons}
    groups: List[NeuronGroup] = []
    for group_dict in group_dicts:
        group_neurons = [
            idx_to_neuron[index]
            for index in group_dict["neuron_indices"]
            if index in idx_to_neuron
        ]
        if not group_neurons:
            continue
        groups.append(
            NeuronGroup(
                group_id=group_dict["group_id"],
                neurons=group_neurons,
                method=method,
                row_indices=group_dict.get("row_indices"),
            )
        )
    return groups


def compute_group_summary_rows(
    groups: List[NeuronGroup],
    *,
    method: str,
    matrices: Dict[str, Optional[np.ndarray]],
) -> List[Dict[str, Any]]:
    """Return one summary row per group with size, activity, and matrix means."""
    rows: List[Dict[str, Any]] = []
    for group in groups:
        summary_stats = [getattr(neuron, "summary_stats", {}) for neuron in group.neurons]
        stats_df = pd.DataFrame(summary_stats)

        rates = stats_df.get("spike_frequency", pd.Series(dtype=float))
        num_spikes = stats_df.get("number_of_spikes", pd.Series(dtype=float))
        mean_of_means = stats_df.filter(like="mean_").mean(numeric_only=True).to_dict()

        row: Dict[str, Any] = {
            "group_id": group.group_id,
            "method": method,
            "number_neurons": int(group.size),
            "neuron_indices": list(getattr(group, "neuron_indices", [])),
            "filtered_idxs": list(getattr(group, "filtered_idxs", [])),
            "spike_rate": float(np.nanmean(rates)) if len(rates) else 0.0,
            "number_of_spikes": float(np.nanmean(num_spikes)) if len(num_spikes) else 0.0,
            **mean_of_means,
        }

        for matrix_name, matrix in matrices.items():
            try:
                row[f"mean_{matrix_name}"] = group.group_mean_similarity(matrix)
            except Exception:
                row[f"mean_{matrix_name}"] = np.nan

        for key in ("t_win", "corr_thresh", "sttc_thresh", "dtw_thresh"):
            value = group.metadata.get(key)
            if value is not None:
                row[key] = value

        rows.append(row)
    return rows


def build_combined_summary(results: Dict[str, GroupingResult]) -> List[dict]:
    """Flatten every strategy's groups into one list of summary rows."""
    matrices = {name: result.matrix for name, result in results.items()}
    all_rows: list[dict] = []
    for name, result in results.items():
        all_rows.extend(
            compute_group_summary_rows(result.groups, method=name, matrices=matrices)
        )
    return all_rows


def _infer_img_size(video: "Video", default=(1024, 1024)) -> tuple[int, int]:
    """Return the ``(Ly, Lx)`` image size from Suite2p ops, or *default*."""
    ops = getattr(video, "suite2p_data", {}).get("ops", {}) if getattr(video, "suite2p_data", None) else {}
    ly = int(ops.get("Ly", default[0]))
    lx = int(ops.get("Lx", default[1]))
    return (ly, lx)


def make_matrix_heatmap(
    matrix: np.ndarray,
    *,
    title: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize=(6, 5),
) -> Optional[Figure]:
    """Render a square matrix as a heatmap figure, or ``None`` if empty."""
    if matrix is None:
        return None
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.size == 0:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    plot_matrix_heatmap(matrix, title=title, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, show_colorbar=True)
    fig.tight_layout()
    return fig


def visualize_grouping(
    video: "Video",
    *,
    strategy_name: str = "corr",
    config_label: Optional[str] = None,
    heatmap_cmap: str = "viridis",
    heatmap_vmin: Optional[float] = None,
    heatmap_vmax: Optional[float] = None,
) -> Tuple[Optional[Figure], Optional[Figure]]:
    """Return (spatial group overlay, matrix heatmap) figures for a strategy.

    Either element is ``None`` when the strategy produced no result, no
    groups, or no matrix.
    """
    result = video.grouping_results.get(strategy_name)
    if result is None:
        return None, None

    groups = result.groups
    matrix = result.matrix

    label = config_label or strategy_name
    heat_title = f"{strategy_name} matrix ({label})"

    overlay_fig: Optional[Figure] = None
    if groups:
        img_size = _infer_img_size(video)
        stat = getattr(video, "suite2p_data", {}).get("stat", np.array([]))
        overlay_fig = visualize_neuron_groups(
            neuron_groups=groups,
            stat=stat,
            img_size=img_size,
            video_path=getattr(video, "path", None),
            config_label=label,
        )

    heatmap_fig = make_matrix_heatmap(
        matrix,
        title=heat_title,
        cmap=heatmap_cmap,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )
    return overlay_fig, heatmap_fig


@dataclass
class GroupingService:
    """Run one or more grouping strategies and compare them."""

    strategies: list[str] = field(default_factory=lambda: ["combined"])

    def _get_grouping_kwargs(self, video: "Video") -> dict[str, Any]:
        all_neurons = list(getattr(video, "neurons", []))
        n_frames = getattr(video, "n_frames", None)
        fs = float(getattr(video, "fs", 15.0))

        active_neurons = list(all_neurons)

        traces = np.asarray(video.savgol_z_f[[neuron.index for neuron in active_neurons]], dtype=float)
        light_evoked_traces = np.asarray(
            video.norm_sm_f[[neuron.index for neuron in active_neurons]],
            dtype=float,
        )
        dtw_traces = np.asarray(video.suite2p_data["F"][[neuron.index for neuron in active_neurons]], dtype=float)

        spike_trains = []
        for neuron in active_neurons:
            times = sorted(
                spike.sm_f_idx / fs
                for spike in neuron.spikes
                if 0 <= spike.sm_f_idx < n_frames
            )
            spike_trains.append(np.asarray(times, dtype=np.float64))

        result: dict[str, Any] = {
            "all_neurons": all_neurons,
            "active_neurons": active_neurons,
            "traces": traces,
            "dtw_traces": dtw_traces,
            "light_evoked_traces": light_evoked_traces,
            "spike_trains": spike_trains,
            "t_stop": n_frames / fs,
            "neuron_indices": np.array([neuron.index for neuron in active_neurons]),
            "n_frames": n_frames,
            "fs": fs,
            "video_id": str(getattr(video, "video_id", "")),
            "schedule_overrides": {"5732L-5": [33, 65, 93, 116, 153, 192]},
        }

        return result

    def run(self, video: "Video", grouping_cfg: dict) -> Optional[GroupingReport]:
        """Run every configured strategy on *video* and store results in place.

        Returns ``None`` when fewer than two neurons are available; otherwise
        populates ``video.grouping_results`` / ``grouping_stats`` and returns a
        :class:`GroupingReport`.
        """
        if len(video.neurons) < 2:
            video.grouping_results = {}
            video.grouping_stats = pd.DataFrame()
            return None

        grouping_cfg = grouping_cfg.copy()
        strat_args = self._get_grouping_kwargs(video)

        results: dict[str, GroupingResult] = {}
        all_neurons = strat_args["all_neurons"]
        for name in self.strategies:
            entry = STRATEGY_REGISTRY.get(name)
            if entry is None:
                raise ValueError(
                    f"Unknown grouping strategy {name!r}. "
                    f"Available: {list(STRATEGY_REGISTRY)}"
                )
            cfg = grouping_cfg.get(name, {}) or {}

            if name == "combined":
                strat_args["corr_config"] = cfg.get("corr", {}) or {}
                strat_args["sttc_config"] = cfg.get("sttc", {}) or {}
                strat_args["cluster_config"] = cfg.get("cluster", {}) or {}
                raw = entry(**strat_args)
            else:
                raw = entry(cfg, **strat_args)

            groups = raw.get("groups", [])
            if groups and isinstance(groups[0], dict):
                groups = neuron_groups_from_dicts(groups, all_neurons, method=name)

            results[name] = GroupingResult(
                groups=groups,
                matrix=raw.get("matrix"),
                config_label=raw.get("config_label", name),
                metadata=raw.get("metadata", {}),
            )

        video.grouping_results = results
        names_run = [name for name in self.strategies if name in results]

        combined = build_combined_summary(results)
        video.grouping_stats = pd.DataFrame(combined) if combined else pd.DataFrame()

        return GroupingReport(
            strategies_run=names_run,
            n_groups={name: len(results[name].groups) for name in names_run},
        )
