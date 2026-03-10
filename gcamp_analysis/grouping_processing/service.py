"""Grouping service: orchestration, comparison, summary, and visualization.

Houses the public ``GroupingService`` entry point plus the helper functions
that compare, summarise and visualise grouping results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gcamp_analysis.data_classes.neuron_group import NeuronGroup
from gcamp_analysis.reports import GroupingReport
from gcamp_analysis.grouping_processing.strategies import STRATEGY_REGISTRY, GroupingResult
from gcamp_analysis.grouping_processing.treatment_comparison import TreatmentComparisonService
from utils.visualization import visualize_neuron_groups, plot_matrix_heatmap
from utils.visualization import (
    plot_delta_corr_vs_dispersion,
    plot_neuron_centroid_distances,
)

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.video import Video


# =====================================================================
#  PER-GROUP SUMMARY
# =====================================================================


def compute_group_summary_rows(
    groups: List[NeuronGroup],
    *,
    method: str,
    matrices: Dict[str, Optional[np.ndarray]],
) -> List[Dict[str, Any]]:
    """Return one summary dict per group.

    Parameters
    ----------
    groups : list of NeuronGroup
    method : strategy name that produced these groups
    matrices : ``{strategy_name: matrix_or_None}`` for connectivity stats
    """
    rows: List[Dict[str, Any]] = []
    for g in groups:
        ss = [getattr(n, "summary_stats", {}) for n in g.neurons]
        df = pd.DataFrame(ss)

        rates = df.get("spike_frequency", pd.Series(dtype=float))
        num_spikes = df.get("number_of_spikes", pd.Series(dtype=float))
        mean_of_means = df.filter(like="mean_").mean(numeric_only=True).to_dict()

        row: Dict[str, Any] = {
            "group_id": g.group_id,
            "method": method,
            "number_neurons": int(g.size),
            "neuron_indices": list(getattr(g, "neuron_indices", [])),
            "filtered_idxs": list(getattr(g, "filtered_idxs", [])),
            "spike_rate": float(np.nanmean(rates)) if len(rates) else 0.0,
            "number_of_spikes": float(np.nanmean(num_spikes)) if len(num_spikes) else 0.0,
            **mean_of_means,
        }

        for mat_name, mat in matrices.items():
            try:
                row[f"mean_{mat_name}"] = g.group_mean_similarity(mat)
            except Exception:
                row[f"mean_{mat_name}"] = np.nan

        for key in ("t_win", "corr_thresh", "sttc_thresh", "dtw_thresh"):
            val = g.metadata.get(key)
            if val is not None:
                row[key] = val

        rows.append(row)
    return rows


# =====================================================================
#  PAIRWISE AGREEMENT / COMBINED SUMMARY
# =====================================================================


def compute_pairwise_agreement(
    results: Dict[str, GroupingResult],
    neurons: list,
) -> Dict[str, float]:
    """Membership agreement between every pair of strategies.

    Returns a dict like ``{"corr_vs_sttc": 0.85, ...}``.
    """
    if len(results) < 2:
        return {}

    memberships: Dict[str, np.ndarray] = {}
    for name, result in results.items():
        m = np.full(len(neurons), -1, dtype=int)
        for i, group in enumerate(result.groups):
            for neuron in group.neurons:
                if neuron in neurons:
                    m[neurons.index(neuron)] = i
        memberships[name] = m

    names = sorted(memberships)
    agreements: Dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            agreements[f"{a}_vs_{b}"] = float(np.mean(memberships[a] == memberships[b]))
    return agreements


def build_combined_summary(
    results: Dict[str, GroupingResult],
) -> List[dict]:
    """Collect per-group summary rows from all strategies."""
    matrices = {name: r.matrix for name, r in results.items()}
    all_rows: list[dict] = []
    for name, result in results.items():
        all_rows.extend(
            compute_group_summary_rows(result.groups, method=name, matrices=matrices)
        )
    return all_rows


# =====================================================================
#  VISUALIZATION
# =====================================================================


def _infer_img_size(video: "Video", default=(1024, 1024)) -> tuple[int, int]:
    ops = getattr(video, "suite2p_data", {}).get("ops", {}) if getattr(video, "suite2p_data", None) else {}
    Ly = int(ops.get("Ly", default[0]))
    Lx = int(ops.get("Lx", default[1]))
    return (Ly, Lx)


def make_matrix_heatmap(
    matrix: np.ndarray,
    *,
    title: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize=(6, 5),
) -> Optional[Figure]:
    """Create a heatmap figure for a similarity/distance matrix."""
    if matrix is None:
        return None
    m = np.asarray(matrix)
    if m.ndim != 2 or m.size == 0:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    plot_matrix_heatmap(m, title=title, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, show_colorbar=True)
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
    """Generate overlay + heatmap figures for a single grouping strategy.

    Parameters
    ----------
    video : Video
        Must have ``grouping_results`` populated.
    strategy_name : str
        Key into ``video.grouping_results``.

    Returns
    -------
    (overlay_fig, heatmap_fig)
    """
    result = video.grouping_results.get(strategy_name)
    if result is None:
        return None, None

    groups = result.groups
    matrix = result.matrix

    label = config_label or strategy_name
    heat_title = f"{strategy_name} matrix ({label})"

    # Overlay
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

    # Heatmap
    heatmap_fig = make_matrix_heatmap(
        matrix,
        title=heat_title,
        cmap=heatmap_cmap,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )

    return overlay_fig, heatmap_fig


def visualize_treatment_comparison(
    video: "Video",
    *,
    strategy_name: str = "corr",
) -> Tuple[Optional[Figure], Optional[Figure]]:
    """Generate spatial-dispersion figures for one treatment comparison.

    Returns ``(delta_corr_fig, centroid_dist_fig)`` or ``(None, None)``
    if no treatment comparison data exists for *strategy_name*.
    """
    tc_results = getattr(video, "treatment_comparison_results", {})
    tc_result = tc_results.get(strategy_name)
    if tc_result is None or not getattr(tc_result, "group_metrics", None):
        return None, None

    gm = tc_result.group_metrics
    label = getattr(video, "path", None)
    name = label.name if label else "video"

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    plot_delta_corr_vs_dispersion(gm, ax=ax1, title=f"{name} \u2014 {strategy_name}")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    plot_neuron_centroid_distances(gm, ax=ax2, title=f"{name} \u2014 {strategy_name}")
    fig2.tight_layout()

    return fig1, fig2


# =====================================================================
#  GROUPING SERVICE (entry point)
# =====================================================================


@dataclass
class GroupingService:
    """Run one or more grouping strategies and compare them.

    Parameters
    ----------
    strategies : list of strategy names (keys in ``STRATEGY_REGISTRY``).
        Default: ``["corr"]``.
    """

    strategies: list[str] = field(default_factory=lambda: ["corr"])

    def run(self, video: "Video", grouping_cfg: dict) -> Optional[GroupingReport]:
        if len(video.neurons) < 2:
            video.grouping_results = {}
            video.grouping_stats = pd.DataFrame()
            return None

        # ── Run each enabled strategy ────────────────────────────────
        results: dict[str, GroupingResult] = {}
        for name in self.strategies:
            cls = STRATEGY_REGISTRY.get(name)
            if cls is None:
                raise ValueError(
                    f"Unknown grouping strategy {name!r}. "
                    f"Available: {list(STRATEGY_REGISTRY)}"
                )
            cfg = grouping_cfg.get(name, {}) or {}
            results[name] = cls().compute(video, cfg)

        video.grouping_results = results
        names_run = [n for n in self.strategies if n in results]

        # ── Treatment comparison (concatenated mode) ─────────────────
        if video.is_concatenated and video.split_frame is not None:
            tc_service = TreatmentComparisonService()
            tc_results: dict = {}
            for name in names_run:
                result = results[name]
                if not result.groups:
                    continue
                cfg = grouping_cfg.get(name, {}) or {}
                tc_results[name] = tc_service.run(video, result, name, cfg)
            video.treatment_comparison_results = tc_results

        # ── Compare strategies ───────────────────────────────────────
        agreements = compute_pairwise_agreement(results, video.neurons)

        # ── Summary stats ────────────────────────────────────────────
        combined = build_combined_summary(results)
        video.grouping_stats = pd.DataFrame(combined) if combined else pd.DataFrame()

        return GroupingReport(
            strategies_run=names_run,
            n_groups={n: len(results[n].groups) for n in names_run},
            agreements=agreements,
        )
