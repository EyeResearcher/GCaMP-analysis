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
from gcamp_analysis.grouping_processing.strategies import STRATEGY_REGISTRY
from gcamp_analysis.grouping_processing.treatment_comparison import (
    run_treatment_comparison,
    TreatmentComparisonResult,
)
from utils.visualization import visualize_neuron_groups, plot_matrix_heatmap
from utils.visualization import (
    plot_delta_corr_vs_dispersion,
    plot_neuron_centroid_distances,
)

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.video import Video


# ── Result container ─────────────────────────────────────────────────


@dataclass(frozen=True)
class GroupingResult:
    """Output of a single grouping strategy."""

    groups: list[NeuronGroup] | list[dict]
    matrix: np.ndarray | None
    config_label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
#  DICT → NEURON-GROUP CONVERSION
# =====================================================================


def neuron_groups_from_dicts(
    group_dicts: List[dict],
    neurons: list,
    *,
    method: str = "combined",
) -> List[NeuronGroup]:
    """Convert plain group dicts (from ``build_groups_from_labels``) into
    ``NeuronGroup`` objects by looking up neurons by index.

    Each dict must have ``group_id`` and ``neuron_indices`` keys.
    """
    idx_to_neuron = {n.index: n for n in neurons}
    groups: List[NeuronGroup] = []
    for gd in group_dicts:
        group_neurons = [idx_to_neuron[i] for i in gd["neuron_indices"] if i in idx_to_neuron]
        if not group_neurons:
            continue
        groups.append(
            NeuronGroup(
                group_id=gd["group_id"],
                neurons=group_neurons,
                method=method,
                row_indices=gd.get("row_indices"),
            )
        )
    return groups


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
#  COMBINED SUMMARY
# =====================================================================


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

    strategies: list[str] = field(default_factory=lambda: ["combined"])

    def _get_grouping_kwargs(self, video):
        all_neurons = list(getattr(video, "neurons", []))
        active_neurons = [
            n for n in all_neurons
            if getattr(getattr(n, "roi", None), "active_segments", {}).get("baseline", True)
        ]

        is_concat = bool(getattr(video, "is_concatenated", False))
        split_frame = getattr(video, "split_frame", None)
        n_frames = getattr(video, "n_frames", None)
        fs = float(getattr(video, "fs", 15.0))
        baseline_source = video.section_traces.get("baseline", {}) if is_concat else {}
        baseline_savgol = baseline_source.get("savgol_z_f", getattr(video, "baseline_savgol_z_f", None))
        baseline_norm_sm = baseline_source.get("norm_sm_f", getattr(video, "baseline_norm_sm_f", None))
        if baseline_savgol is None or np.asarray(baseline_savgol).size == 0:
            baseline_savgol = video.savgol_z_f
        if baseline_norm_sm is None or np.asarray(baseline_norm_sm).size == 0:
            baseline_norm_sm = video.norm_sm_f

        dtw_source = video.suite2p_data["F"][:, video.baseline_slice] if is_concat else video.suite2p_data["F"]
        traces = np.asarray(baseline_savgol[[n.index for n in active_neurons]], dtype=float)
        light_evoked_traces = np.asarray(baseline_norm_sm[[n.index for n in active_neurons]], dtype=float)
        dtw_traces = np.asarray(dtw_source[[n.index for n in active_neurons]], dtype=float)
        max_frame = video.baseline_n_frames if is_concat and video.baseline_n_frames else n_frames

        # Spike trains as sorted time arrays (seconds) for active neurons
        spike_trains = []
        for n in active_neurons:
            times = sorted(
                s.sm_f_idx / fs for s in n.spikes
                if 0 <= s.sm_f_idx < max_frame
            )
            spike_trains.append(np.asarray(times, dtype=np.float64))

        t_stop = max_frame / fs

        # Map from trimmed index (row in traces / spike_trains) → original neuron index
        neuron_indices = np.array([n.index for n in active_neurons])

        result = {
            "all_neurons": all_neurons,
            "active_neurons": active_neurons,
            "traces": traces,
            "dtw_traces": dtw_traces,
            "light_evoked_traces": light_evoked_traces,
            "spike_trains": spike_trains,
            "t_stop": t_stop,
            "neuron_indices": neuron_indices,
            "is_concatenated": is_concat,
            "split_frame": split_frame,
            "n_frames": n_frames,
            "fs": fs,
            "video_id": str(getattr(video, "video_id", "")),
            "schedule_overrides": {"5732L-5": [33, 65, 93, 116, 153, 192]},
        }

        # Treatment data for concatenated videos (same active neurons)
        treatment_section = video.get_legacy_treatment_section() if is_concat else None
        treatment_source = (
            video.section_traces.get(treatment_section.attribute_name, {})
            if treatment_section is not None
            else {}
        )
        treatment_savgol = treatment_source.get("savgol_z_f", getattr(video, "treatment_savgol_z_f", None))

        if is_concat and split_frame is not None and treatment_savgol is not None and np.asarray(treatment_savgol).size:
            result["tx_traces"] = np.asarray(treatment_savgol[[n.index for n in active_neurons]], dtype=float)

            tx_spike_trains = []
            for n in active_neurons:
                times = sorted(
                    (s.sm_f_idx - split_frame) / fs
                    for s in n.spikes
                    if split_frame <= s.sm_f_idx < n_frames
                )
                tx_spike_trains.append(np.asarray(times, dtype=np.float64))
            result["tx_spike_trains"] = tx_spike_trains
            result["tx_t_stop"] = (n_frames - split_frame) / fs

        return result

    def run(self, video: "Video", grouping_cfg: dict) -> Optional[GroupingReport]:
        if len(video.neurons) < 2:
            video.grouping_results = {}
            video.grouping_stats = pd.DataFrame()
            return None
        grouping_cfg = grouping_cfg.copy() 
        strat_args = self._get_grouping_kwargs(video)

        # ── Run each enabled strategy ────────────────────────────────
        results: dict[str, GroupingResult] = {}
        all_neurons = strat_args["all_neurons"]

        for name in self.strategies:
            entry  = STRATEGY_REGISTRY.get(name)
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

            # Convert dict-based groups to NeuronGroup objects
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
        names_run = [n for n in self.strategies if n in results]

        # ── Treatment comparison (concatenated mode) ─────────────────
        if (
            video.is_concatenated
            and video.split_frame is not None
            and "tx_traces" in strat_args
            and "tx_spike_trains" in strat_args
            and "tx_t_stop" in strat_args
        ):
            tc_results: dict = {}
            for name in names_run:
                result = results[name]
                if not result.groups:
                    continue
                cfg = grouping_cfg.get(name, {}) or {}

                if name == "combined":
                    tc_raw = run_treatment_comparison(
                        strat_args["tx_traces"],
                        strat_args["tx_spike_trains"],
                        strat_args["tx_t_stop"],
                        strat_args["neuron_indices"],
                        result.groups,
                        result.matrix,
                        corr_config=cfg.get("corr", {}) or {},
                        sttc_config=cfg.get("sttc", {}) or {},
                        cluster_config=cfg.get("cluster", {}) or {},
                    )
                    tc_results[name] = TreatmentComparisonResult(
                        strategy_name=name,
                        group_metrics=tc_raw["group_metrics"],
                        treatment_matrix=tc_raw["treatment_matrix"],
                        subgroups=tc_raw.get("subgroups", {}),
                    )
            video.treatment_comparison_results = tc_results

        # ── Summary stats ────────────────────────────────────────────
        combined = build_combined_summary(results)
        video.grouping_stats = pd.DataFrame(combined) if combined else pd.DataFrame()

        return GroupingReport(
            strategies_run=names_run,
            n_groups={n: len(results[n].groups) for n in names_run},
        )
