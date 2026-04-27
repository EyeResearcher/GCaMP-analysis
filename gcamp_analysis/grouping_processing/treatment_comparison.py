"""Treatment comparison analytics for concatenated (baseline+treatment) videos.

Given baseline-only grouping results and treatment-half traces/spikes,
this module computes per-group metrics that quantify how each functional
group changed after treatment — and optionally re-clusters group members
on the treatment segment.

The public entry point ``run_treatment_comparison`` mirrors
``run_combined_grouping``: it takes pre-processed numerical data and
returns a plain dict.  The service layer wraps the dict into a
``TreatmentComparisonResult`` dataclass.

Metric functions are registered in ``METRIC_REGISTRY`` so that new
comparison metrics can be added by simply writing a function and
appending it to the list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import pdist

from gcamp_analysis.data_classes.neuron_group import NeuronGroup
from gcamp_analysis.grouping_processing.similarity import (
    max_crosscorr_similarity,
    compute_sttc_matrix,
    compute_combined_similarities,
)
from gcamp_analysis.grouping_processing.clustering import (
    cluster_hierarchical,
    build_groups_from_labels,
)

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.neuron import Neuron
    from gcamp_analysis.data_classes.video import Video


# =====================================================================
#  Result container
# =====================================================================


@dataclass(frozen=True)
class TreatmentComparisonResult:
    """Output of treatment comparison for one grouping strategy."""

    strategy_name: str

    # Per-group metric dicts (one dict per baseline group)
    group_metrics: List[Dict[str, Any]]

    # Full treatment-half similarity matrix (same shape as baseline matrix)
    treatment_matrix: Optional[np.ndarray]

    # Sub-groups found by re-clustering within each baseline group on
    # treatment traces.  Keyed by baseline group_id.
    subgroups: Dict[str, List[NeuronGroup]] = field(default_factory=dict)


# =====================================================================
#  Spatial helpers
# =====================================================================


def _get_centroids(neurons) -> np.ndarray:
    """Return (N, 2) array of [y, x] centroids from Suite2p *med* stat."""
    coords = []
    for n in neurons:
        med = n.stats.get("med") if n.stats else None
        if med is not None:
            coords.append(med)
        else:
            coords.append([np.nan, np.nan])
    return np.asarray(coords, dtype=float)


def _mean_pairwise_dist(centroids: np.ndarray) -> float:
    """Mean Euclidean distance between all pairs of centroids."""
    valid = centroids[~np.isnan(centroids).any(axis=1)]
    if len(valid) < 2:
        return np.nan
    return float(np.mean(pdist(valid, metric="euclidean")))


def _group_centroid(centroids: np.ndarray) -> np.ndarray:
    """Center of mass (mean position) of centroids, ignoring NaNs."""
    return np.nanmean(centroids, axis=0)


def _mean_dist_from_point(centroids: np.ndarray, point: np.ndarray) -> float:
    """Mean Euclidean distance of each centroid from a reference point."""
    valid = centroids[~np.isnan(centroids).any(axis=1)]
    if len(valid) == 0:
        return np.nan
    dists = np.linalg.norm(valid - point, axis=1)
    return float(np.mean(dists))


def _build_neuron_detail(
    neurons,
    bl_centroid: np.ndarray,
    neuron_to_sub: Dict[int, tuple],
    inactive_set: set,
) -> List[Dict[str, Any]]:
    """Per-neuron spatial detail relative to baseline and subgroup centroids."""
    detail = []
    for n in neurons:
        c = _get_centroids([n])[0]
        d_bl = float(np.linalg.norm(c - bl_centroid)) if np.all(np.isfinite(c)) else np.nan
        sub_info = neuron_to_sub.get(n.index)
        if sub_info is not None:
            _, sg_c = sub_info
            d_tx = float(np.linalg.norm(c - sg_c)) if np.all(np.isfinite(c)) else np.nan
            status = "grouped"
            sg_id = sub_info[0]
        elif n.index in inactive_set:
            d_tx = np.nan
            status = "inactive"
            sg_id = None
        else:
            d_tx = np.nan
            status = "ungrouped"
            sg_id = None
        detail.append({
            "neuron_index": n.index,
            "centroid_y": float(c[0]),
            "centroid_x": float(c[1]),
            "dist_from_baseline_centroid": d_bl,
            "dist_from_treatment_centroid": d_tx,
            "treatment_status": status,
            "treatment_subgroup_id": sg_id,
        })
    return detail


# =====================================================================
#  Matrix helpers
# =====================================================================


def _mean_upper_tri_rows(matrix: np.ndarray | None, rows: list) -> float:
    """Mean of upper-triangle entries for the sub-matrix at *rows*."""
    if matrix is None or len(rows) < 2:
        return np.nan
    sub = matrix[np.ix_(rows, rows)]
    tri = sub[np.triu_indices(len(rows), k=1)]
    return float(np.nanmean(tri))


# =====================================================================
#  Metric functions  (group, group_rows, bl_matrix, tx_matrix, config) -> dict
# =====================================================================

MetricFn = Callable[[NeuronGroup, list, np.ndarray, np.ndarray, dict], Dict[str, Any]]


def _delta_mean_correlation(
    group: NeuronGroup,
    group_rows: list,
    bl_matrix: np.ndarray,
    tx_matrix: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Change in mean intra-group similarity from baseline to treatment."""
    bl_corr = _mean_upper_tri_rows(bl_matrix, group_rows)
    tx_corr = _mean_upper_tri_rows(tx_matrix, group_rows)
    return {
        "baseline_mean_corr": bl_corr,
        "treatment_mean_corr": tx_corr,
        "delta_mean_corr": tx_corr - bl_corr if np.isfinite(bl_corr) and np.isfinite(tx_corr) else np.nan,
    }


def _frac_pairs_above_threshold(
    group: NeuronGroup,
    group_rows: list,
    bl_matrix: np.ndarray,
    tx_matrix: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Fraction of within-group neuron pairs whose treatment similarity
    still exceeds the clustering threshold used at baseline."""
    cluster_param = float(config.get("cluster_param", 0.65))
    sim_thresh = 1.0 - cluster_param

    if len(group_rows) < 2 or tx_matrix is None:
        return {"frac_pairs_above_thresh": np.nan}

    sub = tx_matrix[np.ix_(group_rows, group_rows)]
    tri = sub[np.triu_indices(len(group_rows), k=1)]
    if tri.size == 0:
        return {"frac_pairs_above_thresh": np.nan}

    frac = float(np.mean(tri >= sim_thresh))
    return {"frac_pairs_above_thresh": frac}


def _treatment_coherence(
    group: NeuronGroup,
    group_rows: list,
    bl_matrix: np.ndarray,
    tx_matrix: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Mean intra-group similarity on treatment traces alone."""
    tx_corr = _mean_upper_tri_rows(tx_matrix, group_rows)
    return {"treatment_coherence": tx_corr}


def _baseline_spatial_dispersion(
    group: NeuronGroup,
    group_rows: list,
    bl_matrix: np.ndarray,
    tx_matrix: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Baseline group spatial dispersion: centroid and mean pairwise distance."""
    centroids = _get_centroids(group.neurons)
    centroid = _group_centroid(centroids)
    mpd = _mean_pairwise_dist(centroids)
    return {
        "baseline_centroid_y": float(centroid[0]) if np.isfinite(centroid[0]) else np.nan,
        "baseline_centroid_x": float(centroid[1]) if np.isfinite(centroid[1]) else np.nan,
        "baseline_mean_pairwise_dist": mpd,
    }


# Registry — append new metric functions here
METRIC_REGISTRY: List[MetricFn] = [
    _delta_mean_correlation,
    _frac_pairs_above_threshold,
    _treatment_coherence,
    _baseline_spatial_dispersion,
]


def _subgroup_mean_similarities(
    subs: List[NeuronGroup],
    tx_matrix: np.ndarray,
    idx_to_row: Dict[int, int],
) -> tuple:
    """Per-subgroup and size-weighted mean intra-subgroup similarity."""
    if not subs:
        return [], np.nan
    corrs = []
    for sg in subs:
        rows = [idx_to_row[n.index] for n in sg.neurons if n.index in idx_to_row]
        corrs.append(_mean_upper_tri_rows(tx_matrix, rows))
    sizes = np.array([sg.size for sg in subs], dtype=float)
    vals = np.array(corrs)
    finite = np.isfinite(vals)
    mean = float(np.average(vals[finite], weights=sizes[finite])) if finite.any() else np.nan
    return corrs, mean


def _recluster_group(
    group: NeuronGroup,
    active_neurons: list,
    tx_matrix: np.ndarray,
    idx_to_row: Dict[int, int],
    bl_centroid: np.ndarray,
    bl_mpd: float,
    *,
    cluster_param: float = 0.65,
    linkage_method: str = "average",
    cluster_criterion: str = "distance",
    min_group_size: int = 2,
) -> tuple:
    """Re-cluster a baseline group on treatment combined matrix.

    Uses ``cluster_hierarchical`` + ``build_groups_from_labels``, matching
    the baseline grouping workflow.

    Returns ``(row_updates, subs, neuron_to_sub)``.
    """
    active_rows = [idx_to_row[n.index] for n in active_neurons if n.index in idx_to_row]
    if len(active_rows) < min_group_size:
        return {}, [], {}

    tx_sub = tx_matrix[np.ix_(active_rows, active_rows)]
    _Z, labels, _order, _mat_ordered, _labels_ordered = cluster_hierarchical(
        tx_sub,
        linkage_method=linkage_method,
        cluster_criterion=cluster_criterion,
        cluster_param=cluster_param,
    )

    # Map local sub-matrix position → neuron
    active_in_row = [n for n in active_neurons if n.index in idx_to_row]
    sub_row_idxs = np.arange(len(active_in_row))
    sub_neuron_idxs = np.array([n.index for n in active_in_row])
    sub_dicts = build_groups_from_labels(
        labels, sub_row_idxs, sub_neuron_idxs, min_group_size=min_group_size,
    )

    # Convert to NeuronGroups (needed for spatial metrics)
    idx_to_neuron = {n.index: n for n in active_neurons}
    subs: List[NeuronGroup] = []
    for sd in sub_dicts:
        sg_neurons = [idx_to_neuron[i] for i in sd["neuron_indices"] if i in idx_to_neuron]
        if not sg_neurons:
            continue
        subs.append(NeuronGroup(
            group_id=f"{group.group_id}_sub{sd['group_id']}",
            neurons=sg_neurons,
            method="combined",
            parent_group_id=group.group_id,
        ))

    updates: Dict[str, Any] = {
        "n_treatment_subgroups": len(subs),
        "treatment_subgroup_sizes": [sg.size for sg in subs],
        "subgroup_neuron_indices": [list(sg.neuron_indices) for sg in subs],
    }

    # Mean intra-subgroup similarity on treatment matrix
    corrs, mean_corr = _subgroup_mean_similarities(subs, tx_matrix, idx_to_row)
    updates["subgroup_mean_corrs"] = corrs
    updates["treatment_subgroup_mean_corr"] = mean_corr

    # Spatial dispersion per subgroup
    subgroup_mpds = []
    subgroup_dists = []
    subgroup_disp_ratios = []
    for sg in subs:
        sg_centroids = _get_centroids(sg.neurons)
        sg_mpd = _mean_pairwise_dist(sg_centroids)
        sg_dist = _mean_dist_from_point(sg_centroids, bl_centroid)
        sg_ratio = sg_mpd / bl_mpd if np.isfinite(bl_mpd) and bl_mpd > 0 else np.nan
        subgroup_mpds.append(sg_mpd)
        subgroup_dists.append(sg_dist)
        subgroup_disp_ratios.append(sg_ratio)

    updates["subgroup_mean_pairwise_dists"] = subgroup_mpds
    updates["subgroup_dists_from_baseline_centroid"] = subgroup_dists
    updates["subgroup_dispersion_ratios"] = subgroup_disp_ratios

    # Ungrouped neurons
    grouped_indices = {n.index for sg in subs for n in sg.neurons}
    ungrouped = [n for n in group.neurons if n.index not in grouped_indices]
    updates["n_ungrouped"] = len(ungrouped)
    if ungrouped:
        updates["ungrouped_dist_from_baseline_centroid"] = _mean_dist_from_point(
            _get_centroids(ungrouped), bl_centroid
        )

    # Neuron → subgroup lookup for spatial detail
    neuron_to_sub: Dict[int, tuple] = {}
    for sg in subs:
        sg_centroid = _group_centroid(_get_centroids(sg.neurons))
        for n in sg.neurons:
            neuron_to_sub[n.index] = (sg.group_id, sg_centroid)

    return updates, subs, neuron_to_sub



# =====================================================================
#  Public entry point
# =====================================================================


def run_treatment_comparison(
    tx_traces: np.ndarray,
    tx_spike_trains: list,
    tx_t_stop: float,
    neuron_indices: np.ndarray,
    baseline_groups: list,
    bl_matrix: np.ndarray,
    *,
    corr_config: Dict[str, Any] | None = None,
    sttc_config: Dict[str, Any] | None = None,
    cluster_config: Dict[str, Any] | None = None,
    **kwargs,
) -> dict:
    """Compare baseline groups against treatment-half data.

    Computes a treatment combined similarity matrix (corr × STTC),
    identically to the baseline ``run_combined_grouping``, then
    evaluates per-group metrics and optionally re-clusters.

    Parameters
    ----------
    tx_traces : (n_active, n_timepoints) array
        Treatment-segment traces for the same active neurons used at
        baseline (savgol z-scored).
    tx_spike_trains : list of array-like
        Sorted spike-time arrays (seconds) per active neuron in the
        treatment segment.
    tx_t_stop : float
        Treatment segment duration in seconds.
    neuron_indices : array of int
        Original neuron/ROI indices for each row (same as baseline).
    baseline_groups : list of NeuronGroup
        Groups found during baseline clustering.
    bl_matrix : (n_active, n_active) array
        Baseline combined similarity matrix.
    corr_config, sttc_config, cluster_config : dict, optional
        Per-component configuration overrides.

    Returns
    -------
    dict
        Keys: ``group_metrics``, ``treatment_matrix``, ``subgroups``,
        ``metadata``.
    """
    corr_config = corr_config or {}
    sttc_config = sttc_config or {}
    cluster_config = cluster_config or {}

    # ── Treatment combined matrix (identical to baseline computation) ──
    max_lag = int(corr_config.get("max_lag", 5))
    dt = float(sttc_config.get("dt", 1.75))

    tx_corr = max_crosscorr_similarity(tx_traces, max_lag=max_lag)
    tx_sttc = compute_sttc_matrix(tx_spike_trains, dt, 0.0, tx_t_stop)
    tx_matrix = compute_combined_similarities(tx_corr, tx_sttc)

    # ── Index mapping: neuron_index → matrix row ──
    idx_to_row = {int(idx): row for row, idx in enumerate(neuron_indices)}

    cluster_param = float(cluster_config.get("cluster_param", 0.65))
    linkage_method = str(cluster_config.get("linkage_method", "average"))
    cluster_criterion = str(cluster_config.get("cluster_criterion", "distance"))
    min_group_size = int(cluster_config.get("min_group_size", 2))

    # ── Per-group metrics ──
    all_group_metrics: List[Dict[str, Any]] = []
    subgroups: Dict[str, List[NeuronGroup]] = {}

    for group in baseline_groups:
        group_rows = [idx_to_row[i] for i in group.neuron_indices if i in idx_to_row]

        row: Dict[str, Any] = {
            "group_id": group.group_id,
            "n_neurons": group.size,
            "neuron_indices": list(group.neuron_indices),
        }

        for metric_fn in METRIC_REGISTRY:
            row.update(metric_fn(group, group_rows, bl_matrix, tx_matrix, cluster_config))

        all_group_metrics.append(row)

        # ── Re-cluster within group on treatment segment ──
        active_neurons = [
            n for n in group.neurons
            if getattr(getattr(n, "roi", None), "active_segments", {}).get("treatment", True)
        ]
        inactive = [n for n in group.neurons if n not in active_neurons]
        inactive_set = {n.index for n in inactive}

        bl_centroids = _get_centroids(group.neurons)
        bl_centroid = _group_centroid(bl_centroids)

        row["n_treatment_active"] = len(active_neurons)
        row["n_treatment_inactive"] = len(inactive)
        row.update({
            "n_treatment_subgroups": 0,
            "treatment_subgroup_sizes": [],
            "subgroup_neuron_indices": [],
            "subgroup_mean_pairwise_dists": [],
            "subgroup_dists_from_baseline_centroid": [],
            "subgroup_dispersion_ratios": [],
            "subgroup_mean_corrs": [],
            "treatment_subgroup_mean_corr": np.nan,
            "n_ungrouped": 0,
            "ungrouped_dist_from_baseline_centroid": np.nan,
        })

        bl_mpd = row.get("baseline_mean_pairwise_dist", np.nan)
        updates, subs, neuron_to_sub = _recluster_group(
            group, active_neurons, tx_matrix, idx_to_row,
            bl_centroid, bl_mpd,
            cluster_param=cluster_param,
            linkage_method=linkage_method,
            cluster_criterion=cluster_criterion,
            min_group_size=min_group_size,
        )
        row.update(updates)
        if subs:
            subgroups[group.group_id] = subs

        row["neuron_spatial_detail"] = _build_neuron_detail(
            group.neurons, bl_centroid, neuron_to_sub, inactive_set,
        )

    return {
        "group_metrics": all_group_metrics,
        "treatment_matrix": tx_matrix,
        "subgroups": subgroups,
        "metadata": {"tx_corr_matrix": tx_corr, "tx_sttc_matrix": tx_sttc},
    }
