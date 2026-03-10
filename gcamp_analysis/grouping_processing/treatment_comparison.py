"""Treatment comparison analytics for concatenated (baseline+treatment) videos.

Given baseline-only grouping results and treatment-half traces/spikes,
this module computes per-group metrics that quantify how each functional
group changed after treatment — and optionally re-clusters group members
on the treatment segment.

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
from gcamp_analysis.grouping_processing.similarity import compute_correlation_matrix
from gcamp_analysis.grouping_processing.clustering import recluster_within_group

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.neuron import Neuron
    from gcamp_analysis.data_classes.video import Video
    from gcamp_analysis.grouping_processing.strategies import GroupingResult


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
#  Metric functions  (signature: group, bl_matrix, tx_matrix, config) -> dict)
# =====================================================================

MetricFn = Callable[[NeuronGroup, np.ndarray, np.ndarray, dict], Dict[str, Any]]


def _delta_mean_correlation(
    group: NeuronGroup,
    bl_matrix: np.ndarray,
    tx_matrix: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Change in mean intra-group correlation from baseline to treatment."""
    bl_corr = group._mean_upper_tri(bl_matrix)
    tx_corr = group._mean_upper_tri(tx_matrix)
    return {
        "baseline_mean_corr": bl_corr,
        "treatment_mean_corr": tx_corr,
        "delta_mean_corr": tx_corr - bl_corr if np.isfinite(bl_corr) and np.isfinite(tx_corr) else np.nan,
    }


def _frac_pairs_above_threshold(
    group: NeuronGroup,
    bl_matrix: np.ndarray,
    tx_matrix: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Fraction of within-group neuron pairs whose treatment correlation
    still exceeds the clustering threshold used at baseline."""
    threshold = float(config.get("distance_threshold", 0.6))
    corr_thresh = 1.0 - threshold  # distance_threshold → correlation threshold

    idxs = group.filtered_idxs
    if len(idxs) < 2 or tx_matrix is None:
        return {"frac_pairs_above_thresh": np.nan}

    sub = tx_matrix[np.ix_(idxs, idxs)]
    tri = sub[np.triu_indices(len(idxs), k=1)]
    if tri.size == 0:
        return {"frac_pairs_above_thresh": np.nan}

    frac = float(np.mean(tri >= corr_thresh))
    return {"frac_pairs_above_thresh": frac}


def _treatment_coherence(
    group: NeuronGroup,
    bl_matrix: np.ndarray,
    tx_matrix: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Mean intra-group correlation on treatment traces alone."""
    tx_corr = group._mean_upper_tri(tx_matrix)
    return {"treatment_coherence": tx_corr}


def _baseline_spatial_dispersion(
    group: NeuronGroup,
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


def _recluster_group(
    group: NeuronGroup,
    active_neurons: list,
    tx_matrix: np.ndarray,
    bl_centroid: np.ndarray,
    bl_mpd: float,
    strategy_name: str,
    strategy_config: dict,
) -> tuple:
    """Re-cluster a baseline group on treatment traces and compute subgroup metrics.

    Returns ``(row_updates, subs, neuron_to_sub)`` where *row_updates* is
    a dict of metrics to merge into the per-group row, *subs* is the list
    of treatment subgroups, and *neuron_to_sub* maps neuron index to
    ``(subgroup_id, subgroup_centroid)``.
    """
    active_fidxs = [n.filtered_index for n in active_neurons]
    if len(active_fidxs) < 2:
        return {}, [], {}

    corr_sub = tx_matrix[np.ix_(active_fidxs, active_fidxs)]
    subs = recluster_within_group(
        active_neurons,
        corr_sub,
        parent_group_id=group.group_id,
        method=strategy_name,
        distance_threshold=float(strategy_config.get("distance_threshold", 0.6)),
        min_group_size=int(strategy_config.get("min_group_size", 2)),
        corr_config=strategy_config,
    )

    updates: Dict[str, Any] = {
        "n_treatment_subgroups": len(subs),
        "treatment_subgroup_sizes": [sg.size for sg in subs],
        "subgroup_neuron_indices": [
            list(getattr(sg, "neuron_indices", [])) for sg in subs
        ],
        "subgroup_filtered_idxs": [
            list(getattr(sg, "filtered_idxs", [])) for sg in subs
        ],
    }

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
#  Service entry point
# =====================================================================


class TreatmentComparisonService:
    """Compute per-group treatment comparison metrics and sub-clustering.

    Called by ``GroupingService`` after baseline grouping when the video
    is concatenated.
    """

    def run(
        self,
        video: "Video",
        baseline_result: "GroupingResult",
        strategy_name: str,
        strategy_config: dict,
    ) -> TreatmentComparisonResult:
        """Run all registered metrics and optional re-clustering.

        Parameters
        ----------
        video : Video
            Must have ``treatment_norm_sm_f`` populated.
        baseline_result : GroupingResult
            Groups and matrix from baseline-only clustering.
        strategy_name : str
            Name of the grouping strategy (e.g. ``"corr"``).
        strategy_config : dict
            Per-strategy config from the pipeline YAML.
        """
        # Build treatment-half correlation matrix using same neurons
        tx_traces_full = np.asarray(video.treatment_norm_sm_f, float)
        neuron_idxs = [n.index for n in video.neurons]
        tx_traces = tx_traces_full[neuron_idxs, :]

        tx_matrix = compute_correlation_matrix(
            tx_traces,
            method=str(strategy_config.get("method", "pearson")),
            remove_global=bool(strategy_config.get("remove_global", True)),
            use_diff=bool(strategy_config.get("use_diff", True)),
            diff_order=int(strategy_config.get("diff_order", 1)),
            zscore_each=bool(strategy_config.get("zscore_each", True)),
            clip_negatives=bool(strategy_config.get("clip_negatives", True)),
        )

        bl_matrix = baseline_result.matrix

        # --- Per-group metrics ---
        all_group_metrics: List[Dict[str, Any]] = []
        subgroups: Dict[str, List[NeuronGroup]] = {}

        for group in baseline_result.groups:
            row: Dict[str, Any] = {
                "group_id": group.group_id,
                "n_neurons": group.size,
                "neuron_indices": list(getattr(group, "neuron_indices", [])),
                "filtered_idxs": list(getattr(group, "filtered_idxs", [])),
            }

            for metric_fn in METRIC_REGISTRY:
                row.update(metric_fn(group, bl_matrix, tx_matrix, strategy_config))

            all_group_metrics.append(row)

            # --- Re-cluster within group on treatment traces ---
            # Only include neurons flagged as active during treatment
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
                "subgroup_filtered_idxs": [],
                "subgroup_mean_pairwise_dists": [],
                "subgroup_dists_from_baseline_centroid": [],
                "subgroup_dispersion_ratios": [],
                "n_ungrouped": 0,
                "ungrouped_dist_from_baseline_centroid": np.nan,
            })

            bl_mpd = row.get("baseline_mean_pairwise_dist", np.nan)
            updates, subs, neuron_to_sub = _recluster_group(
                group, active_neurons, tx_matrix, bl_centroid, bl_mpd,
                strategy_name, strategy_config,
            )
            row.update(updates)
            if subs:
                subgroups[group.group_id] = subs

            row["neuron_spatial_detail"] = _build_neuron_detail(
                group.neurons, bl_centroid, neuron_to_sub, inactive_set,
            )

        return TreatmentComparisonResult(
            strategy_name=strategy_name,
            group_metrics=all_group_metrics,
            treatment_matrix=tx_matrix,
            subgroups=subgroups,
        )
