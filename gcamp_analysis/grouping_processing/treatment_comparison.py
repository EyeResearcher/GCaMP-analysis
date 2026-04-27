"""Baseline-vs-section comparison analytics for concatenated videos."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import pdist

from gcamp_analysis.data_classes.neuron_group import NeuronGroup
from gcamp_analysis.grouping_processing.similarity import (
    compute_combined_similarities,
    compute_sttc_matrix,
    max_crosscorr_similarity,
)
from gcamp_analysis.grouping_processing.clustering import (
    build_groups_from_labels,
    cluster_hierarchical,
)

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.neuron import Neuron


@dataclass(frozen=True)
class SectionComparisonResult:
    """Output of one baseline-vs-section comparison for one strategy."""

    strategy_name: str
    section_key: str
    section_kind: str
    group_metrics: List[Dict[str, Any]]
    section_matrix: Optional[np.ndarray]
    subgroups: Dict[str, List[NeuronGroup]] = field(default_factory=dict)


def _get_centroids(neurons) -> np.ndarray:
    coords = []
    for neuron in neurons:
        med = neuron.stats.get("med") if neuron.stats else None
        coords.append(med if med is not None else [np.nan, np.nan])
    return np.asarray(coords, dtype=float)


def _mean_pairwise_dist(centroids: np.ndarray) -> float:
    valid = centroids[~np.isnan(centroids).any(axis=1)]
    if len(valid) < 2:
        return np.nan
    return float(np.mean(pdist(valid, metric="euclidean")))


def _group_centroid(centroids: np.ndarray) -> np.ndarray:
    return np.nanmean(centroids, axis=0)


def _mean_dist_from_point(centroids: np.ndarray, point: np.ndarray) -> float:
    valid = centroids[~np.isnan(centroids).any(axis=1)]
    if len(valid) == 0:
        return np.nan
    dists = np.linalg.norm(valid - point, axis=1)
    return float(np.mean(dists))


def _build_neuron_detail(
    neurons,
    baseline_centroid: np.ndarray,
    neuron_to_subgroup: Dict[int, tuple],
    inactive_set: set[int],
) -> List[Dict[str, Any]]:
    detail = []
    for neuron in neurons:
        centroid = _get_centroids([neuron])[0]
        baseline_distance = (
            float(np.linalg.norm(centroid - baseline_centroid))
            if np.all(np.isfinite(centroid)) else np.nan
        )
        subgroup_info = neuron_to_subgroup.get(neuron.index)
        if subgroup_info is not None:
            _, subgroup_centroid = subgroup_info
            section_distance = (
                float(np.linalg.norm(centroid - subgroup_centroid))
                if np.all(np.isfinite(centroid)) else np.nan
            )
            status = "grouped"
            subgroup_id = subgroup_info[0]
        elif neuron.index in inactive_set:
            section_distance = np.nan
            status = "inactive"
            subgroup_id = None
        else:
            section_distance = np.nan
            status = "ungrouped"
            subgroup_id = None

        detail.append(
            {
                "neuron_index": neuron.index,
                "centroid_y": float(centroid[0]),
                "centroid_x": float(centroid[1]),
                "dist_from_baseline_centroid": baseline_distance,
                "dist_from_section_centroid": section_distance,
                "section_status": status,
                "section_subgroup_id": subgroup_id,
            }
        )
    return detail


def _mean_upper_tri_rows(matrix: np.ndarray | None, rows: list[int]) -> float:
    if matrix is None or len(rows) < 2:
        return np.nan
    sub_matrix = matrix[np.ix_(rows, rows)]
    tri = sub_matrix[np.triu_indices(len(rows), k=1)]
    return float(np.nanmean(tri))


MetricFn = Callable[[NeuronGroup, list[int], np.ndarray, np.ndarray, dict], Dict[str, Any]]


def _delta_mean_correlation(
    group: NeuronGroup,
    group_rows: list[int],
    baseline_matrix: np.ndarray,
    section_matrix: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    baseline_corr = _mean_upper_tri_rows(baseline_matrix, group_rows)
    section_corr = _mean_upper_tri_rows(section_matrix, group_rows)
    return {
        "baseline_mean_corr": baseline_corr,
        "section_mean_corr": section_corr,
        "delta_mean_corr": (
            section_corr - baseline_corr
            if np.isfinite(baseline_corr) and np.isfinite(section_corr)
            else np.nan
        ),
    }


def _frac_pairs_above_threshold(
    group: NeuronGroup,
    group_rows: list[int],
    baseline_matrix: np.ndarray,
    section_matrix: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    cluster_param = float(config.get("cluster_param", 0.65))
    similarity_threshold = 1.0 - cluster_param

    if len(group_rows) < 2 or section_matrix is None:
        return {"frac_pairs_above_thresh": np.nan}

    sub_matrix = section_matrix[np.ix_(group_rows, group_rows)]
    tri = sub_matrix[np.triu_indices(len(group_rows), k=1)]
    if tri.size == 0:
        return {"frac_pairs_above_thresh": np.nan}

    return {"frac_pairs_above_thresh": float(np.mean(tri >= similarity_threshold))}


def _section_coherence(
    group: NeuronGroup,
    group_rows: list[int],
    baseline_matrix: np.ndarray,
    section_matrix: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    return {"section_coherence": _mean_upper_tri_rows(section_matrix, group_rows)}


def _baseline_spatial_dispersion(
    group: NeuronGroup,
    group_rows: list[int],
    baseline_matrix: np.ndarray,
    section_matrix: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    centroids = _get_centroids(group.neurons)
    centroid = _group_centroid(centroids)
    return {
        "baseline_centroid_y": float(centroid[0]) if np.isfinite(centroid[0]) else np.nan,
        "baseline_centroid_x": float(centroid[1]) if np.isfinite(centroid[1]) else np.nan,
        "baseline_mean_pairwise_dist": _mean_pairwise_dist(centroids),
    }


METRIC_REGISTRY: List[MetricFn] = [
    _delta_mean_correlation,
    _frac_pairs_above_threshold,
    _section_coherence,
    _baseline_spatial_dispersion,
]


def _subgroup_mean_similarities(
    subgroups: List[NeuronGroup],
    section_matrix: np.ndarray,
    index_to_row: Dict[int, int],
) -> tuple[list[float], float]:
    if not subgroups:
        return [], np.nan

    similarities = []
    for subgroup in subgroups:
        rows = [index_to_row[neuron.index] for neuron in subgroup.neurons if neuron.index in index_to_row]
        similarities.append(_mean_upper_tri_rows(section_matrix, rows))

    sizes = np.array([subgroup.size for subgroup in subgroups], dtype=float)
    values = np.array(similarities, dtype=float)
    finite = np.isfinite(values)
    weighted_mean = float(np.average(values[finite], weights=sizes[finite])) if finite.any() else np.nan
    return similarities, weighted_mean


def _recluster_group(
    group: NeuronGroup,
    active_neurons: list,
    section_matrix: np.ndarray,
    index_to_row: Dict[int, int],
    baseline_centroid: np.ndarray,
    baseline_mean_pairwise_dist: float,
    *,
    cluster_param: float = 0.65,
    linkage_method: str = "average",
    cluster_criterion: str = "distance",
    min_group_size: int = 2,
) -> tuple[Dict[str, Any], List[NeuronGroup], Dict[int, tuple]]:
    active_rows = [index_to_row[neuron.index] for neuron in active_neurons if neuron.index in index_to_row]
    if len(active_rows) < min_group_size:
        return {}, [], {}

    section_submatrix = section_matrix[np.ix_(active_rows, active_rows)]
    _z, labels, _order, _matrix_ordered, _labels_ordered = cluster_hierarchical(
        section_submatrix,
        linkage_method=linkage_method,
        cluster_criterion=cluster_criterion,
        cluster_param=cluster_param,
    )

    active_neurons_in_row = [neuron for neuron in active_neurons if neuron.index in index_to_row]
    subgroup_row_indices = np.arange(len(active_neurons_in_row))
    subgroup_neuron_indices = np.array([neuron.index for neuron in active_neurons_in_row])
    subgroup_dicts = build_groups_from_labels(
        labels,
        subgroup_row_indices,
        subgroup_neuron_indices,
        min_group_size=min_group_size,
    )

    neuron_lookup = {neuron.index: neuron for neuron in active_neurons}
    subgroups: List[NeuronGroup] = []
    for subgroup_dict in subgroup_dicts:
        subgroup_neurons = [
            neuron_lookup[index]
            for index in subgroup_dict["neuron_indices"]
            if index in neuron_lookup
        ]
        if not subgroup_neurons:
            continue
        subgroups.append(
            NeuronGroup(
                group_id=f"{group.group_id}_sub{subgroup_dict['group_id']}",
                neurons=subgroup_neurons,
                method="combined",
                parent_group_id=group.group_id,
            )
        )

    updates: Dict[str, Any] = {
        "n_section_subgroups": len(subgroups),
        "section_subgroup_sizes": [subgroup.size for subgroup in subgroups],
        "subgroup_neuron_indices": [list(subgroup.neuron_indices) for subgroup in subgroups],
    }

    subgroup_corrs, subgroup_mean_corr = _subgroup_mean_similarities(
        subgroups,
        section_matrix,
        index_to_row,
    )
    updates["subgroup_mean_corrs"] = subgroup_corrs
    updates["section_subgroup_mean_corr"] = subgroup_mean_corr

    subgroup_mean_pairwise_dists = []
    subgroup_distances = []
    subgroup_dispersion_ratios = []
    for subgroup in subgroups:
        subgroup_centroids = _get_centroids(subgroup.neurons)
        subgroup_mpd = _mean_pairwise_dist(subgroup_centroids)
        subgroup_distance = _mean_dist_from_point(subgroup_centroids, baseline_centroid)
        subgroup_ratio = (
            subgroup_mpd / baseline_mean_pairwise_dist
            if np.isfinite(baseline_mean_pairwise_dist) and baseline_mean_pairwise_dist > 0
            else np.nan
        )
        subgroup_mean_pairwise_dists.append(subgroup_mpd)
        subgroup_distances.append(subgroup_distance)
        subgroup_dispersion_ratios.append(subgroup_ratio)

    updates["subgroup_mean_pairwise_dists"] = subgroup_mean_pairwise_dists
    updates["subgroup_dists_from_baseline_centroid"] = subgroup_distances
    updates["subgroup_dispersion_ratios"] = subgroup_dispersion_ratios

    grouped_indices = {neuron.index for subgroup in subgroups for neuron in subgroup.neurons}
    ungrouped = [neuron for neuron in group.neurons if neuron.index not in grouped_indices]
    updates["n_ungrouped"] = len(ungrouped)
    if ungrouped:
        updates["ungrouped_dist_from_baseline_centroid"] = _mean_dist_from_point(
            _get_centroids(ungrouped),
            baseline_centroid,
        )

    neuron_to_subgroup: Dict[int, tuple] = {}
    for subgroup in subgroups:
        subgroup_centroid = _group_centroid(_get_centroids(subgroup.neurons))
        for neuron in subgroup.neurons:
            neuron_to_subgroup[neuron.index] = (subgroup.group_id, subgroup_centroid)

    return updates, subgroups, neuron_to_subgroup


def run_section_comparison(
    section_traces: np.ndarray,
    section_spike_trains: list,
    section_t_stop: float,
    neuron_indices: np.ndarray,
    baseline_groups: list,
    baseline_matrix: np.ndarray,
    *,
    section_key: str,
    section_kind: str,
    corr_config: Dict[str, Any] | None = None,
    sttc_config: Dict[str, Any] | None = None,
    cluster_config: Dict[str, Any] | None = None,
    **kwargs,
) -> dict:
    """Compare baseline groups against one non-baseline section."""
    corr_config = corr_config or {}
    sttc_config = sttc_config or {}
    cluster_config = cluster_config or {}

    max_lag = int(corr_config.get("max_lag", 5))
    dt = float(sttc_config.get("dt", 1.75))

    section_corr = max_crosscorr_similarity(section_traces, max_lag=max_lag)
    section_sttc = compute_sttc_matrix(section_spike_trains, dt, 0.0, section_t_stop)
    section_matrix = compute_combined_similarities(section_corr, section_sttc)

    index_to_row = {int(index): row for row, index in enumerate(neuron_indices)}

    cluster_param = float(cluster_config.get("cluster_param", 0.65))
    linkage_method = str(cluster_config.get("linkage_method", "average"))
    cluster_criterion = str(cluster_config.get("cluster_criterion", "distance"))
    min_group_size = int(cluster_config.get("min_group_size", 2))

    all_group_metrics: List[Dict[str, Any]] = []
    subgroups: Dict[str, List[NeuronGroup]] = {}

    for group in baseline_groups:
        group_rows = [index_to_row[index] for index in group.neuron_indices if index in index_to_row]
        row: Dict[str, Any] = {
            "group_id": group.group_id,
            "n_neurons": group.size,
            "neuron_indices": list(group.neuron_indices),
            "section_key": section_key,
            "section_kind": section_kind,
        }

        for metric_fn in METRIC_REGISTRY:
            row.update(metric_fn(group, group_rows, baseline_matrix, section_matrix, cluster_config))

        active_neurons = [
            neuron
            for neuron in group.neurons
            if getattr(getattr(neuron, "roi", None), "active_segments", {}).get(section_key, True)
        ]
        inactive = [neuron for neuron in group.neurons if neuron not in active_neurons]
        inactive_set = {neuron.index for neuron in inactive}

        baseline_centroids = _get_centroids(group.neurons)
        baseline_centroid = _group_centroid(baseline_centroids)

        row["n_section_active"] = len(active_neurons)
        row["n_section_inactive"] = len(inactive)
        row.update(
            {
                "n_section_subgroups": 0,
                "section_subgroup_sizes": [],
                "subgroup_neuron_indices": [],
                "subgroup_mean_pairwise_dists": [],
                "subgroup_dists_from_baseline_centroid": [],
                "subgroup_dispersion_ratios": [],
                "subgroup_mean_corrs": [],
                "section_subgroup_mean_corr": np.nan,
                "n_ungrouped": 0,
                "ungrouped_dist_from_baseline_centroid": np.nan,
            }
        )

        baseline_mpd = row.get("baseline_mean_pairwise_dist", np.nan)
        subgroup_updates, group_subgroups, neuron_to_subgroup = _recluster_group(
            group,
            active_neurons,
            section_matrix,
            index_to_row,
            baseline_centroid,
            baseline_mpd,
            cluster_param=cluster_param,
            linkage_method=linkage_method,
            cluster_criterion=cluster_criterion,
            min_group_size=min_group_size,
        )
        row.update(subgroup_updates)
        if group_subgroups:
            subgroups[group.group_id] = group_subgroups

        row["neuron_spatial_detail"] = _build_neuron_detail(
            group.neurons,
            baseline_centroid,
            neuron_to_subgroup,
            inactive_set,
        )
        all_group_metrics.append(row)

    return {
        "group_metrics": all_group_metrics,
        "section_matrix": section_matrix,
        "subgroups": subgroups,
        "metadata": {
            "section_corr_matrix": section_corr,
            "section_sttc_matrix": section_sttc,
        },
    }


TreatmentComparisonResult = SectionComparisonResult
run_treatment_comparison = run_section_comparison
