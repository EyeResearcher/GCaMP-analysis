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

from gcamp_analysis.data_classes.neuron_group import NeuronGroup
from gcamp_analysis.grouping_processing.similarity import compute_correlation_matrix
from gcamp_analysis.grouping_processing.clustering import cluster

if TYPE_CHECKING:
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


# Registry — append new metric functions here
METRIC_REGISTRY: List[MetricFn] = [
    _delta_mean_correlation,
    _frac_pairs_above_threshold,
    _treatment_coherence,
]


# =====================================================================
#  Per-group re-clustering on treatment traces
# =====================================================================


def recluster_within_group(
    group: NeuronGroup,
    tx_traces: np.ndarray,
    *,
    method: str = "corr",
    distance_threshold: float = 0.6,
    min_group_size: int = 2,
    corr_config: Optional[dict] = None,
) -> List[NeuronGroup]:
    """Re-cluster a baseline group's members using treatment-half traces.

    Parameters
    ----------
    group : NeuronGroup
        Baseline group whose members to re-cluster.
    tx_traces : np.ndarray
        ``(N_group, T_treatment)`` trace matrix for the group members.
    method : str
        Clustering method used at baseline (``"corr"``, ``"sttc"``, etc.).
    distance_threshold : float
        Distance threshold for clustering.
    min_group_size : int
        Minimum cluster size.
    corr_config : dict, optional
        Correlation matrix parameters (remove_global, use_diff, …).

    Returns
    -------
    list[NeuronGroup]
        Sub-groups found within the treatment segment.  May be empty if
        no cluster meets ``min_group_size``.
    """
    if len(group.neurons) < min_group_size:
        return []

    cfg = corr_config or {}
    C = compute_correlation_matrix(
        tx_traces,
        method=cfg.get("method", "pearson"),
        remove_global=cfg.get("remove_global", True),
        use_diff=cfg.get("use_diff", True),
        diff_order=cfg.get("diff_order", 1),
        zscore_each=cfg.get("zscore_each", True),
        clip_negatives=cfg.get("clip_negatives", True),
    )

    sub_groups = cluster(
        group.neurons,
        1.0 - C,
        cluster_method=cfg.get("cluster", "hierarchical"),
        threshold=distance_threshold,
        min_group_size=min_group_size,
        method=method,
    )

    # Re-label sub-group IDs to reflect parent
    for i, sg in enumerate(sub_groups):
        sg.group_id = f"{group.group_id}_sub{i}"
        sg.metadata["parent_group_id"] = group.group_id

    return sub_groups


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
            # Extract treatment traces for this group's members
            group_member_indices = group.filtered_idxs
            if len(group_member_indices) >= 2:
                group_tx_traces = tx_traces[group_member_indices, :]
                subs = recluster_within_group(
                    group,
                    group_tx_traces,
                    method=strategy_name,
                    distance_threshold=float(strategy_config.get("distance_threshold", 0.6)),
                    min_group_size=int(strategy_config.get("min_group_size", 2)),
                    corr_config=strategy_config,
                )
                subgroups[group.group_id] = subs
                row["n_treatment_subgroups"] = len(subs)
                row["treatment_subgroup_sizes"] = [sg.size for sg in subs]
                row["subgroup_neuron_indices"] = [
                    list(getattr(sg, "neuron_indices", [])) for sg in subs
                ]
                row["subgroup_filtered_idxs"] = [
                    list(getattr(sg, "filtered_idxs", [])) for sg in subs
                ]
            else:
                row["n_treatment_subgroups"] = 0
                row["treatment_subgroup_sizes"] = []
                row["subgroup_neuron_indices"] = []
                row["subgroup_filtered_idxs"] = []

        return TreatmentComparisonResult(
            strategy_name=strategy_name,
            group_metrics=all_group_metrics,
            treatment_matrix=tx_matrix,
            subgroups=subgroups,
        )
