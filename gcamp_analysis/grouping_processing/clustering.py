"""Clustering functions used by grouping strategies."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from gcamp_analysis.data_classes.neuron_group import NeuronGroup

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.neuron import Neuron


# ── Pure numerical clustering (no Neuron dependency) ─────────────────


def cluster_hierarchical(
    sim_mat: np.ndarray,
    linkage_method: str = "average",
    cluster_criterion: str = "distance",
    cluster_param: int | float = .65,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hierarchical clustering on a similarity matrix.

    Returns ``(Z, labels, order, mat_ordered, labels_ordered)``.
    """
    mat = np.nan_to_num(sim_mat.copy(), nan=0.0, posinf=0.0, neginf=0.0)

    distance = 1.0 - mat
    np.fill_diagonal(distance, 0.0)
    distance = 0.5 * (distance + distance.T)

    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method=linkage_method)

    labels = fcluster(Z, t=cluster_param, criterion=cluster_criterion)

    order = leaves_list(Z)
    mat_ordered = mat[order][:, order]
    labels_ordered = labels[order]

    return Z, labels, order, mat_ordered, labels_ordered

# Object Oriented Threshold Graph Clustering (Connected Components)
def cluster_threshold_graph(
    neurons: List["Neuron"],
    dist: np.ndarray,
    *,
    threshold: float,
    min_group_size: int = 2,
    method: str = "corr",
    **metadata,
) -> List[NeuronGroup]:
    """Connected-component clustering on a thresholded distance graph."""
    n = dist.shape[0]
    adj = (dist <= threshold) & np.isfinite(dist)
    np.fill_diagonal(adj, True)

    visited = np.zeros(n, dtype=bool)
    components: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack, comp = [i], []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where(adj[u])[0]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if len(comp) >= min_group_size:
            components.append(sorted(comp))

    return [
        NeuronGroup(group_id=k, neurons=[neurons[i] for i in idxs], method=method, **metadata)
        for k, idxs in enumerate(components, start=1)
    ]

# Agnostic Clustering Dispatcher
# OOP Dependently on cluster_threshold_graph
# Functional Dependently on cluster_hierarchical
def cluster(
    neurons: List["Neuron"],
    dist: np.ndarray,
    *,
    cluster_method: str = "hierarchical",
    **kwargs,
) -> List[NeuronGroup]:
    """Unified clustering dispatcher.

    Parameters
    ----------
    neurons : list of Neuron
        Neurons to cluster.
    dist : np.ndarray
        Pairwise distance matrix.
    cluster_method : str
        One of "graph" (connected-component) or "hierarchical" (agglomerative).
    **kwargs
        Passed to the underlying clustering function (threshold, min_group_size,
        linkage_method, method, group_id_prefix, etc.).

    Returns
    -------
    list of NeuronGroup
    """
    dispatch = {
        "graph": cluster_threshold_graph,
        "hierarchical": cluster_hierarchical,
    }
    if cluster_method not in dispatch.keys():
        raise ValueError(f"Unknown cluster_method '{cluster_method}'. Choose from {list(dispatch.keys())}")
    return dispatch[cluster_method](neurons, dist, **kwargs)

# OOP Within Group Re-Clustering 
def recluster_within_group(
    neurons: List["Neuron"],
    corr_sub: np.ndarray,
    *,
    parent_group_id: str = "",
    method: str = "corr",
    cluster_param: float = 0.6,
    min_group_size: int = 2,
    corr_config: Optional[dict] = None,
) -> List[NeuronGroup]:
    """Re-cluster treatment-active neurons from a baseline group.

    Parameters
    ----------
    neurons : list[Neuron]
        Treatment-active neurons to re-cluster (pre-filtered by caller).
    corr_sub : np.ndarray
        ``(N_neurons, N_neurons)`` correlation sub-matrix for *neurons*,
        extracted from the full treatment correlation matrix so that
        global-signal removal reflects the whole population — not just
        the small subset.
    parent_group_id : str
        Group ID of the parent baseline group (used to label sub-groups).
    method : str
        Clustering method used at baseline (``"corr"``, ``"sttc"``, etc.).
    cluster_param : float
        Parameter for clustering (distance threshold or number of clusters).
    min_group_size : int
        Minimum cluster size.
    corr_config : dict, optional
        Extra clustering parameters (cluster method, etc.).

    Returns
    -------
    list[NeuronGroup]
        Sub-groups found within the treatment segment.  May be empty if
        no cluster meets ``min_group_size``.
    """
    if len(neurons) < min_group_size:
        return []

    cfg = corr_config or {}

    sub_groups = cluster(
        neurons,
        1.0 - corr_sub,
        cluster_method=cfg.get("cluster", "hierarchical"),
        cluster_param=cfg.get("cluster_param", 0.6),
        min_group_size=min_group_size,
        method=method,
    )

    # Re-label sub-group IDs to reflect parent
    for i, sg in enumerate(sub_groups):
        sg.group_id = f"{parent_group_id}_sub{i}"
        sg.metadata["parent_group_id"] = parent_group_id

    return sub_groups

# OOP Light-Evoked Clustering (Schedule-Based)
def light_evoked_cluster(
    neurons: List["Neuron"],
    activated: np.ndarray,
    n_pulses: int,
    schedule: Optional[list[int]] = None,
    bin_size: int = 3,
    response_window: int = 10,
    **metadata,
) -> List[NeuronGroup]:
    # When schedule + spike objects are available, count per-pulse responses
    # from true spike detections. Otherwise fall back to the activated matrix
    # so lightweight tests and exploratory uses do not need full spike objects.
    n_neurons = len(neurons)
    on_counts = np.zeros(n_neurons, dtype=int)
    off_counts = np.zeros(n_neurons, dtype=int)
    use_spike_schedule = bool(schedule) and any(getattr(neuron, "spikes", None) for neuron in neurons)
    if use_spike_schedule:
        for i, neuron in enumerate(neurons):
            matched_pulses: set[int] = set()
            used_spikes: set[int] = set()
            for p_idx, pulse_frame in enumerate(schedule or []):
                best_si: int | None = None
                best_dist = float("inf")
                for si, spike in enumerate(neuron.spikes):
                    if si in used_spikes:
                        continue
                    dist = spike.sm_f_idx - pulse_frame
                    if 0 <= dist <= response_window and dist < best_dist:
                        best_dist = dist
                        best_si = si
                if best_si is not None:
                    matched_pulses.add(p_idx)
                    used_spikes.add(best_si)
            on_counts[i] = len(matched_pulses)
    else:
        on_counts = np.sum(activated > 0, axis=1).astype(int)
        off_counts = np.sum(activated < 0, axis=1).astype(int)

    groups = []
    for n in range(1, n_pulses + 1):
        on_idxs = np.where(on_counts == n)[0]
        if len(on_idxs) > 0:
            groups.append(
                NeuronGroup(
                    group_id=f"ON_{n}_response(s)",
                    neurons=[neurons[i] for i in on_idxs],
                    method="light-evoked",
                    **metadata,
                )
            )
        off_idxs = np.where(off_counts == n)[0]
        if len(off_idxs) > 0:
            groups.append(
                NeuronGroup(
                    group_id=f"OFF_{n}_response(s)",
                    neurons=[neurons[i] for i in off_idxs],
                    method="light-evoked",
                    **metadata,
                )
            )
    return groups


# Functional Clustering Dispatcher 
def build_groups_from_labels(
    labels: np.ndarray,
    row_indices: np.ndarray,
    neuron_indices: np.ndarray,
    *,
    min_group_size: int = 2,
) -> List[dict]:
    """Convert cluster labels into plain group dicts.

    Parameters
    ----------
    labels : array of cluster IDs, one per neuron.
    row_indices : positional indices into the filtered array.
    neuron_indices : original neuron/ROI indices.
    min_group_size : clusters smaller than this are dropped.

    Returns
    -------
    list of dicts with keys ``group_id``, ``row_indices``,
    ``neuron_indices``, ``n_neurons``.
    """
    labels = np.asarray(labels)
    row_indices = np.asarray(row_indices)
    neuron_indices = np.asarray(neuron_indices)

    groups: List[dict] = []
    for cid in np.unique(labels):
        mask = labels == cid
        if int(np.sum(mask)) < min_group_size:
            continue
        groups.append({
            "group_id": int(cid),
            "row_indices": row_indices[mask].tolist(),
            "neuron_indices": neuron_indices[mask].tolist(),
            "n_neurons": int(np.sum(mask)),
        })
    return sorted(groups, key=lambda g: g["group_id"])

