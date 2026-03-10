"""Clustering functions used by grouping strategies."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from gcamp_analysis.data_classes.neuron_group import NeuronGroup

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.neuron import Neuron


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


def cluster_hierarchical(
    neurons: List["Neuron"],
    dist: np.ndarray,
    *,
    threshold: float,
    linkage_method: str = "average",
    min_group_size: int = 2,
    method: str = "unknown",
    group_id_prefix: str = "grp",
    **metadata,
) -> List[NeuronGroup]:
    """Agglomerative hierarchical clustering via scipy."""
    if len(neurons) < 2:
        return []
    d = np.asarray(dist, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        return []
    np.fill_diagonal(d, 0.0)
    labels = fcluster(linkage(squareform(d, checks=False), method=linkage_method), threshold, criterion="distance")

    return [
        NeuronGroup(
            group_id=f"{group_id_prefix}_{int(cid)}",
            neurons=[neurons[i] for i in range(len(neurons)) if labels[i] == cid],
            method=method,
            **metadata,
        )
        for cid in np.unique(labels)
        if sum(labels == cid) >= min_group_size
    ]

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
    if cluster_method not in dispatch:
        raise ValueError(f"Unknown cluster_method '{cluster_method}'. Choose from {list(dispatch.keys())}")
    return dispatch[cluster_method](neurons, dist, **kwargs)

def recluster_within_group(
    neurons: List["Neuron"],
    corr_sub: np.ndarray,
    *,
    parent_group_id: str = "",
    method: str = "corr",
    distance_threshold: float = 0.6,
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
    distance_threshold : float
        Distance threshold for clustering.
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
        threshold=distance_threshold,
        min_group_size=min_group_size,
        method=method,
    )

    # Re-label sub-group IDs to reflect parent
    for i, sg in enumerate(sub_groups):
        sg.group_id = f"{parent_group_id}_sub{i}"
        sg.metadata["parent_group_id"] = parent_group_id

    return sub_groups

def light_evoked_cluster(
    neurons: List["Neuron"],
    activated: np.ndarray,
    n_pulses: int,
    schedule: list[int],
    bin_size: int = 3,
    response_window: int = 10,
    **metadata,
) -> List[NeuronGroup]:
    # Count per-pulse responses using each neuron's actual detected spikes,
    # not derivative peaks.  A spike is matched to a pulse if its peak frame
    # falls within [pulse_frame, pulse_frame + response_window].
    n_neurons = len(neurons)
    pulse_counts = np.zeros(n_neurons, dtype=int)
    for i, neuron in enumerate(neurons):
        matched_pulses: set[int] = set()
        used_spikes: set[int] = set()
        for p_idx, pulse_frame in enumerate(schedule):
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
        pulse_counts[i] = len(matched_pulses)
    groups = []
    for n in range(1, n_pulses + 1):
        idxs = np.where(pulse_counts == n)[0]
        if len(idxs) > 0:
            groups.append(
                NeuronGroup(
                    group_id=f"ON_{n}_response(s)",
                    neurons=[neurons[i] for i in idxs],
                    method="light-evoked",
                    **metadata,
                )
            )
    return groups