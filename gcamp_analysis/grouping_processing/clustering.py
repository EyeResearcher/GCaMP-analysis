"""Clustering functions used by grouping strategies."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

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


def light_evoked_cluster(neurons: List["Neuron"], activated: np.ndarray, n_pulses: int, **metadata) -> List[NeuronGroup]:
    pulses_by_neuron = np.sum(activated, axis=1)
    groups = []
    for n in range(1, n_pulses + 1):
        on_idxs = np.where(pulses_by_neuron == n)[0]
        if len(on_idxs) > 0:
            groups.append(
                NeuronGroup(
                    group_id=f"ON_{n}_response(s)",
                    neurons=[neurons[i] for i in on_idxs],
                    method="light-evoked",
                    **metadata,
                )
            )
        off_idxs = np.where(pulses_by_neuron == -n)[0]
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