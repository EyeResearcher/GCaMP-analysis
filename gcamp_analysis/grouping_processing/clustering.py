"""Clustering functions used by grouping strategies."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from gcamp_analysis.data_classes.neuron_group import NeuronGroup

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

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


def cluster_louvain(
    neurons: List["Neuron"],
    similarity: np.ndarray,
    *,
    edge_threshold: float = 0.0,
    resolution: float = 1.0,
    min_group_size: int = 2,
    method: str = "louvain",
    group_id_prefix: str = "louv",
    seed: int | None = 42,
    **metadata,
) -> List[NeuronGroup]:
    """Louvain modularity-based community detection.

    Unlike threshold/hierarchical clustering, Louvain automatically determines
    the number of communities by maximizing modularity — no fixed threshold
    on group count.

    Parameters
    ----------
    neurons : list of Neuron
        Neurons to cluster.
    similarity : np.ndarray
        Similarity matrix (NOT distance). Higher values = stronger connection.
        Typically correlation values in [0, 1].
    edge_threshold : float, default 0.0
        Minimum similarity to create an edge. Edges below this are excluded.
    resolution : float, default 1.0
        Resolution parameter for Louvain. Higher = more smaller communities,
        lower = fewer larger communities.
    min_group_size : int, default 2
        Minimum neurons per group.
    method : str
        Method label for metadata.
    group_id_prefix : str
        Prefix for group IDs.
    seed : int or None
        Random seed for reproducibility.
    **metadata
        Additional metadata to attach to groups.

    Returns
    -------
    list of NeuronGroup
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required for Louvain clustering. Install with: pip install networkx")

    if len(neurons) < 2:
        return []

    S = np.asarray(similarity, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        return []

    n = S.shape[0]
    np.fill_diagonal(S, 0.0)  # No self-loops

    # Build weighted graph from similarity matrix
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = S[i, j]
            if np.isfinite(w) and w > edge_threshold:
                G.add_edge(i, j, weight=w)

    # Run Louvain community detection
    communities = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)

    # Convert to NeuronGroups
    groups = []
    for k, comm in enumerate(communities, start=1):
        idxs = sorted(comm)
        if len(idxs) >= min_group_size:
            groups.append(
                NeuronGroup(
                    group_id=f"{group_id_prefix}_{k}",
                    neurons=[neurons[i] for i in idxs],
                    method=method,
                    resolution=resolution,
                    edge_threshold=edge_threshold,
                    **metadata,
                )
            )

    return groups
