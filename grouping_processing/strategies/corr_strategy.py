from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, TYPE_CHECKING
import numpy as np

from grouping_processing.similarity.correlation import TraceCorrelationSimilarity

if TYPE_CHECKING:
    from data_classes.video import Video


@dataclass
class CorrelationStrategy:
    name: str = "corr"

    def compute(self, video: "Video", config: Dict[str, Any]) -> Dict[str, Any]:
        trace_key = str(config.get("trace", "norm_sm_f"))  # you said use fluorescence dynamics
        method = str(config.get("method", "pearson"))

        sim_backend = TraceCorrelationSimilarity(
            method=method,
            remove_global=bool(config.get("remove_global", True)),
            use_diff=bool(config.get("use_diff", True)),
            diff_order=int(config.get("diff_order", 1)),
            zscore_each=bool(config.get("zscore_each", True)),
            clip_negatives=bool(config.get("clip_negatives", True)),
        )

        # choose trace array
        if trace_key == "F":
            traces_full = np.asarray(video.suite2p_data["F"], float)
        else:
            traces_full = np.asarray(getattr(video, trace_key), float)

        # restrict to current neurons
        idxs = [n.index for n in video.neurons]
        traces = traces_full[idxs, :]

        C = sim_backend.compute(traces)

        # simple grouping rule: hierarchical clustering over distance = 1 - corr
        # (keeps this strategy self-contained without new dependencies)
        dist = 1.0 - C

        # threshold: smaller = stricter grouping
        dist_thr = float(config.get("distance_threshold", 0.6))
        min_group = int(config.get("min_group_size", 2))

        groups = _cluster_connected_by_threshold(
            video=video,
            dist=dist,
            threshold=dist_thr,
            min_group_size=min_group,
        )

        label = f"corr_{trace_key}_{method}"
        if sim_backend.remove_global:
            label += "_rmglobal"
        if sim_backend.use_diff:
            label += f"_diff{sim_backend.diff_order}"
        if sim_backend.zscore_each:
            label += "_z"

        return {"groups": groups, "matrix": C, "config_label": label}


def _cluster_connected_by_threshold(video: "Video", dist: np.ndarray, threshold: float, min_group_size: int):
    """
    Fast, dependency-free grouping:
    make an undirected graph where edge exists if dist <= threshold,
    groups = connected components.
    """
    n = dist.shape[0]
    # adjacency
    adj = (dist <= threshold) & np.isfinite(dist)
    np.fill_diagonal(adj, True)

    visited = np.zeros(n, dtype=bool)
    components = []

    for i in range(n):
        if visited[i]:
            continue
        # BFS/DFS
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            nbrs = np.where(adj[u])[0]
            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if len(comp) >= min_group_size:
            components.append(sorted(comp))

    # build NeuronGroups using your existing class
    from data_classes.neuron_group import NeuronGroup

    groups = []
    for k, idxs in enumerate(components, start=1):
        neurons = [video.neurons[i] for i in idxs]
        groups.append(
            NeuronGroup(
                group_id=k,
                neurons=neurons,
                method="corr",
            )
        )
    return groups
