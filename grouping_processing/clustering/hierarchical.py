from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING, Tuple
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from data_classes.neuron_group import NeuronGroup

if TYPE_CHECKING:
    from data_classes.neuron import Neuron

@dataclass
class HierarchicalClusterer:
    linkage_method: str = "average"
    distance_threshold: float = 0.3
    min_group_size: int = 2

    def cluster_from_distance(
        self,
        neurons: List["Neuron"],
        dist: np.ndarray,
        *,
        group_id_prefix: str,
        method: str,
        meta: dict,
    ) -> List[NeuronGroup]:
        if len(neurons) < 2:
            return []

        d = np.asarray(dist, dtype=float)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            return []

        np.fill_diagonal(d, 0.0)
        condensed = squareform(d, checks=False)
        Z = linkage(condensed, method=self.linkage_method)
        clusters = fcluster(Z, self.distance_threshold, criterion="distance")

        groups: List[NeuronGroup] = []
        for cid in np.unique(clusters):
            members = [neurons[i] for i in range(len(neurons)) if clusters[i] == cid]
            if len(members) >= self.min_group_size:
                gid = f"{group_id_prefix}_{int(cid)}"
                groups.append(NeuronGroup(gid, members, method=method, **meta))
        return groups
