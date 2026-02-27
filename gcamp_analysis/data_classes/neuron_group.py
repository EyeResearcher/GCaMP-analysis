"""NeuronGroup class for grouped neurons."""
from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .neuron import Neuron


class NeuronGroup:
    """A group of functionally connected neurons identified by a clustering method."""

    def __init__(
        self,
        group_id,
        neurons: List[Neuron],
        method: str = "corr",
        **metadata,
    ) -> None:
        self.group_id = group_id
        self.neurons = neurons
        self.size = len(neurons)
        self.method = method
        self.metadata = metadata

        self.neuron_indices = [n.index for n in neurons]
        self.filtered_idxs = [n.filtered_index for n in neurons]

        self.mean_spk_rate: Optional[float] = None
        self.mean_spk_stats: dict = {}

    # Convenience accessors for common metadata
    @property
    def t_win(self) -> Optional[float]:
        return self.metadata.get("t_win")

    @property
    def corr_thresh(self) -> Optional[float]:
        return self.metadata.get("corr_thresh")

    @property
    def sttc_thresh(self) -> Optional[float]:
        return self.metadata.get("sttc_thresh")

    # ------------------------------------------------------------------
    # Pairwise similarity helpers
    # ------------------------------------------------------------------

    def _mean_upper_tri(self, matrix: Optional[np.ndarray]) -> float:
        """Mean of upper-triangle entries for this group's sub-matrix. NaN if < 2 members or matrix unavailable."""
        if matrix is None or not isinstance(matrix, np.ndarray) or matrix.ndim < 2:
            return float("nan")
        if len(self.filtered_idxs) < 2:
            return float("nan")
        sub = matrix[np.ix_(self.filtered_idxs, self.filtered_idxs)]
        tri = sub[np.triu_indices(len(self.filtered_idxs), k=1)]
        return float(np.nanmean(tri))

    def group_mean_similarity(self, matrix: Optional[np.ndarray]) -> float:
        """Mean pairwise similarity/distance for this group given any matrix."""
        return self._mean_upper_tri(matrix)

    # ------------------------------------------------------------------
    # Aggregate spike statistics
    # ------------------------------------------------------------------

    def get_mean_spike_stats(self, matrices: dict[str, Optional[np.ndarray]] | None = None) -> dict:
        """Compute and store mean spike statistics across group members.

        Parameters
        ----------
        matrices : {strategy_name: matrix}, optional
            Similarity/distance matrices to compute per-group connectivity.
        """
        rates = [n.summary_stats["spike_frequency"] for n in self.neurons]
        self.mean_spk_rate = float(np.mean(rates)) if rates else 0.0
        mean_num_spikes = float(np.mean([len(n.spikes) for n in self.neurons])) if rates else 0.0

        mean_of_means = pd.DataFrame(
            [n.summary_stats for n in self.neurons]
        ).filter(like="mean_").mean()

        self.mean_spk_stats = mean_of_means.to_dict()
        self.mean_spk_stats["spike_rate"] = self.mean_spk_rate
        self.mean_spk_stats["number_of_spikes"] = mean_num_spikes

        for name, mat in (matrices or {}).items():
            self.mean_spk_stats[f"mean_{name}"] = self.group_mean_similarity(mat)
        return self.mean_spk_stats

    def __repr__(self) -> str:
        return f"NeuronGroup(id={self.group_id}, size={self.size}, method={self.method})"