"""NeuronGroup class for grouped neurons."""
from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .neuron import Neuron


class NeuronGroup:
    """A group of functionally connected neurons identified by a clustering method."""

    def __init__(
        self,
        group_id,
        neurons: List[Neuron],
        method: str = "corr",
        row_indices: Optional[List[int]] = None,
        **metadata,
    ) -> None:
        self.group_id = group_id
        self.neurons = neurons
        self.size = len(neurons)
        self.method = method
        self.metadata = metadata

        self.neuron_indices = [n.index for n in neurons]
        self.filtered_idxs = [n.filtered_index for n in neurons]
        # Row positions in the similarity matrix (may differ from filtered_idxs
        # when the matrix is built from a subset, e.g. active neurons only).
        self.row_indices = row_indices if row_indices is not None else self.filtered_idxs

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
        idxs = self.row_indices
        if len(idxs) < 2:
            return float("nan")
        sub = matrix[np.ix_(idxs, idxs)]
        tri = sub[np.triu_indices(len(idxs), k=1)]
        return float(np.nanmean(tri))

    def group_mean_similarity(self, matrix: Optional[np.ndarray]) -> float:
        """Mean pairwise similarity/distance for this group given any matrix."""
        return self._mean_upper_tri(matrix)

    def __repr__(self) -> str:
        return f"NeuronGroup(id={self.group_id}, size={self.size}, method={self.method})"
