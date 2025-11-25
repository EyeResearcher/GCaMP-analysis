"""NeuronGroup class for grouped neurons."""
from __future__ import annotations
from typing import Any, List, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .neuron import Neuron

class NeuronGroup:
    """Represents a group of functionally connected neurons."""
    
    def __init__(self,
                 group_id: int,
                 neurons: List[Neuron],
                 method: str = 'sttc'):
        """
        Initialize NeuronGroup.
        
        Parameters:
            group_id: Group identifier
            neurons: List of Neuron objects in group
            method: Grouping method ('sttc' or 'dtw')
        """
        self.group_id = group_id
        self.neurons = neurons
        self.method = method
        self.mean_spk_rate = None
        self.mean_spk_stats = {}
        self.idxs = [n.filtered_index for n in self.neurons]
    @property
    def size(self) -> int:
        """Number of neurons in group."""
        return len(self.neurons)
    
    @property
    def neuron_indices(self) -> List[int]:
        """Original ROI indices of neurons."""
        return [n.index for n in self.neurons]
    
    def get_mean_spike_stats(self, sttc: np.ndarray, dtw: np.ndarray) -> float:
        """Mean spike stats across the group.
            Returns: mean_spike_stats dictionary
                {}
                    """
        rates = [n.raw_stats['spike_frequency'] for n in self.neurons]
        self.mean_spk_rate = np.mean(rates) if rates else 0.0
        self.mean_num_spikes = np.mean([len(n.spikes) for n in self.neurons]) if rates else 0.0
        mean_of_means = pd.DataFrame([n.summary_stats for n in self.neurons]).filter(like='mean_').mean()
        self.mean_spk_stats = mean_of_means.to_dict()    
        self.mean_spk_stats['spike_rate'] = self.mean_spk_rate
        self.mean_spk_stats['number_of_spikes'] = self.mean_num_spikes
        self.mean_spk_stats['mean_sttc'] = self.group_mean_sttc(self, sttc, self.neurons)   
        self.mean_spk_stats['mean_dtw'] = self.group_mean_dtw(self, dtw, self.neurons)
        return self.mean_spk_stats
    
    def group_mean_sttc(group : NeuronGroup, sttc_matrix: np.ndarray, neurons: List[Any]) -> float:
        """Mean pairwise STTC for members of `group`. Returns np.nan if group size < 2."""
        idx = [n.filtered_index for n in group.neurons]
        if len(idx) < 2:
            return float('nan')
        sub = sttc_matrix[np.ix_(idx, idx)]
        n = len(idx)
        tri = sub[np.triu_indices(n, k=1)]
        return float(np.nanmean(tri))

    def group_mean_dtw(group : NeuronGroup, dtw_matrix: np.ndarray, neurons: List[Any]) -> float:
        """Mean pairwise DTW cost for members of `group`. Returns np.nan if group size < 2 or dtw_matrix is None."""
        if dtw_matrix is None:
            return float('nan')
        idx = [n.filtered_index for n in group.neurons]
        if len(idx) < 2:
            return float('nan')
        sub = dtw_matrix[np.ix_(idx, idx)]
        n = len(idx)
        tri = sub[np.triu_indices(n, k=1)]
        return float(np.nanmean(tri))
    
    def __repr__(self):
        return f"NeuronGroup(id={self.group_id}, size={self.size}, method={self.method})"