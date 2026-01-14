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
                 method: str = 'sttc',
                 t_win: float = None,
                 sttc_thresh: float = None,
                 dtw_thresh: float = None):
        """
        Initialize NeuronGroup.
        
        Parameters:
            group_id: Group identifier
            neurons: List of Neuron objects in group
            method: Grouping method ('sttc' or 'dtw')
        """
        self.group_id = group_id
        self.size = len(neurons)
        self.neurons = neurons
        self.method = method
        self.mean_spk_rate = None
        self.mean_spk_stats = {}
        self.filtered_idxs = [n.filtered_index for n in self.neurons]
        self.neuron_indices = [n.index for n in self.neurons]
        self.t_win = t_win
        self.sttc_thresh = sttc_thresh
        self.dtw_thresh = dtw_thresh
 


    
    def get_mean_spike_stats(self, sttc: np.ndarray, dtw: np.ndarray) -> float:
        """Mean spike stats across the group.
            Returns: mean_spike_stats dictionary
                {}
                    """
        rates = [n.summary_stats['spike_frequency'] for n in self.neurons]
        self.mean_spk_rate = np.mean(rates) if rates else 0.0
        self.mean_num_spikes = np.mean([len(n.spikes) for n in self.neurons]) if rates else 0.0
        mean_of_means = pd.DataFrame([n.summary_stats for n in self.neurons]).filter(like='mean_').mean()
        self.mean_spk_stats = mean_of_means.to_dict()    
        self.mean_spk_stats['spike_rate'] = self.mean_spk_rate
        self.mean_spk_stats['number_of_spikes'] = self.mean_num_spikes
        self.mean_spk_stats['mean_sttc'] = self.group_mean_sttc(sttc)   
        self.mean_spk_stats['mean_dtw'] = self.group_mean_dtw(dtw)
        return self.mean_spk_stats
    
    def group_mean_sttc(self, sttc_matrix: np.ndarray) -> float:
        """Mean pairwise STTC for members of this group. Returns np.nan if group size < 2."""
        idx = [n.filtered_index for n in self.neurons]
        if len(idx) < 2:
            return float('nan')
        sub = sttc_matrix[np.ix_(idx, idx)]
        n = len(idx)
        tri = sub[np.triu_indices(n, k=1)]
        return float(np.nanmean(tri))

    def group_mean_dtw(self, dtw_matrix: np.ndarray) -> float:
        """Mean pairwise DTW cost for members of this group. Returns np.nan if group size < 2 or dtw_matrix is None/empty."""
        # Handle None, 0-dim arrays, or empty arrays
        if dtw_matrix is None or not isinstance(dtw_matrix, np.ndarray) or dtw_matrix.ndim < 2:
            return float('nan')
        idx = [n.filtered_index for n in self.neurons]
        if len(idx) < 2:
            return float('nan')
        sub = dtw_matrix[np.ix_(idx, idx)]
        n = len(idx)
        tri = sub[np.triu_indices(n, k=1)]
        return float(np.nanmean(tri))
    
    def __repr__(self):
        return f"NeuronGroup(id={self.group_id}, size={self.size}, method={self.method})"