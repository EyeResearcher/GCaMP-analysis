"""NeuronGroup class for grouped neurons."""
from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

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
        
    @property
    def size(self) -> int:
        """Number of neurons in group."""
        return len(self.neurons)
    
    @property
    def neuron_indices(self) -> List[int]:
        """Original ROI indices of neurons."""
        return [n.row_index for n in self.neurons]
    
    def get_mean_spike_rate(self) -> float:
        """Mean spike rate across group."""
        rates = [n.get_spike_rate() for n in self.neurons]
        return np.mean(rates) if rates else 0.0
    
    def __repr__(self):
        return f"NeuronGroup(id={self.group_id}, size={self.size}, method={self.method})"