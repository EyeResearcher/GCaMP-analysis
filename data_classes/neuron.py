"""Neuron class for filtered ROIs."""
import numpy as np
from typing import List, Optional

class Neuron:
    """Represents a validated neuron after ROI filtering."""
    
    def __init__(self,
                 row_index: int,
                 f_trace: np.ndarray,
                 cascade_prob: np.ndarray,
                 fs: float = 30.0):
        """
        Initialize Neuron.
        
        Parameters:
            row_index: Original ROI index
            f_trace: Fluorescence trace
            cascade_prob: Cascade spike probability
            fs: Sampling frequency in Hz
        """
        self.row_index = row_index
        self.raw_fluorescence = f_trace
        self.cascade_prob = cascade_prob
        self.fs = fs
        
        # Will be populated by pipeline
        self.spikes = []
        self.filtered_index = None  # Index after filtering
        self.binary_spike_train = None
        
    def get_spike_times(self) -> np.ndarray:
        """Get spike times in seconds."""
        return np.array([s.frame_index / self.fs for s in self.spikes])
    
    def get_spike_rate(self) -> float:
        """Calculate mean spike rate in Hz."""
        duration = len(self.raw_fluorescence) / self.fs
        return len(self.spikes) / duration if duration > 0 else 0.0
    
    def __repr__(self):
        return f"Neuron(index={self.row_index}, spikes={len(self.spikes)}, rate={self.get_spike_rate():.2f}Hz)"