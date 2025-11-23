"""Neuron class for filtered ROIs."""
import numpy as np
from typing import List, Optional
from roi import ROI
from spike_classifier.prepare_data import detect_spikes
class Neuron(ROI):
    """Represents a validated neuron after ROI filtering."""
    
    def __init__(self,
                 roi_instance : ROI,
                 fs: float = 30.0):
        """
        Initialize Neuron.
        
        Parameters:
            row_index: Original ROI index
            f_trace: Fluorescence trace
            cascade_prob: Cascade spike probability
            fs: Sampling frequency in Hz
        """
        self.__dict__.update(roi_instance.__dict__)
        self.fs = fs
        
        # Will be populated by pipeline
        self.spikes = []
        self.filtered_index = None  # Index after filtering
        self.binary_spike_train = None
        self.spk_features = []
        
    def get_spike_features(self, sm_sp) -> List[dict]:
        self.spk_features, _ = detect_spikes(sm_sp, self.peaks, mode="inference")
        for i, spike in enumerate(self.spk_features):
            spike['key'] = f"{self.index}_{self.peaks[i]}"
        return self.spk_features
    
    def filter_spikes(self, spike_classifier_model) -> None:
        key = 
    def __repr__(self):
        return f"Neuron(index={self.row_index}, spikes={len(self.spikes)}, rate={self.get_spike_rate():.2f}Hz)"