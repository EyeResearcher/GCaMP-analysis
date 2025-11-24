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
        self.n_peaks_raw = len(self.peaks)
        self.peaks_filtered = []
        
    def get_spike_features(self, sm_sp) -> list:
        """Extract spike features using the spike detection module.
        Args: 
            sm_sp: Smoothed spike probability signal
        Returns:
            list[Dict]: List of feature dictionaries for each spike"""
        self.spk_features, __ = detect_spikes(sm_sp, self.peaks, roi_idx=self.index, mode="inference")
        return self.spk_features
    
    def filter_spikes(self, predictions) -> None:
        """Filter spikes based on model predictions.
        Args:
            predictions (list[bool]): List indicating whether each spike is valid.
        """
        self.peaks_filtered = [peak for peak, pred in zip(self.peaks, predictions) if pred]
        
    def __repr__(self):
        return f"Neuron(index={self.row_index}, spikes={len(self.spikes)}, rate={self.get_spike_rate():.2f}Hz)"