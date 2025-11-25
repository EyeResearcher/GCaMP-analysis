"""Neuron class for filtered ROIs."""
from data_classes.spike import Spike
import numpy as np
from typing import List, Optional
from roi import ROI
from spike_classifier.prepare_data import detect_spikes, _create_large_window, _create_small_window
from scipy.signal import peak_prominences
from scipy.ndimage import gaussian_filter1d
class Neuron(ROI):
    """Represents a validated neuron after ROI filtering."""
    
    def __init__(self,
                 roi_instance : ROI,
                 filtered_index: int,
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
        self.filtered_index = filtered_index  # Index after filtering
        
        # Will be populated by pipeline
        self.spikes = []
        self.binary_spike_train = None
        self.spk_features
        self.n_peaks_raw = len(self.peaks)
        self.peaks_filtered = []
        self.all_spk_stats = []
    def get_spike_features(self, sm_norm_f, raw_norm_f) -> list:
        """Extract spike features using the spike detection module.
        Args: 
            sm_sp: Smoothed spike probability signal
        Returns:
            list[Dict]: List of feature dictionaries for each spike"""
        self.spk_features, __ = detect_spikes(sm_norm_f, raw_norm_f, self.peaks, roi_idx=self.index, mode="inference")
        return self.spk_features
    
    def filter_spikes(self, predictions) -> None:
        """Filter spikes based on model predictions.
        Args:
            predictions (list[bool]): List indicating whether each spike is valid.
        """
        peaks_filtered = [peak for peak, pred in zip(self.peaks, predictions) if bool(pred)]

        return peaks_filtered
 
    
    def instantiate_spikes(self, sm_norm_f, sg_norm_f) -> list[Spike]:
        """Instantiate Spike objects for each filtered spike.
        Args:
            sm_norm_f (np.ndarray): Smoothed normalized fluorescence trace
            sg_norm_f (np.ndarray): SavGol filtered normalized fluorescence trace
        Returns:
            list[Spike]: List of Spike objects"""
        # Extract the valid portion
        prominences, left_bases, right_bases = peak_prominences(
            sm_norm_f, self.peaks_filtered)
        
        self.spikes = [None] * len(self.peaks_filtered)  # Pre-allocate list
        self.all_spk_stats = []
        for i, peak in enumerate(self.peaks_filtered):
            spike = Spike(sm_f_idx=peak, position_idx=i)

            spike.left_base, spike.right_base = left_bases[i], right_bases[i]
            spike.prev_position_idx = self.peaks_filtered[i-1] if i > 0 else 0
            spike.next_position_idx = self.peaks_filtered[i+1] if i < len(self.peaks_filtered) - 1 else len(sm_norm_f)
            large_win, small_win = spike.create_windows(sg_norm_f)
            spike.f_value = sg_norm_f[spike.sm_f_idx]
            spike.stats = spike.get_statistics()
            self.all_spk_stats.append(spike.stats)
            self.spikes[i] = spike    

        return self.spikes

    def summarize_spike_statistics(self) -> tuple[dict, dict]:
        """Summarize spike statistics across all spikes.
        Returns:
            tuple: (summary_dict, raw_dict) where:
                - summary_dict: Dictionary with mean/std for each feature
                - raw_dict: Dictionary mapping feature names to lists of values
        """
        if not self.all_spk_stats:
            return {}, {}
        
        import pandas as pd
        
        # Convert list of dicts to DataFrame
        stats_df = pd.DataFrame(self.all_spk_stats)
        
        # Create raw dictionary: {feature_name: [val1, val2, ...]}
        raw_dict = {col: stats_df[col].tolist() for col in stats_df.columns}
        spike_freq = len(self.spikes) / (len(self.f_trace) / self.fs) if len(self.f_trace) > 0 else 0.0
        raw_dict['spike_frequency'] = spike_freq
        # Compute mean and std for all columns
        summary = {
            **{f'mean_{col}': stats_df[col].mean() for col in stats_df.columns},
            **{f'std_{col}': stats_df[col].std() for col in stats_df.columns}
        }
    
        return summary, raw_dict
    def __repr__(self):
        return f"Neuron(index={self.row_index}, spikes={len(self.spikes)}, rate={self.get_spike_rate():.2f}Hz)"