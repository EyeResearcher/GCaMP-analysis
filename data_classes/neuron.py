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
        self.filtered_index = None  # Index after filtering
        
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
        peaks_filtered = [peak for peak, pred in zip(self.peaks, predictions) if bool(pred)]
        self.spikes = [Spike(peak, i) for i, peak in enumerate(peaks_filtered)]
        return peaks_filtered
    def valid_spike_prob_region(self, sm_sp) -> np.ndarray:
        """Get the start and end indices of the valid (non-NaN) region in the smoothed spike probability.
        
        Args:
            sm_sp: Smoothed spike probability signal;

        Returns:
            (np.ndar
        """
        valid_mask = ~np.isnan(sm_sp)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return None, None
        start_idx = valid_indices[0]
        end_idx = valid_indices[-1] + 1
        valid_sp = sm_sp[start_idx:end_idx]
        return valid_sp, start_idx, end_idx
    
    def instantiate_spikes(self,sm_sp,norm_f) -> None:
        
        # Extract the valid portion
        valid_spike_prob, start_idx, end_idx = self.valid_spike_prob_region(sm_sp)
        prominences, left_bases, right_bases = peak_prominences(
        valid_spike_prob, self.peaks_filtered)
        for i, spike in enumerate(self.spikes):
            spike = self.get_spike_stats(i, spike, valid_spike_prob, start_idx, end_idx, norm_f, left_bases[i], right_bases[i])
            self.spikes[i] = spike
        return self.spikes
    def get_spike_stats(self, i, spike : Spike, valid_sp, start_idx, end_idx, norm_f, left_base, right_base) -> List[str]:
        """Get list of spike detection methods used."""
        prev_idx = spike.prev_position_idx 
        next_idx = spike.next_position_idx if i < len(self.spikes) - 1 else len(valid_sp)
        large_window = _create_large_window(valid_sp, spike.cascade_peak_idx, left_base, right_base, start_idx=0)
        small_window, small_low_bounds, small_upper_bounds = _create_small_window(valid_sp, spike.cascade_peak_idx, prev_idx, next_idx, start_idx=start_idx)
        spike.f_small_window = norm_f[small_low_bounds: small_upper_bounds]
        spike.smooth_f_window(sigma=2.0)
        spike.f_index = np.argmax(spike.f_small_window_smooth) + small_low_bounds
        spike.f_value = norm_f[spike.f_index]
        rise_slope, decay_tau, decay_shape, decay_shape_features, f_value = spike.get_statistics()
        spike.rise_slope = rise_slope
        spike.decay_tau = decay_tau
        spike.f_value = f_value
        return spike
    def __repr__(self):
        return f"Neuron(index={self.row_index}, spikes={len(self.spikes)}, rate={self.get_spike_rate():.2f}Hz)"