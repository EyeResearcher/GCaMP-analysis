"""Neuron class for filtered ROIs."""
from data_classes.spike import Spike
import numpy as np
from typing import List, Optional
from .roi import ROI
from spike_classifier.prepare_data import detect_spikes, _create_large_window, _create_small_window
from scipy.signal import peak_prominences, find_peaks
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
        self.spikes : List[Spike] = []
        self.binary_spike_train = None
        self.spk_features = []
        self.peaks = np.ndarray([])
        self.n_peaks_raw = None
        self.peaks_filtered = []
        self.all_spk_stats = []
        self.summary_stats = {}
        self.raw_stats = {}
    def get_spike_features(self, sm_norm_f) -> list:
        """Extract spike features using the spike detection module.
        Args: 
            sm_sp: Smoothed spike probability signal
        Returns:
            list[Dict]: List of feature dictionaries for each spike"""
        # Detect peaks on the (smoothed) spike-probability or provided smoothed signal
        peaks, _ = find_peaks(sm_norm_f)
        self.n_peaks_raw = int(len(peaks))

        # Run detection to compute per-spike feature dicts (inference mode returns list, keys)
        feats_list, spike_keys = detect_spikes(sm_norm_f, peaks, roi_idx=self.index, mode="inference")

        # Return both the features list and the peak indices so callers can assign them
        return feats_list, peaks
    
    def filter_spikes(self, predictions) -> None:
        """Filter spikes based on model predictions.
        Args:
            predictions (list[bool]): List indicating whether each spike is valid.
        """
        import numpy as _np

        # Normalize peaks to a 1-D numpy array (handle None, scalars, lists)
        peaks_arr = self.peaks
        if peaks_arr is None:
            peaks_arr = _np.array([], dtype=int)
        else:
            peaks_arr = _np.asarray(peaks_arr)
            if peaks_arr.ndim == 0:
                # scalar -> treat as single-element array
                peaks_arr = peaks_arr.reshape(1)

        n_peaks = peaks_arr.size

        # Coerce predictions to a 1-D array so single-value predictions still iterate
        preds_arr = _np.asarray(predictions)
        if preds_arr.ndim == 0:
            preds_arr = preds_arr.reshape(1)

        # Ensure length matches; if not, truncate or pad with False
        if preds_arr.size < n_peaks:
            pad = _np.zeros(n_peaks - preds_arr.size, dtype=bool)
            preds_arr = _np.concatenate([preds_arr, pad])
        elif preds_arr.size > n_peaks:
            preds_arr = preds_arr[:n_peaks]

        # Build filtered peaks as Python ints
        peaks_filtered = [int(p) for p, pred in zip(peaks_arr.tolist(), preds_arr) if bool(pred)]

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
            spike.prominence = prominences[i]  # Set prominence from peak_prominences
            spike.prev_position_idx = self.peaks_filtered[i-1] if i > 0 else 0
            spike.next_position_idx = self.peaks_filtered[i+1] if i < len(self.peaks_filtered) - 1 else len(sm_norm_f)
            large_win, small_win = spike.create_windows(sg_norm_f)
            spike.f_value = sg_norm_f[spike.sm_f_idx]
            spike.stats = spike.get_statistics()
            # Copy computed stats back to spike attributes for io_handlers access
            spike.rise_slope = spike.stats.get('rise_slope', 0.0)
            spike.decay_tau = spike.stats.get('decay_tau', 5.0)
            self.all_spk_stats.append(spike.stats)
            self.spikes[i] = spike    

        # Return both so main process can reassign after parallel execution
        return (self.spikes, self.all_spk_stats)

    def summarize_spike_statistics(self) -> dict:
        """Summarize spike statistics across all spikes.

        Returns:
            dict: Ordered mapping of summary statistics. For each column in the
                per-spike stats DataFrame (in the same column order) this
                returns mean and variance entries as:
                    mean_<col>, var_<col>
                Finally 'spike_frequency' is appended.
        """
        if not self.all_spk_stats:
            return {}

        import pandas as pd
        from collections import OrderedDict

        # Convert list of dicts to DataFrame
        stats_df = pd.DataFrame(self.all_spk_stats)

        # Create raw dictionary: {feature_name: [val1, val2, ...]}
        self.raw_stats = {col: stats_df[col].tolist() for col in stats_df.columns}

        # spike frequency (Hz)
        spike_freq = len(self.spikes) / (len(self.f_trace) / self.fs) if len(self.f_trace) > 0 else 0.0
        self.raw_stats['spike_frequency'] = spike_freq

        # Build ordered summary: for each original column, add mean and variance
        summary = OrderedDict()
        # include original ROI index and filtered index so dataframe rows can be simple 0..N-1 (filtered order)
        summary['neuron_idx'] = int(self.index)
        summary['filtered_index'] = int(self.filtered_index)
        summary["spike_frequency"] = float(spike_freq)
        summary["number_of_spikes"] = len(self.spikes)
        for col in stats_df.columns:
            mean_val = float(stats_df[col].mean())
            var_val = float(stats_df[col].var())  # sample variance (ddof=1) by pandas default
            summary[f"mean_{col}"] = mean_val
            summary[f"var_{col}"] = var_val

       

        # store for backwards access
        self.summary_stats = dict(summary)

        return self.summary_stats
    def __repr__(self):
        return f"Neuron(index={self.index}, spikes={len(self.spikes)})"