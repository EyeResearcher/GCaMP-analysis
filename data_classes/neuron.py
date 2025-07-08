import sys


from utils.io_utils import SummaryFiles
from utils.spike_utils import find_spikes, compute_spike_constants, compute_area_under_curve
import numpy as np
import neo
from data_classes.spike import Spike
class Neuron:
    def __init__(self, row_index, features: dict, summary_files: SummaryFiles, fs=30):
        self.row_index = row_index
        self.features = features
        self.filtered_index = None
        roi_data = summary_files.get_roi_data(row_index)
        self.raw_fluorescence = roi_data['F']
        self.spike_prob       = roi_data['spike_prob']
        self.ops              = roi_data['ops']
        self.sampling_rate    = fs
        self.spike_prob_peak_indices = []
        self.spike_prob_peak_values  = None
        self.f_peak_indices = []
        self.f_peak_values  = None
        self.spike_prob_average_amplitude = None
        self.f_average_amplitude = None
        self.num_spikes = 0
        self.spike_frequency = 0
        self.tau_stats = None
        self.rise_constants_stats = None
        self.area_under_curve = None
        self.area_per_spike = None
        self.binary_spike_train = None

    def _find_spikes(self, sigma=4, window_size=10):
        if self.spike_prob is None or self.raw_fluorescence is None:
            raise ValueError("spike_prob and raw_fluorescence must be set before calling find_spikes")
        (self.spike_prob_peak_indices,
         self.spike_prob_peak_values,
         self.f_peak_indices,
         self.f_peak_values) = find_spikes(
            self.spike_prob,
            self.raw_fluorescence,
            sigma=sigma,
            window_size=window_size
        )
        self.spikes = [
            Spike(idx_prob, val_prob, idx_raw, val_raw)
            for idx_prob, val_prob, idx_raw, val_raw in zip(
                self.spike_prob_peak_indices,
                self.spike_prob_peak_values,
                self.f_peak_indices,
                self.f_peak_values
            )]
        
        for spike in self.spikes:
            spike.compute_features(
                self.raw_fluorescence,
                self.spike_prob,
                self.features.get("spike_prom_skew", 0.0)
            )

    def _average_amplitudes(self):
        if self.spike_prob_peak_values is not None and len(self.spike_prob_peak_values):
            self.spike_prob_average_amplitude = float(np.mean(self.spike_prob_peak_values))
        if self.f_peak_values is not None and len(self.f_peak_values):
            self.f_average_amplitude = float(np.mean(self.f_peak_values))

    def _model_rise_and_decay(self):
        params = [compute_spike_constants(self.raw_fluorescence, idx, fs=self.sampling_rate)
                  for idx in self.f_peak_indices]
        arr = np.array(params)
        self.rise_constants_stats = arr[:,0] if arr.size else np.array([])
        self.tau_stats            = arr[:,1] if arr.size else np.array([])

    def _compute_area_under_curve(self):
        self.area_under_curve = compute_area_under_curve(
            self.raw_fluorescence, fs=self.sampling_rate)
        n = len(self.f_peak_indices)
        self.area_per_spike = self.area_under_curve / n if n else 0

    def _make_spike_train(self):
        times = 1000 * np.array(self.f_peak_indices) / self.ops['fs']
        self.binary_spike_train = neo.SpikeTrain(
            times, units='ms', t_stop=1000*self.ops['nframes']/self.ops['fs'])

    def _compute_all_spike_stats(self):
        self._find_spikes()
        self._average_amplitudes()
        self._model_rise_and_decay()
        self._compute_area_under_curve()
        self._make_spike_train()
        self.num_spikes = len(self.f_peak_indices)
        self.spike_frequency = (
            self.num_spikes / (len(self.raw_fluorescence) / self.sampling_rate)
        )
