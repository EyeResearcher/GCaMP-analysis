from filtering.feature_utils import compute_spike_features, zscore_features
import numpy as np
class Spike:
    def __init__(self, idx_prob, val_prob, idx_raw, val_raw):
        self.roi_index = None
        self.idx_prob = idx_prob
        self.val_prob = val_prob
        self.idx_raw = idx_raw
        self.val_raw = val_raw
        # Placeholders for computed features
        self.features = {}
        self.z_features = {}
        self.i = None  # Index in the spike list
        

    def compute_features(self, i, raw_trace, spike_prob_trace, left_base_prominences, neuron_prom_skew):
        """
        Compute spike features using the provided traces and neuron-level prominence skew.
        """
        self.i = i
        self.features = compute_spike_features(
            i, raw_trace, spike_prob_trace, left_base_prominences, self.idx_prob, neuron_prom_skew
        )
       
    
    def _set_roi_index(self, roi_index):
        """
        Set the ROI index for this spike.
        """
        self.roi_index = roi_index