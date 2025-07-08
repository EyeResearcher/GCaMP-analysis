from filtering.feature_utils import compute_spike_features

class Spike:
    def __init__(self, idx_prob, val_prob, idx_raw, val_raw):
        self.idx_prob = idx_prob
        self.val_prob = val_prob
        self.idx_raw = idx_raw
        self.val_raw = val_raw
        # Placeholders for computed features
        self.left_base_prominence = None
        self.spike_prob_value = val_prob
        self.delta_skew = None

    def compute_features(self, raw_trace, spike_prob_trace, neuron_prom_skew):
        """
        Compute spike features using the provided traces and neuron-level prominence skew.
        """
        self.left_base_prominence, self.spike_prob_value, self.delta_skew = compute_spike_features(
            raw_trace, spike_prob_trace, self.idx_prob, neuron_prom_skew
        )