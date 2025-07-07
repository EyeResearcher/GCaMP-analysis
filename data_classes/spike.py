from utils.spike_utils import compute_spike_constants
class Spike:
    def __init__(self, fluo, peak_idx, fs=30,
                 rise_fraction=0.1, decay_fraction=0.9):
        self.m, self.tau = compute_spike_constants(fluo, peak_idx, fs = fs)
