import numpy as np
import pytest

from data_classes import Neuron
from utils.spike_utils import compute_area_under_curve, find_spikes

import numpy as np
import pytest


class DummySummary:
    """Dummy summary object with spike_prob and raw_fluorescence arrays."""
    def __init__(self, spike_prob, raw_fluorescence):
        self.spike_prob = spike_prob
        self.raw_fluorescence = raw_fluorescence


def test_compute_all_spike_stats_basic():
    # Synthetic trace: 100 frames, baseline=1
    spike_prob = np.zeros(100, dtype=float)
    raw_fluo   = np.ones(100, dtype=float)
    
    # Two spikes: magnitudes 200 and 100
    spike_prob[20] = 1.0
    spike_prob[80] = 1.0
    raw_fluo[20]   = 200.0
    raw_fluo[80]   = 100.0

    summary = DummySummary(spike_prob, raw_fluo)
    neuron = Neuron(row_index=0, summary_files=summary, fs=1.0)

    # Run full pipeline
    neuron.compute_all_spike_stats()

    # Spike detection should find peaks at 20 and 80
    assert np.array_equal(neuron.spike_prob_peak_indices, np.array([20, 80]))
    assert np.array_equal(neuron.f_peak_indices,           np.array([20, 80]))

    # Average amplitudes
    assert pytest.approx(neuron.spike_prob_average_amplitude) == 1.0
    assert pytest.approx(neuron.f_average_amplitude)          == 150.0

    # Number of spikes and frequency
    assert neuron.num_spikes == 2
    # fs=1 Hz, 100 frames => duration=100s => freq = 2/100
    assert pytest.approx(neuron.spike_frequency, rel=1e-6)    == 2/100

    
    # Area under curve: baseline=1 => peaks at 199,99
    # AUC = two spikes: 99.5*2 + 49.5*2 = 298.0
    assert pytest.approx(neuron.area_under_curve, rel=1e-6) == 298.0
    # Area per spike = 298.0 / 2 = 149.0
    assert pytest.approx(neuron.area_per_spike, rel=1e-6)    == 149.0


def test_missing_data_raises():
    # Missing spike_prob raises ValueError
    summary = DummySummary(None, np.ones(50))
    neuron = Neuron(row_index=0, summary_files=summary)
    with pytest.raises(ValueError):
        neuron.compute_all_spike_stats()

    # Missing raw_fluorescence raises ValueError
    summary = DummySummary(np.zeros(50), None)
    neuron = Neuron(row_index=0, summary_files=summary)
    with pytest.raises(ValueError):
        neuron.compute_all_spike_stats()
