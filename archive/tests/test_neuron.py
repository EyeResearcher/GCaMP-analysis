import numpy as np
import pytest

from utils.spike_utils import (
    compute_spike_constants,
    find_spikes,
    window_spike_transients,
)


def test_find_spikes_aligns_with_fluorescence_maxima():
    prob = np.zeros(100, dtype=float)
    fluo = np.random.default_rng(0).normal(loc=0.0, scale=0.1, size=100)

    prob[:32] = np.nan
    prob[-32:] = np.nan

    prob[40] = 1.0
    prob[65] = 0.8

    fluo[39:42] += np.array([0.5, 3.0, 1.0])
    fluo[63:67] += np.array([0.2, 2.5, 0.4, 0.1])

    (prob_idx, _, _, fluo_idx, fluo_vals), _ = find_spikes(prob, fluo, sigma=2.0, window_radius=5)

    expected_peaks = np.array([40, 65])
    assert np.allclose(prob_idx, expected_peaks, atol=1)
    assert np.allclose(fluo_idx, expected_peaks, atol=1)

    for idx, peak in enumerate(expected_peaks):
        window = slice(max(0, peak - 5), min(len(fluo), peak + 6))
        max_idx = window.start + int(np.argmax(fluo[window]))
        assert fluo_idx[idx] == max_idx
        assert fluo_vals[idx] == pytest.approx(fluo[max_idx])


def test_window_spike_transients_tiles_trace():
    trace = np.array([0, 0, 1, 6, 3, 1, 0, 2, 5, 2, 1, 0, 1, 4, 1, 0], dtype=float)
    peaks = np.array([3, 8, 13])

    windows = window_spike_transients(trace, peaks)

    assert len(windows) == len(peaks)
    prev_end = None
    for i, ((start, peak, end), expected_peak) in enumerate(zip(windows, peaks)):
        assert peak == expected_peak
        assert start <= peak <= end
        if i > 0:
            assert start == prev_end
        prev_end = end


def test_compute_spike_constants_with_clear_decay():
    trace = np.zeros(120)
    trace[50] = 5.0
    decay = 5.0 * np.exp(-np.arange(1, 30) / 6.0)
    trace[51:51 + decay.size] = decay

    rise_slope, decay_tau = compute_spike_constants(trace, peak_idx=50, fs=30.0)

    assert not np.isnan(rise_slope)
    assert not np.isnan(decay_tau)
    assert decay_tau == pytest.approx(0.2, rel=0.5)


def test_compute_spike_constants_when_decay_plateaus():
    trace = np.zeros(60)
    trace[20] = 4.0
    trace[21:30] = 3.0

    rise_slope, decay_tau = compute_spike_constants(trace, peak_idx=20, fs=20.0)

    assert np.isnan(rise_slope) or rise_slope >= 0
    assert not np.isnan(decay_tau)
