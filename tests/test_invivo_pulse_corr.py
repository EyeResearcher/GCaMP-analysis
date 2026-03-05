"""Unit tests for pulse-correlation grouping pipeline.

Covers:
  - pulse_similarity   (similarity.py)
  - pulse_cluster      (clustering.py)
  - PulseStrategy      (strategies.py)
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from gcamp_analysis.data_classes.roi import ROI
from gcamp_analysis.data_classes.neuron import Neuron
from gcamp_analysis.grouping_processing.similarity import pulse_similarity
from gcamp_analysis.grouping_processing.clustering import pulse_cluster
from gcamp_analysis.grouping_processing.strategies import PulseStrategy


# ── helpers ──────────────────────────────────────────────────────────

def _make_neuron(index: int, n_frames: int = 100) -> Neuron:
    """Create a minimal Neuron backed by a zero-filled ROI trace."""
    roi = ROI(index=index, f_trace=np.zeros(n_frames))
    return Neuron(roi=roi, filtered_index=index)


def _make_video(neurons, norm_sm_f, n_frames, fs=15.0):
    """Build a MagicMock Video with the subset of attributes used by PulseStrategy."""
    vid = MagicMock()
    vid.neurons = neurons
    vid.norm_sm_f = norm_sm_f
    vid.n_frames = n_frames
    vid.fs = fs
    return vid


# =====================================================================
# pulse_similarity tests
# =====================================================================

class TestPulseSimilarity:
    """Tests for similarity.pulse_similarity."""

    def test_output_shape_matches_input(self):
        """Verify (n_neurons, n_frames) output shape for random input with 3 scheduled pulses."""
        n_neurons, n_frames = 5, 200
        sm = np.random.default_rng(0).random((n_neurons, n_frames))
        schedule = [50, 100, 150]
        activated = pulse_similarity(sm, bin_size=3, schedule=schedule, n_frames=n_frames)
        assert activated.shape == (n_neurons, n_frames)

    def test_pulses_mask_correct_bins(self):
        """With bin_size=1 and spikes at schedule frames, activation must be zero outside those frames."""
        n_neurons, n_frames = 2, 50
        sm = np.zeros((n_neurons, n_frames))
        schedule = [10, 30]
        for f in schedule:
            sm[:, f] = 1.0
        activated = pulse_similarity(sm, bin_size=1, schedule=schedule, n_frames=n_frames)
        non_sched = np.ones(n_frames, dtype=bool)
        non_sched[schedule] = False
        assert np.all(activated[:, non_sched] == 0.0), "No activation outside pulse bins"

    def test_wider_bin_expands_window(self):
        """With bin_size=5 the pulse mask spans [centre-2, centre+2]; frames outside must be zero."""
        n_frames = 50
        sm = np.zeros((1, n_frames))
        schedule = [25]
        sm[0, 25] = 5.0
        activated = pulse_similarity(sm, bin_size=5, schedule=schedule, n_frames=n_frames)
        assert activated.shape == (1, n_frames)
        outside = np.concatenate([np.arange(0, 23), np.arange(28, n_frames)])
        assert np.all(activated[0, outside] == 0.0)

    def test_no_peaks_gives_zeros(self):
        """A constant trace produces zero diff everywhere, so activated should be all zeros."""
        n_frames = 100
        sm = np.ones((3, n_frames))
        activated = pulse_similarity(sm, bin_size=3, schedule=[20, 50, 80], n_frames=n_frames)
        assert np.all(activated == 0.0)

    def test_schedule_at_boundaries(self):
        """Schedule frames at 0 and n_frames-1 must not raise and must return the correct shape."""
        n_frames = 30
        sm = np.random.default_rng(1).random((2, n_frames))
        activated = pulse_similarity(sm, bin_size=3, schedule=[0, n_frames - 1], n_frames=n_frames)
        assert activated.shape == (2, n_frames)

    def test_empty_schedule(self):
        """An empty schedule produces an all-zero pulse mask, so activated is all zeros."""
        sm = np.random.default_rng(2).random((2, 40))
        activated = pulse_similarity(sm, bin_size=3, schedule=[], n_frames=40)
        assert np.all(activated == 0.0)


# =====================================================================
# pulse_cluster tests
# =====================================================================

class TestPulseCluster:
    """Tests for clustering.pulse_cluster."""

    def test_groups_by_pulse_count(self):
        """Neurons responding to 1, 2, or 3 pulses each get their own group."""
        neurons = [_make_neuron(i) for i in range(4)]
        activated = np.zeros((4, 50))
        activated[0, [10, 20, 30]] = 1.0
        activated[1, [10, 20]] = 1.0
        activated[2, [10]] = 1.0
        activated[3, [10, 20, 30]] = 1.0

        groups = pulse_cluster(neurons, activated, n_pulses=3)
        ids = {g.group_id for g in groups}
        assert "1_pulses" in ids
        assert "2_pulses" in ids
        assert "3_pulses" in ids

    def test_group_neuron_assignment(self):
        """Two neurons with 1 pulse and one with 2 pulses must be assigned to the correct groups."""
        neurons = [_make_neuron(i) for i in range(3)]
        activated = np.zeros((3, 20))
        activated[0, [5]] = 1.0
        activated[1, [5, 15]] = 1.0
        activated[2, [5]] = 1.0

        groups = pulse_cluster(neurons, activated, n_pulses=2)
        grp_map = {g.group_id: g for g in groups}

        assert len(grp_map["1_pulses"].neurons) == 2
        assert len(grp_map["2_pulses"].neurons) == 1

    def test_no_activated_neurons(self):
        """All-zero activated array must produce an empty group list."""
        neurons = [_make_neuron(i) for i in range(3)]
        activated = np.zeros((3, 20))
        groups = pulse_cluster(neurons, activated, n_pulses=3)
        assert groups == []

    def test_method_metadata(self):
        """Every returned NeuronGroup must carry method='pulse'."""
        neurons = [_make_neuron(0)]
        activated = np.zeros((1, 20))
        activated[0, [5, 10]] = 1.0
        groups = pulse_cluster(neurons, activated, n_pulses=2)
        for g in groups:
            assert g.method == "pulse"

    def test_extra_metadata_forwarded(self):
        """Arbitrary keyword arguments must appear in NeuronGroup.metadata."""
        neurons = [_make_neuron(0)]
        activated = np.zeros((1, 10))
        activated[0, [3]] = 1.0
        groups = pulse_cluster(neurons, activated, n_pulses=1, custom_key="hello")
        assert groups[0].metadata.get("custom_key") == "hello"

    def test_neurons_with_zero_pulses_excluded(self):
        """Neurons that respond to zero pulses must not appear in any group."""
        neurons = [_make_neuron(i) for i in range(3)]
        activated = np.zeros((3, 20))
        activated[0, [5]] = 1.0
        groups = pulse_cluster(neurons, activated, n_pulses=2)
        all_grouped = [n for g in groups for n in g.neurons]
        assert len(all_grouped) == 1
        assert all_grouped[0].roi.index == 0


# =====================================================================
# PulseStrategy tests
# =====================================================================

class TestPulseStrategy:
    """Tests for strategies.PulseStrategy."""

    def test_name(self):
        """Strategy name must be 'pulse'."""
        assert PulseStrategy().name == "pulse"

    def test_compute_with_program(self):
        """With 'program' config, strategy builds a schedule from start/interval and returns a valid GroupingResult."""
        n_neurons, n_frames = 4, 200
        neurons = [_make_neuron(i, n_frames) for i in range(n_neurons)]
        sm = np.random.default_rng(42).random((n_neurons, n_frames))
        video = _make_video(neurons, sm, n_frames)

        config = {
            "program": True,
            "start": 50,
            "interval": 50,
            "bin_size": 3,
        }
        result = PulseStrategy().compute(video, config)

        assert result.config_label == "pulse_corr"
        assert result.matrix is not None
        assert result.matrix.shape == (n_neurons, n_frames)

    def test_compute_with_explicit_schedule(self):
        """With 'schedule' config, strategy uses the provided frame list directly."""
        n_neurons, n_frames = 3, 100
        neurons = [_make_neuron(i, n_frames) for i in range(n_neurons)]
        sm = np.random.default_rng(7).random((n_neurons, n_frames))
        video = _make_video(neurons, sm, n_frames)

        config = {
            "schedule": [20, 60],
            "bin_size": 5,
        }
        result = PulseStrategy().compute(video, config)
        assert result.matrix.shape == (n_neurons, n_frames)
        assert isinstance(result.groups, list)

    def test_raises_without_program_or_schedule(self):
        """ValueError must be raised when neither 'program' nor 'schedule' is provided."""
        video = _make_video([], np.zeros((0, 10)), 10)
        with pytest.raises(ValueError, match="program.*schedule|schedule.*program"):
            PulseStrategy().compute(video, {"bin_size": 3})

    def test_make_sched_generates_correct_frames(self):
        """_make_sched(start=10, interval=30, frames=100) must return [10, 40, 70]."""
        strat = PulseStrategy()
        sched = strat._make_sched(start=10, interval=30, frames=100)
        assert sched == [10, 40, 70]

    def test_make_sched_empty_when_start_exceeds_frames(self):
        """_make_sched must return an empty list when start >= frames."""
        strat = PulseStrategy()
        sched = strat._make_sched(start=200, interval=30, frames=100)
        assert sched == []

    def test_groups_are_neuron_groups(self):
        """Every group in the result must be a NeuronGroup instance, using traces with spikes at pulse frames."""
        from gcamp_analysis.data_classes.neuron_group import NeuronGroup

        n_neurons, n_frames = 4, 100
        neurons = [_make_neuron(i, n_frames) for i in range(n_neurons)]
        sm = np.zeros((n_neurons, n_frames))
        for f in [20, 60]:
            sm[:, f] = 5.0
        video = _make_video(neurons, sm, n_frames)

        config = {"schedule": [20, 60], "bin_size": 3}
        result = PulseStrategy().compute(video, config)
        for g in result.groups:
            assert isinstance(g, NeuronGroup)

