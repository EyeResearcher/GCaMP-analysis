"""Unit tests for pulse-correlation and light-evoked grouping pipelines.

Covers:
  - pulse_similarity          (similarity.py)
  - align_light_evoked        (similarity.py)
  - pulse_cluster             (clustering.py)
  - light_evoked_cluster      (clustering.py)
  - PulseStrategy             (strategies.py)
  - LightEvokedStrategy       (strategies.py)
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from gcamp_analysis.data_classes.roi import ROI
from gcamp_analysis.data_classes.neuron import Neuron
from gcamp_analysis.grouping_processing.similarity import align_light_evoked
from gcamp_analysis.grouping_processing.clustering import light_evoked_cluster
from gcamp_analysis.grouping_processing.strategies import LightEvokedStrategy


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

class TestAlignLightEvokedBasic:
    """Basic shape and masking tests for similarity.align_light_evoked."""

    def test_output_shape_matches_input(self):
        """Verify (n_neurons, n_frames) output shape for random input with 3 scheduled pulses."""
        n_neurons, n_frames = 5, 200
        sm = np.random.default_rng(0).random((n_neurons, n_frames))
        schedule = [50, 100, 150]
        activated = align_light_evoked(sm, bin_size=3, schedule=schedule, n_frames=n_frames)
        assert activated.shape == (n_neurons, n_frames)

    def test_pulses_mask_correct_bins(self):
        """With bin_size=1 and spikes at schedule frames, activation must be zero outside those frames."""
        n_neurons, n_frames = 2, 50
        sm = np.zeros((n_neurons, n_frames))
        schedule = [10, 30]
        for f in schedule:
            sm[:, f] = 1.0
        activated = align_light_evoked(sm, bin_size=1, schedule=schedule, n_frames=n_frames)
        non_sched = np.ones(n_frames, dtype=bool)
        non_sched[schedule] = False
        assert np.all(activated[:, non_sched] == 0.0), "No activation outside pulse bins"

    def test_wider_bin_expands_window(self):
        """With bin_size=5 the pulse mask spans [centre-2, centre+2]; frames outside must be zero."""
        n_frames = 50
        sm = np.zeros((1, n_frames))
        schedule = [25]
        sm[0, 25] = 5.0
        activated = align_light_evoked(sm, bin_size=5, schedule=schedule, n_frames=n_frames)
        assert activated.shape == (1, n_frames)
        outside = np.concatenate([np.arange(0, 23), np.arange(28, n_frames)])
        assert np.all(activated[0, outside] == 0.0)

    def test_no_peaks_gives_zeros(self):
        """A constant trace produces zero diff everywhere, so activated should be all zeros."""
        n_frames = 100
        sm = np.ones((3, n_frames))
        activated = align_light_evoked(sm, bin_size=3, schedule=[20, 50, 80], n_frames=n_frames)
        assert np.all(activated == 0.0)

    def test_schedule_at_boundaries(self):
        """Schedule frames at 0 and n_frames-1 must not raise and must return the correct shape."""
        n_frames = 30
        sm = np.random.default_rng(1).random((2, n_frames))
        activated = align_light_evoked(sm, bin_size=3, schedule=[0, n_frames - 1], n_frames=n_frames)
        assert activated.shape == (2, n_frames)

    def test_empty_schedule(self):
        """An empty schedule produces an all-zero pulse mask, so activated is all zeros."""
        sm = np.random.default_rng(2).random((2, 40))
        activated = align_light_evoked(sm, bin_size=3, schedule=[], n_frames=40)
        assert np.all(activated == 0.0)


# =====================================================================
# pulse_cluster tests
# =====================================================================

class TestLightEvokedClusterBasic:
    """Basic ON-only tests for clustering.light_evoked_cluster (backwards compat with old pulse_cluster tests)."""

    def test_on_groups_by_pulse_count(self):
        """Neurons responding to 1, 2, or 3 ON pulses each get their own ON group."""
        neurons = [_make_neuron(i) for i in range(4)]
        activated = np.zeros((4, 50))
        activated[0, [10, 20, 30]] = 1.0
        activated[1, [10, 20]] = 1.0
        activated[2, [10]] = 1.0
        activated[3, [10, 20, 30]] = 1.0

        groups = light_evoked_cluster(neurons, activated, n_pulses=3)
        ids = {g.group_id for g in groups}
        assert "ON_1_response(s)" in ids
        assert "ON_2_response(s)" in ids
        assert "ON_3_response(s)" in ids

    def test_group_neuron_assignment(self):
        """Two neurons with 1 ON pulse and one with 2 ON pulses must be assigned to the correct groups."""
        neurons = [_make_neuron(i) for i in range(3)]
        activated = np.zeros((3, 20))
        activated[0, [5]] = 1.0
        activated[1, [5, 15]] = 1.0
        activated[2, [5]] = 1.0

        groups = light_evoked_cluster(neurons, activated, n_pulses=2)
        grp_map = {g.group_id: g for g in groups}

        assert len(grp_map["ON_1_response(s)"].neurons) == 2
        assert len(grp_map["ON_2_response(s)"].neurons) == 1

    def test_no_activated_neurons(self):
        """All-zero activated array must produce an empty group list."""
        neurons = [_make_neuron(i) for i in range(3)]
        activated = np.zeros((3, 20))
        groups = light_evoked_cluster(neurons, activated, n_pulses=3)
        assert groups == []

    def test_method_metadata(self):
        """Every returned NeuronGroup must carry method='light-evoked'."""
        neurons = [_make_neuron(0)]
        activated = np.zeros((1, 20))
        activated[0, [5, 10]] = 1.0
        groups = light_evoked_cluster(neurons, activated, n_pulses=2)
        for g in groups:
            assert g.method == "light-evoked"

    def test_extra_metadata_forwarded(self):
        """Arbitrary keyword arguments must appear in NeuronGroup.metadata."""
        neurons = [_make_neuron(0)]
        activated = np.zeros((1, 10))
        activated[0, [3]] = 1.0
        groups = light_evoked_cluster(neurons, activated, n_pulses=1, custom_key="hello")
        assert groups[0].metadata.get("custom_key") == "hello"

    def test_neurons_with_zero_pulses_excluded(self):
        """Neurons that respond to zero pulses must not appear in any group."""
        neurons = [_make_neuron(i) for i in range(3)]
        activated = np.zeros((3, 20))
        activated[0, [5]] = 1.0
        groups = light_evoked_cluster(neurons, activated, n_pulses=2)
        all_grouped = [n for g in groups for n in g.neurons]
        assert len(all_grouped) == 1
        assert all_grouped[0].roi.index == 0


# =====================================================================
# LightEvokedStrategy basic tests
# =====================================================================

class TestLightEvokedStrategyBasic:
    """Basic tests for strategies.LightEvokedStrategy (schedule building, shape, error handling)."""

    def test_name(self):
        """Strategy name must be 'light-evoked'."""
        assert LightEvokedStrategy().name == "light-evoked"

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
        result = LightEvokedStrategy().compute(video, config)

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
        result = LightEvokedStrategy().compute(video, config)
        assert result.matrix.shape == (n_neurons, n_frames)
        assert isinstance(result.groups, list)

    def test_raises_without_program_or_schedule(self):
        """ValueError must be raised when neither start/interval nor schedule is provided."""
        video = _make_video([], np.zeros((0, 10)), 10)
        with pytest.raises(ValueError):
            LightEvokedStrategy().compute(video, {"bin_size": 3})

    def test_make_sched_generates_correct_frames(self):
        """_make_sched(start=10, interval=30, frames=100) must return [10, 40, 70]."""
        strat = LightEvokedStrategy()
        sched = strat._make_sched(start=10, interval=30, frames=100)
        assert sched == [10, 40, 70]

    def test_make_sched_empty_when_start_exceeds_frames(self):
        """_make_sched must return an empty list when start >= frames."""
        strat = LightEvokedStrategy()
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
        result = LightEvokedStrategy().compute(video, config)
        for g in result.groups:
            assert isinstance(g, NeuronGroup)


# =====================================================================
# align_light_evoked tests
# =====================================================================

class TestAlignLightEvoked:
    """Tests for similarity.align_light_evoked (ON/OFF detection)."""

    def test_output_shape(self):
        """Output shape must match (n_neurons, n_frames)."""
        n_neurons, n_frames = 4, 100
        sm = np.random.default_rng(0).random((n_neurons, n_frames))
        activated = align_light_evoked(sm, bin_size=3, schedule=[20, 60], n_frames=n_frames)
        assert activated.shape == (n_neurons, n_frames)

    def test_on_cell_positive(self):
        """A sustained increase starting at the pulse frame should produce a +1 entry in activated."""
        n_frames = 50
        sm = np.zeros((1, n_frames))
        sm[0, 20:] = 5.0
        activated = align_light_evoked(sm, bin_size=3, schedule=[20], n_frames=n_frames)
        assert np.any(activated > 0), "ON response should produce positive values"

    def test_off_cell_negative(self):
        """A sharp decrease at a pulse frame should produce a -1 entry in activated."""
        n_frames = 50
        sm = np.zeros((1, n_frames))
        sm[0, 19] = 5.0
        sm[0, 20] = 0.0
        activated = align_light_evoked(sm, bin_size=3, schedule=[20], n_frames=n_frames)
        assert np.any(activated < 0), "OFF response should produce negative values"

    def test_flat_trace_all_zeros(self):
        """A constant trace has no diff peaks, so activated must be all zeros."""
        n_frames = 60
        sm = np.ones((2, n_frames)) * 3.0
        activated = align_light_evoked(sm, bin_size=3, schedule=[10, 30, 50], n_frames=n_frames)
        assert np.all(activated == 0.0)

    def test_values_limited_to_neg1_zero_pos1(self):
        """All entries in activated must be -1, 0, or +1."""
        n_frames = 80
        sm = np.random.default_rng(5).random((3, n_frames))
        activated = align_light_evoked(sm, bin_size=5, schedule=[15, 40, 65], n_frames=n_frames)
        unique_vals = set(np.unique(activated))
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_activation_only_within_pulse_bins(self):
        """Non-zero entries may only occur at frames covered by the pulse mask."""
        n_frames = 50
        sm = np.random.default_rng(3).random((2, n_frames))
        schedule = [15, 35]
        activated = align_light_evoked(sm, bin_size=1, schedule=schedule, n_frames=n_frames)
        non_sched = np.ones(n_frames, dtype=bool)
        non_sched[schedule] = False
        assert np.all(activated[:, non_sched] == 0.0)

    def test_empty_schedule(self):
        """An empty schedule must produce all zeros."""
        sm = np.random.default_rng(4).random((2, 40))
        activated = align_light_evoked(sm, bin_size=3, schedule=[], n_frames=40)
        assert np.all(activated == 0.0)


# =====================================================================
# light_evoked_cluster tests
# =====================================================================

class TestLightEvokedCluster:
    """Tests for clustering.light_evoked_cluster (ON/OFF grouping)."""

    def test_on_groups_created(self):
        """Neurons with positive pulse sums should land in ON groups."""
        neurons = [_make_neuron(i) for i in range(3)]
        activated = np.zeros((3, 30))
        activated[0, [5, 15]] = 1.0
        activated[1, [5]] = 1.0
        activated[2, [5, 15]] = 1.0

        groups = light_evoked_cluster(neurons, activated, n_pulses=2)
        ids = {g.group_id for g in groups}
        assert "ON_1_response(s)" in ids
        assert "ON_2_response(s)" in ids

    def test_off_groups_created(self):
        """Neurons with negative pulse sums should land in OFF groups."""
        neurons = [_make_neuron(i) for i in range(3)]
        activated = np.zeros((3, 30))
        activated[0, [5, 15]] = -1.0
        activated[1, [5]] = -1.0
        activated[2, [5]] = -1.0

        groups = light_evoked_cluster(neurons, activated, n_pulses=2)
        ids = {g.group_id for g in groups}
        assert "OFF_1_response(s)" in ids
        assert "OFF_2_response(s)" in ids

    def test_mixed_on_off(self):
        """ON and OFF neurons in the same input should produce both group types."""
        neurons = [_make_neuron(i) for i in range(4)]
        activated = np.zeros((4, 30))
        activated[0, [5]] = 1.0
        activated[1, [5]] = -1.0
        activated[2, [5, 15]] = 1.0
        activated[3, [5, 15]] = -1.0

        groups = light_evoked_cluster(neurons, activated, n_pulses=2)
        ids = {g.group_id for g in groups}
        assert "ON_1_response(s)" in ids
        assert "OFF_1_response(s)" in ids
        assert "ON_2_response(s)" in ids
        assert "OFF_2_response(s)" in ids

    def test_neuron_assignment_on_vs_off(self):
        """Verify correct neurons land in ON vs OFF groups."""
        neurons = [_make_neuron(i) for i in range(3)]
        activated = np.zeros((3, 20))
        activated[0, [5]] = 1.0
        activated[1, [5]] = -1.0
        activated[2, [5]] = 1.0

        groups = light_evoked_cluster(neurons, activated, n_pulses=1)
        grp_map = {g.group_id: g for g in groups}
        assert len(grp_map["ON_1_response(s)"].neurons) == 2
        assert len(grp_map["OFF_1_response(s)"].neurons) == 1
        assert grp_map["OFF_1_response(s)"].neurons[0].roi.index == 1

    def test_zero_response_excluded(self):
        """Neurons with zero pulse sum must not appear in any group."""
        neurons = [_make_neuron(i) for i in range(3)]
        activated = np.zeros((3, 20))
        activated[0, [5]] = 1.0
        groups = light_evoked_cluster(neurons, activated, n_pulses=2)
        all_grouped = [n for g in groups for n in g.neurons]
        assert len(all_grouped) == 1

    def test_no_activated_returns_empty(self):
        """All-zero input must return no groups."""
        neurons = [_make_neuron(i) for i in range(3)]
        activated = np.zeros((3, 20))
        groups = light_evoked_cluster(neurons, activated, n_pulses=3)
        assert groups == []

    def test_method_is_light_evoked(self):
        """All groups must carry method='light-evoked'."""
        neurons = [_make_neuron(0), _make_neuron(1)]
        activated = np.zeros((2, 20))
        activated[0, [5]] = 1.0
        activated[1, [5]] = -1.0
        groups = light_evoked_cluster(neurons, activated, n_pulses=1)
        for g in groups:
            assert g.method == "light-evoked"

    def test_metadata_forwarded(self):
        """Keyword arguments must appear in NeuronGroup.metadata."""
        neurons = [_make_neuron(0)]
        activated = np.zeros((1, 10))
        activated[0, [3]] = 1.0
        groups = light_evoked_cluster(neurons, activated, n_pulses=1, stim_type="470nm")
        assert groups[0].metadata.get("stim_type") == "470nm"


# =====================================================================
# LightEvokedStrategy tests
# =====================================================================

class TestLightEvokedStrategy:
    """Tests for strategies.LightEvokedStrategy."""

    def test_name(self):
        """Strategy name must be 'light-evoked'."""
        assert LightEvokedStrategy().name == "light-evoked"

    def test_compute_with_program(self):
        """With 'program' config, strategy builds a schedule and returns a valid GroupingResult."""
        n_neurons, n_frames = 4, 200
        neurons = [_make_neuron(i, n_frames) for i in range(n_neurons)]
        sm = np.random.default_rng(42).random((n_neurons, n_frames))
        video = _make_video(neurons, sm, n_frames)

        config = {"program": True, "start": 50, "interval": 50, "bin_size": 3}
        result = LightEvokedStrategy().compute(video, config)
        assert result.matrix is not None
        assert result.matrix.shape == (n_neurons, n_frames)

    def test_compute_with_schedule(self):
        """With 'schedule' config, strategy uses the provided frame list."""
        n_neurons, n_frames = 3, 100
        neurons = [_make_neuron(i, n_frames) for i in range(n_neurons)]
        sm = np.random.default_rng(7).random((n_neurons, n_frames))
        video = _make_video(neurons, sm, n_frames)

        config = {"schedule": [20, 60], "bin_size": 5}
        result = LightEvokedStrategy().compute(video, config)
        assert result.matrix.shape == (n_neurons, n_frames)
        assert isinstance(result.groups, list)

    def test_on_off_groups_from_strategy(self):
        """Strategy should produce ON and OFF groups when traces have both increase and decrease at pulse frames."""
        from gcamp_analysis.data_classes.neuron_group import NeuronGroup

        n_frames = 100
        sm = np.zeros((2, n_frames))
        sm[0, 50] = 5.0
        sm[1, 49] = 5.0
        sm[1, 50] = 0.0
        neurons = [_make_neuron(i, n_frames) for i in range(2)]
        video = _make_video(neurons, sm, n_frames)

        config = {"schedule": [50], "bin_size": 3}
        result = LightEvokedStrategy().compute(video, config)

        ids = {g.group_id for g in result.groups}
        has_on = any("ON" in gid for gid in ids)
        has_off = any("OFF" in gid for gid in ids)
        assert has_on or has_off, "Should detect at least one ON or OFF group"
        for g in result.groups:
            assert isinstance(g, NeuronGroup)


# =====================================================================
# Processor-level tests (group_stats, bottom-up aggregation, sibling
# comparison) for light-evoked ON/OFF cell counts.
# =====================================================================

from pathlib import Path
from gcamp_analysis.experiments.processor import VideoRunRecord, ExperimentProcessor
from gcamp_analysis.experiments.tree import TreeNode
from gcamp_analysis.experiments.summary_utils import StatSummary


def _empty_stat() -> StatSummary:
    return StatSummary()


def _make_video_record(
    *,
    n_neurons: int = 10,
    n_neurons_grouped: int = 6,
    n_neurons_ungrouped: int = 4,
    n_groups_per_strategy: dict | None = None,
    group_stats: dict | None = None,
    video_dir: str = "vid",
) -> VideoRunRecord:
    """Construct a minimal VideoRunRecord for testing aggregation."""
    return VideoRunRecord(
        video_dir=Path(video_dir),
        metrics_dir=Path(video_dir) / "metrics",
        n_rois_total=20,
        n_rois_good=15,
        n_neurons=n_neurons,
        n_spikes_kept=100,
        n_neurons_grouped=n_neurons_grouped,
        n_neurons_ungrouped=n_neurons_ungrouped,
        n_groups_per_strategy=n_groups_per_strategy or {},
        group_stats=group_stats or {},
        kin_unweighted=_empty_stat(),
        kin_weighted_spikes=_empty_stat(),
        freq_unweighted=_empty_stat(),
        kin_grouped=_empty_stat(),
        kin_ungrouped=_empty_stat(),
        freq_grouped=_empty_stat(),
        freq_ungrouped=_empty_stat(),
    )


def _build_two_video_tree(
    rec_a: VideoRunRecord,
    rec_b: VideoRunRecord,
) -> TreeNode:
    """Build a minimal tree: root -> (video_a, video_b)."""
    root = TreeNode(name="root", path=Path("root"))
    child_a = TreeNode(name="video_a", path=Path("root/video_a"))
    child_b = TreeNode(name="video_b", path=Path("root/video_b"))
    child_a.payload = rec_a
    child_b.payload = rec_b
    root.add_child(child_a)
    root.add_child(child_b)
    return root


class TestGroupStatsLightEvokedFields:
    """Verify the per-video group_stats dict for light-evoked has ON/OFF cell count keys."""

    def test_n_cells_keys_present(self):
        """group_stats for light-evoked must have n_cells_ON_*, n_cells_OFF_*, and totals."""
        gs = {
            "light-evoked": {
                "mean_group_size": 2.0,
                "median_group_size": 2.0,
                "mean_group_corr": 0.5,
                "mean_spikes_per_group": 3.0,
                "n_cells_ON_1_response(s)": 3,
                "n_cells_ON_2_response(s)": 2,
                "n_cells_OFF_1_response(s)": 1,
                "total_ON_cells": 5,
                "total_OFF_cells": 1,
            }
        }
        rec = _make_video_record(group_stats=gs, n_groups_per_strategy={"light-evoked": 3})
        le = rec.group_stats["light-evoked"]
        assert le["n_cells_ON_1_response(s)"] == 3
        assert le["n_cells_ON_2_response(s)"] == 2
        assert le["n_cells_OFF_1_response(s)"] == 1
        assert le["total_ON_cells"] == 5
        assert le["total_OFF_cells"] == 1

    def test_no_light_evoked_means_no_count_keys(self):
        """Without a light-evoked strategy, no count keys must exist."""
        gs = {
            "corr": {
                "mean_group_size": 3.0,
                "median_group_size": 3.0,
                "mean_group_corr": 0.7,
                "mean_spikes_per_group": 5.0,
            }
        }
        rec = _make_video_record(group_stats=gs, n_groups_per_strategy={"corr": 2})
        assert "light-evoked" not in rec.group_stats
        corr_stats = rec.group_stats["corr"]
        assert not any(k.startswith("n_cells_") for k in corr_stats)
        assert "total_ON_cells" not in corr_stats
        assert "total_OFF_cells" not in corr_stats


class TestBottomUpAggregation:
    """Bottom-up aggregation must weighted-average all group_stats fields by n_groups_per_strategy."""

    @staticmethod
    def _aggregate_two(gs_a, gs_b, ng_a, ng_b):
        """Build a two-leaf tree, run bottom-up, return the root's group_stats."""
        rec_a = _make_video_record(
            group_stats=gs_a, n_groups_per_strategy=ng_a, video_dir="a"
        )
        rec_b = _make_video_record(
            group_stats=gs_b, n_groups_per_strategy=ng_b, video_dir="b"
        )
        root = _build_two_video_tree(rec_a, rec_b)
        # ExperimentProcessor needs a runner, but we only call bottom-up
        proc = ExperimentProcessor.__new__(ExperimentProcessor)
        proc._compute_bottom_up_summaries(root)
        return root.group_stats

    def test_all_fields_weighted_averaged(self):
        """All group_stats fields — including counts — are weighted-averaged by n_groups_per_strategy."""
        gs_a = {
            "light-evoked": {
                "mean_group_size": 2.0,
                "n_cells_ON_1_response(s)": 3,
                "total_ON_cells": 4,
                "total_OFF_cells": 2,
            }
        }
        gs_b = {
            "light-evoked": {
                "mean_group_size": 4.0,
                "n_cells_ON_1_response(s)": 5,
                "total_ON_cells": 8,
                "total_OFF_cells": 1,
            }
        }
        merged = self._aggregate_two(
            gs_a, gs_b,
            {"light-evoked": 3}, {"light-evoked": 4},
        )
        le = merged["light-evoked"]
        # Weighted average: (val_a * w_a + val_b * w_b) / (w_a + w_b)
        assert le["mean_group_size"] == pytest.approx((2*3 + 4*4) / 7)
        assert le["n_cells_ON_1_response(s)"] == pytest.approx((3*3 + 5*4) / 7)
        assert le["total_ON_cells"] == pytest.approx((4*3 + 8*4) / 7)
        assert le["total_OFF_cells"] == pytest.approx((2*3 + 1*4) / 7)

    def test_equal_weights(self):
        """With equal n_groups_per_strategy, the result is a simple average."""
        gs_a = {
            "light-evoked": {
                "mean_group_size": 2.0,
                "total_ON_cells": 10,
                "total_OFF_cells": 0,
            }
        }
        gs_b = {
            "light-evoked": {
                "mean_group_size": 4.0,
                "total_ON_cells": 6,
                "total_OFF_cells": 2,
            }
        }
        merged = self._aggregate_two(
            gs_a, gs_b,
            {"light-evoked": 2}, {"light-evoked": 2},
        )
        le = merged["light-evoked"]
        assert le["mean_group_size"] == pytest.approx(3.0)
        assert le["total_ON_cells"] == pytest.approx(8.0)
        assert le["total_OFF_cells"] == pytest.approx(1.0)

    def test_asymmetric_weights(self):
        """When one child has more groups, its values pull the average."""
        gs_a = {
            "light-evoked": {
                "mean_group_size": 2.0,
                "total_ON_cells": 3,
                "total_OFF_cells": 1,
            }
        }
        gs_b = {
            "light-evoked": {
                "mean_group_size": 6.0,
                "total_ON_cells": 7,
                "total_OFF_cells": 4,
            }
        }
        merged = self._aggregate_two(
            gs_a, gs_b,
            {"light-evoked": 1}, {"light-evoked": 3},
        )
        le = merged["light-evoked"]
        # Weighted average: (2*1 + 6*3) / (1+3) = 5.0
        assert le["mean_group_size"] == pytest.approx(5.0)
        # (3*1 + 7*3) / 4 = 6.0
        assert le["total_ON_cells"] == pytest.approx(6.0)
        # (1*1 + 4*3) / 4 = 3.25
        assert le["total_OFF_cells"] == pytest.approx(3.25)

    def test_non_light_evoked_unaffected(self):
        """The corr strategy must still weighted-average normally."""
        gs_a = {"corr": {"mean_group_size": 2.0, "mean_group_corr": 0.8}}
        gs_b = {"corr": {"mean_group_size": 4.0, "mean_group_corr": 0.6}}
        merged = self._aggregate_two(
            gs_a, gs_b,
            {"corr": 3}, {"corr": 3},
        )
        assert merged["corr"]["mean_group_size"] == pytest.approx(3.0)
        assert merged["corr"]["mean_group_corr"] == pytest.approx(0.7)


class TestSiblingComparisonLightEvokedColumns:
    """_compare_one must surface ON/OFF count columns in the DataFrame."""

    @staticmethod
    def _compare(gs_a, gs_b, ng_a, ng_b):
        rec_a = _make_video_record(
            group_stats=gs_a, n_groups_per_strategy=ng_a, video_dir="a", n_neurons=5,
        )
        rec_b = _make_video_record(
            group_stats=gs_b, n_groups_per_strategy=ng_b, video_dir="b", n_neurons=5,
        )
        root = _build_two_video_tree(rec_a, rec_b)
        proc = ExperimentProcessor.__new__(ExperimentProcessor)
        proc._compute_bottom_up_summaries(root)
        return ExperimentProcessor._compare_one(root)

    def test_total_on_off_columns(self):
        """DataFrame must contain total_ON_cells and total_OFF_cells columns."""
        gs = {
            "light-evoked": {
                "mean_group_size": 2.0,
                "median_group_size": 2.0,
                "mean_group_corr": 0.5,
                "mean_spikes_per_group": 1.0,
                "n_cells_ON_1_response(s)": 3,
                "total_ON_cells": 3,
                "total_OFF_cells": 0,
            }
        }
        ng = {"light-evoked": 1}
        df = self._compare(gs, gs, ng, ng)
        assert df is not None
        assert "total_ON_cells" in df.columns
        assert "total_OFF_cells" in df.columns

    def test_n_cells_columns(self):
        """DataFrame must contain n_cells_ON_* and n_cells_OFF_* columns."""
        gs = {
            "light-evoked": {
                "mean_group_size": 2.0,
                "median_group_size": 2.0,
                "mean_group_corr": 0.5,
                "mean_spikes_per_group": 1.0,
                "n_cells_ON_1_response(s)": 2,
                "n_cells_OFF_2_response(s)": 1,
                "total_ON_cells": 2,
                "total_OFF_cells": 1,
            }
        }
        ng = {"light-evoked": 2}
        df = self._compare(gs, gs, ng, ng)
        assert df is not None
        assert "n_cells_ON_1_response(s)" in df.columns
        assert "n_cells_OFF_2_response(s)" in df.columns

    def test_values_match_per_child(self):
        """The ON/OFF cell counts in each row must match the children's group_stats."""
        gs_a = {
            "light-evoked": {
                "mean_group_size": 2.0,
                "median_group_size": 2.0,
                "mean_group_corr": 0.5,
                "mean_spikes_per_group": 1.0,
                "n_cells_ON_1_response(s)": 4,
                "total_ON_cells": 4,
                "total_OFF_cells": 0,
            }
        }
        gs_b = {
            "light-evoked": {
                "mean_group_size": 3.0,
                "median_group_size": 3.0,
                "mean_group_corr": 0.6,
                "mean_spikes_per_group": 2.0,
                "n_cells_ON_1_response(s)": 7,
                "total_ON_cells": 7,
                "total_OFF_cells": 0,
            }
        }
        ng = {"light-evoked": 1}
        df = self._compare(gs_a, gs_b, ng, ng)
        assert df is not None
        row_a = df[df["child"] == "video_a"].iloc[0]
        row_b = df[df["child"] == "video_b"].iloc[0]
        assert row_a["n_cells_ON_1_response(s)"] == 4
        assert row_a["total_ON_cells"] == 4
        assert row_b["n_cells_ON_1_response(s)"] == 7
        assert row_b["total_ON_cells"] == 7

    def test_no_light_evoked_no_extra_columns(self):
        """Without light-evoked, no ON/OFF columns must appear."""
        gs = {
            "corr": {
                "mean_group_size": 2.0,
                "median_group_size": 2.0,
                "mean_group_corr": 0.5,
                "mean_spikes_per_group": 1.0,
            }
        }
        ng = {"corr": 2}
        df = self._compare(gs, gs, ng, ng)
        assert df is not None
        assert "total_ON_cells" not in df.columns
        assert "total_OFF_cells" not in df.columns


# =====================================================================
# Real-data regression tests for ON/OFF RGC assignment
# =====================================================================

class TestAlignLightEvokedRealData:
    """Regression tests using real fluorescence traces.

    Test data:
        5729L14.npy  (24×200)  – 6 evenly-spaced pulses
        5732L5.npy   (27×200)  – 4 irregularly-spaced pulses

    Traces are Gaussian-smoothed (sigma=0.8) before calling
    ``align_light_evoked`` with magnitude-based deconfliction and
    ``prominence=0.03`` to filter noise peaks.

    Expected directions and approximate counts were determined by
    manual inspection of the overlay plots.
    """

    SIGMA = 0.8

    @staticmethod
    def _load_and_smooth(path: str, rows: list[int], sigma: float) -> np.ndarray:
        from scipy.ndimage import gaussian_filter1d
        import os
        data = np.load(os.path.join("tests", "data", path))
        return gaussian_filter1d(data[rows].astype(float), sigma=sigma, axis=1)

    @staticmethod
    def _net_label(activated_row: np.ndarray) -> tuple[str, int]:
        s = int(np.sum(activated_row))
        if s > 0:
            return "ON", s
        elif s < 0:
            return "OFF", abs(s)
        return "ungrouped", 0

    # ── 5729L14 ──────────────────────────────────────────────────

    def test_5729L14_directions(self):
        """All tested rows in 5729L14 must have the correct ON/OFF direction."""
        rows = [12, 11, 2, 17]
        sm = self._load_and_smooth("5729L14.npy", rows, self.SIGMA)
        sched = list(range(30, 200, 30))
        act = align_light_evoked(sm, bin_size=7, schedule=sched,
                                 n_frames=200, prominence=0.03)

        expected_dirs = ["ON", "ON", "ungrouped", "ON"]
        for i, (row, exp_dir) in enumerate(zip(rows, expected_dirs)):
            direction, _ = self._net_label(act[i])
            assert direction == exp_dir, (
                f"Row {row}: expected {exp_dir}, got {direction}"
            )

    def test_5729L14_net_counts(self):
        """Lock in exact net pulse counts for 5729L14 as a regression baseline."""
        rows = [12, 11, 2, 17]
        sm = self._load_and_smooth("5729L14.npy", rows, self.SIGMA)
        sched = list(range(30, 200, 30))
        act = align_light_evoked(sm, bin_size=7, schedule=sched,
                                 n_frames=200, prominence=0.03)

        expected_nets = [+4, +5, 0, +3]
        for i, (row, exp_net) in enumerate(zip(rows, expected_nets)):
            net = int(np.sum(act[i]))
            assert net == exp_net, (
                f"Row {row}: expected net={exp_net}, got net={net}"
            )

    # ── 5732L5 ───────────────────────────────────────────────────

    def test_5732L5_directions(self):
        """All tested rows in 5732L5 must have the correct ON/OFF direction."""
        rows = [18, 17, 2, 5, 25, 24, 0]
        sm = self._load_and_smooth("5732L5.npy", rows, self.SIGMA)
        sched = [33, 65, 116, 192]
        act = align_light_evoked(sm, bin_size=7, schedule=sched,
                                 n_frames=200, prominence=0.03)

        expected_dirs = ["ON", "ON", "OFF", "OFF", "ON", "ON", "ungrouped"]
        for i, (row, exp_dir) in enumerate(zip(rows, expected_dirs)):
            direction, _ = self._net_label(act[i])
            assert direction == exp_dir, (
                f"Row {row}: expected {exp_dir}, got {direction}"
            )

    def test_5732L5_net_counts(self):
        """Lock in exact net pulse counts for 5732L5 as a regression baseline."""
        rows = [18, 17, 2, 5, 25, 24, 0]
        sm = self._load_and_smooth("5732L5.npy", rows, self.SIGMA)
        sched = [33, 65, 116, 192]
        act = align_light_evoked(sm, bin_size=7, schedule=sched,
                                 n_frames=200, prominence=0.03)

        expected_nets = [+3, +3, -4, -3, +2, +3, 0]
        for i, (row, exp_net) in enumerate(zip(rows, expected_nets)):
            net = int(np.sum(act[i]))
            assert net == exp_net, (
                f"Row {row}: expected net={exp_net}, got net={net}"
            )

    # ── prominence parameter behaviour ───────────────────────────

    def test_prominence_filters_noise_peaks(self):
        """With prominence=0.03, fewer peaks should be detected than with None."""
        rows = [12, 11]
        sm = self._load_and_smooth("5729L14.npy", rows, self.SIGMA)
        sched = list(range(30, 200, 30))

        act_no_prom = align_light_evoked(sm, bin_size=7, schedule=sched,
                                         n_frames=200, prominence=None)
        act_prom = align_light_evoked(sm, bin_size=7, schedule=sched,
                                      n_frames=200, prominence=0.03)

        total_no_prom = np.sum(np.abs(act_no_prom) > 0)
        total_prom = np.sum(np.abs(act_prom) > 0)
        assert total_prom <= total_no_prom, (
            f"prominence filtering should reduce peak count: "
            f"{total_prom} > {total_no_prom}"
        )

    def test_output_values_are_valid(self):
        """Activated array entries must be in {-1, 0, +1}."""
        rows = [12, 11, 2, 17]
        sm = self._load_and_smooth("5729L14.npy", rows, self.SIGMA)
        sched = list(range(30, 200, 30))
        act = align_light_evoked(sm, bin_size=7, schedule=sched,
                                 n_frames=200, prominence=0.03)
        unique = set(np.unique(act))
        assert unique.issubset({-1.0, 0.0, 1.0})

