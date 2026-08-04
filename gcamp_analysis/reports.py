"""Immutable per-stage report records returned by the video pipeline.

Each processing service returns one of these small frozen dataclasses so the
runner and callers can narrate progress without reaching into ``Video`` state.
They carry counts and summary scalars only; they do not own arrays or perform
any computation.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TraceReport:
    """Counts produced by the trace-processing stage."""

    n_rois: int
    n_frames: int
    fs: float


@dataclass(frozen=True)
class ROIReport:
    """ROI-classifier outcome counts and pass rate."""

    n_rois_total: int
    n_rois_good: int
    n_rois_bad: int
    pass_rate: float


@dataclass(frozen=True)
class SpikeReport:
    """Neuron/event retention counts from the spike-processing stage."""

    n_neurons_in: int
    n_neurons_out: int
    n_spikes_raw: int
    n_spikes_kept: int
    mean_metrics: dict[str, float]


@dataclass(frozen=True)
class GroupingReport:
    """Strategies run and the group count produced by each."""

    strategies_run: list[str]
    n_groups: dict[str, int]
