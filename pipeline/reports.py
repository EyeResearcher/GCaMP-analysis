from dataclasses import dataclass

@dataclass(frozen=True)
class TraceReport:
    n_rois: int
    n_frames: int
    fs: float

@dataclass(frozen=True)
class ROIReport:
    n_rois_total: int
    n_rois_good: int
    n_rois_bad: int
    pass_rate: float

@dataclass(frozen=True)
class SpikeReport:
    n_neurons_in: int
    n_neurons_out: int
    n_spikes_raw: int
    n_spikes_kept: int
    mean_metrics: dict[str, float]

@dataclass(frozen=True)
class GroupingReport:
    method: str
    n_groups: int
    agreement: float | None
