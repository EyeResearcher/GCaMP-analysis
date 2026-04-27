"""Grouping strategies implemented as pure functions.

The service layer is responsible for extracting state from ``Video`` and
writing results back. Strategy functions operate on immutable inputs and
return plain dict payloads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict

import numpy as np

from gcamp_analysis.grouping_processing.similarity import (
    compute_dtw_matrix_from_traces,
    align_light_evoked,
    max_crosscorr_similarity,
    compute_sttc_matrix,
    compute_combined_similarities,
)
from gcamp_analysis.grouping_processing.clustering import (
    cluster_hierarchical,
    build_groups_from_labels,
    light_evoked_cluster,
)


def run_combined_grouping(
    traces: np.ndarray,
    spike_trains: list,
    t_stop: float,
    neuron_indices: np.ndarray,
    *,
    corr_config: Dict[str, Any] | None = None,
    sttc_config: Dict[str, Any] | None = None,
    cluster_config: Dict[str, Any] | None = None,
    **_kwargs,
) -> dict:
    """Compute corr + STTC matrices, combine, cluster, and return groups."""
    corr_config = corr_config or {}
    sttc_config = sttc_config or {}
    cluster_config = cluster_config or {}

    max_lag = int(corr_config.get("max_lag", 5))
    dt = float(sttc_config.get("dt", 1.75))

    corr_mat = max_crosscorr_similarity(traces, max_lag=max_lag)
    sttc_mat = compute_sttc_matrix(spike_trains, dt, 0.0, t_stop)
    combined = compute_combined_similarities(corr_mat, sttc_mat)

    cluster_param = cluster_config.get("cluster_param", 0.65)
    linkage_method = str(cluster_config.get("linkage_method", "average"))
    cluster_criterion = str(cluster_config.get("cluster_criterion", "distance"))
    min_group_size = int(cluster_config.get("min_group_size", 2))

    _Z, labels, _order, _mat_ordered, _labels_ordered = cluster_hierarchical(
        combined,
        linkage_method=linkage_method,
        cluster_criterion=cluster_criterion,
        cluster_param=cluster_param,
    )

    row_indices = np.arange(len(neuron_indices))
    group_dicts = build_groups_from_labels(
        labels,
        row_indices,
        neuron_indices,
        min_group_size=min_group_size,
    )

    return {
        "groups": group_dicts,
        "matrix": combined,
        "config_label": f"combined_dt{dt}_cp{cluster_param}",
        "metadata": {"corr_matrix": corr_mat, "sttc_matrix": sttc_mat},
    }


def run_dtw_grouping(
    config: Dict[str, Any],
    *,
    dtw_traces: np.ndarray,
    neuron_indices: np.ndarray,
    **_kwargs,
) -> dict:
    """Compute a DTW matrix from pre-selected traces and cluster on it."""
    if dtw_traces.size == 0 or len(neuron_indices) < 2:
        return {"groups": [], "matrix": None, "config_label": "dtw_empty"}

    down = int(config.get("downsample_factor", 3))
    gpu = bool(config.get("use_gpu", True))
    link = str(config.get("linkage_method", "average"))
    pctl = int(config.get("distance_percentile", 30))
    min_group = int(config.get("min_group_size", 2))

    dtw = compute_dtw_matrix_from_traces(
        dtw_traces,
        downsample_factor=down,
        use_gpu=gpu,
    )
    if dtw is None:
        return {"groups": [], "matrix": None, "config_label": "dtw_skipped"}

    dtw = np.asarray(dtw, dtype=float)
    nonzero = dtw[dtw > 0]
    if nonzero.size == 0:
        return {"groups": [], "matrix": dtw, "config_label": "dtw_empty"}

    thresh = float(np.percentile(nonzero, pctl))
    groups = _build_groups_from_distance_threshold(dtw, neuron_indices, thresh, min_group)
    return {"groups": groups, "matrix": dtw, "config_label": "dtw"}


def _build_groups_from_distance_threshold(
    distance_matrix: np.ndarray,
    neuron_indices: np.ndarray,
    threshold: float,
    min_group_size: int,
) -> list[dict]:
    """Build plain group dicts from connected components in a distance graph."""
    n = int(distance_matrix.shape[0])
    if n == 0:
        return []

    adjacency = (distance_matrix <= threshold) & np.isfinite(distance_matrix)
    np.fill_diagonal(adjacency, True)
    visited = np.zeros(n, dtype=bool)
    groups: list[dict] = []
    next_id = 1

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        members: list[int] = []
        visited[start] = True
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in np.where(adjacency[current])[0]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        members = sorted(members)
        if len(members) < min_group_size:
            continue
        groups.append(
            {
                "group_id": next_id,
                "row_indices": members,
                "neuron_indices": neuron_indices[members].tolist(),
                "n_neurons": len(members),
            }
        )
        next_id += 1

    return groups


def _make_sched(start: int, interval: int, frames: int) -> list[int]:
    pulses = []
    current = start
    while current < frames:
        pulses.append(current)
        current += interval
    return pulses


def _resolve_light_schedule(
    video_id: str,
    n_frames: int,
    config: Dict[str, Any],
    schedule_overrides: Dict[str, list[int]],
) -> list[int]:
    if video_id in schedule_overrides:
        return schedule_overrides[video_id]
    if config.get("start") is not None and config.get("interval") is not None:
        return _make_sched(int(config["start"]), int(config["interval"]), int(n_frames))
    if config.get("schedule"):
        return list(config["schedule"])
    raise ValueError("Either 'start' and 'interval' or 'schedule' must be specified in the config.")


def run_light_evoked_grouping(
    config: Dict[str, Any],
    *,
    active_neurons: list,
    light_evoked_traces: np.ndarray,
    n_frames: int,
    video_id: str,
    schedule_overrides: Dict[str, list[int]] | None = None,
    **_kwargs,
) -> dict:
    """Light-evoked response grouping strategy using service-prepared inputs."""
    schedule_overrides = schedule_overrides or {}
    sched = _resolve_light_schedule(video_id, n_frames, config, schedule_overrides)
    bin_size = int(config.get("bin_size", 3))
    prominence = config.get("prominence", None)
    activated = align_light_evoked(
        light_evoked_traces,
        bin_size=bin_size,
        schedule=sched,
        n_frames=n_frames,
        prominence=prominence,
    )
    groups = light_evoked_cluster(
        active_neurons,
        activated,
        n_pulses=len(sched),
        schedule=sched,
        bin_size=bin_size,
        response_window=int(config.get("response_window", 10)),
    )
    return {
        "groups": groups,
        "matrix": activated,
        "config_label": "light_evoked",
        "metadata": {"schedule": sched, "bin_size": bin_size, "prominence": prominence},
    }


@dataclass
class LightEvokedStrategy:
    """Backward-compatible wrapper around the pure strategy function."""

    name: str = "light-evoked"
    SCHEDULE_OVERRIDES: dict = field(default_factory=lambda: {
        "5732L-5": [33, 65, 93, 116, 153, 192],
    })

    def _make_sched(self, start, interval, frames):
        return _make_sched(start, interval, frames)

    def compute(self, video, config: Dict[str, Any], **kwargs) -> dict:
        raw = run_light_evoked_grouping(
            config,
            active_neurons=list(video.neurons),
            light_evoked_traces=np.asarray(video.norm_sm_f, dtype=float),
            n_frames=int(video.n_frames),
            video_id=str(video.video_id),
            schedule_overrides=self.SCHEDULE_OVERRIDES,
            **kwargs,
        )
        return SimpleNamespace(**raw)


StrategyEntry = Callable[..., dict]

STRATEGY_REGISTRY: Dict[str, StrategyEntry] = {
    "combined": run_combined_grouping,
    "dtw": run_dtw_grouping,
    "light-evoked": run_light_evoked_grouping,
}
