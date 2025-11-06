"""Neuron grouping via STTC and spatial heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import quantities as pq
import neo

from analysis.correlation import compute_sttc_matrix
from analysis.grouping import main_grouping

from .config import GroupingConfig


@dataclass(slots=True)
class GroupingResult:
    sttc_matrix: np.ndarray
    neuron_groups: List[List[int]]
    dispersion_metrics: Dict[str, float]
    average_group_sttc: List[float]


def _to_spike_trains(spike_indices: List[np.ndarray], fs: float, nframes: int) -> List[neo.SpikeTrain]:
    spike_trains: List[neo.SpikeTrain] = []
    duration = (nframes / fs) * pq.s
    for spikes in spike_indices:
        if spikes.size == 0:
            spike_trains.append(
                neo.SpikeTrain(np.array([]) * pq.s, t_stop=duration)
            )
            continue
        times = (spikes / fs) * pq.s
        spike_trains.append(neo.SpikeTrain(times, t_stop=duration))
    return spike_trains


def group_neurons(
    spike_indices: List[np.ndarray],
    summary_stat,
    fs: float,
    nframes: int,
    config: GroupingConfig,
) -> GroupingResult:
    """Compute STTC matrix and group neurons using graph-based clustering."""

    spike_trains = _to_spike_trains(spike_indices, fs=fs, nframes=nframes)
    sttc = compute_sttc_matrix(spike_trains)

    groups, dispersion, avg_sttc = main_grouping(sttc, summary_stat)

    # Enforce minimum group size if specified
    if config.min_group_size > 1:
        groups = [grp for grp in groups if len(grp) >= config.min_group_size]

    return GroupingResult(
        sttc_matrix=sttc,
        neuron_groups=groups,
        dispersion_metrics=dispersion,
        average_group_sttc=avg_sttc,
    )
