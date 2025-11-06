"""Spike detection utilities for the modular pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from utils.spike_utils import find_spikes, window_spike_transients

from .config import SpikeDetectionConfig


@dataclass(slots=True)
class NeuronSpikes:
    """Spike detection results for a single neuron."""

    neuron_index: int
    spike_prob_indices: np.ndarray
    spike_prob_values: np.ndarray
    left_prominences: np.ndarray
    fluorescence_indices: np.ndarray
    fluorescence_values: np.ndarray
    smoothed_prob: np.ndarray
    windows: np.ndarray

    def as_binary_train(self, length: int) -> np.ndarray:
        train = np.zeros(length, dtype=int)
        train[self.fluorescence_indices.astype(int)] = 1
        return train


@dataclass(slots=True)
class SpikeDetectionResult:
    spikes_per_neuron: List[NeuronSpikes]

    @property
    def neuron_count(self) -> int:
        return len(self.spikes_per_neuron)


def detect_spikes(
    fluorescence: np.ndarray,
    cascade_prob: np.ndarray,
    config: SpikeDetectionConfig,
) -> SpikeDetectionResult:
    """Detect candidate spikes for all neurons in a video."""

    spikes: List[NeuronSpikes] = []
    for neuron_idx in range(fluorescence.shape[0]):
        prob_trace = cascade_prob[neuron_idx]
        fluo_trace = fluorescence[neuron_idx]

        (prob_idx, prob_vals, left_proms, fluoro_idx, fluoro_vals), smoothed = find_spikes(
            spike_prob_trace=prob_trace,
            raw_fluorescence=fluo_trace,
            sigma=config.prob_sigma,
            window_radius=config.window_size,
            edge=config.edge_trim,
        )

        mask = np.ones(prob_idx.shape, dtype=bool)
        if config.min_prominence is not None:
            mask &= left_proms >= config.min_prominence

        if config.min_distance and prob_idx.size:
            distance_mask = np.ones_like(mask)
            last_index = -np.inf
            for i, idx in enumerate(fluoro_idx):
                if last_index != -np.inf and (idx - last_index) < config.min_distance:
                    distance_mask[i] = False
                    continue
                last_index = idx
            mask &= distance_mask.astype(bool)

        prob_idx = prob_idx[mask]
        prob_vals = prob_vals[mask]
        left_proms = left_proms[mask]
        fluoro_idx = fluoro_idx[mask]
        fluoro_vals = fluoro_vals[mask]

        windows = window_spike_transients(fluo_trace, fluoro_idx.astype(int)) if fluoro_idx.size else np.empty((0, 3), dtype=int)

        spikes.append(
            NeuronSpikes(
                neuron_index=neuron_idx,
                spike_prob_indices=prob_idx,
                spike_prob_values=prob_vals,
                left_prominences=left_proms,
                fluorescence_indices=fluoro_idx,
                fluorescence_values=fluoro_vals,
                smoothed_prob=smoothed,
                windows=np.asarray(windows, dtype=int),
            )
        )

    return SpikeDetectionResult(spikes_per_neuron=spikes)
