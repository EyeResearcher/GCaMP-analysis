"""Spike feature computation and classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils.spike_utils import compute_spike_constants, window_spike_transients

from .config import SpikeFilteringConfig
from .spike_detection import NeuronSpikes


@dataclass(slots=True)
class SpikeFilteringResult:
    """Filtered spike table with classification flags."""

    table: pd.DataFrame
    accepted_spike_indices: Dict[int, np.ndarray]

    def spikes_for_neuron(self, neuron_index: int) -> np.ndarray:
        return self.accepted_spike_indices.get(neuron_index, np.array([], dtype=int))


def _compute_single_spike_features(
    neuron_idx: int,
    spike_idx: int,
    spike_prob_idx: int,
    spike_prob_value: float,
    prominence: float,
    fluorescence_trace: np.ndarray,
    spike_indices: np.ndarray,
    window_triplets: Sequence[tuple[int, int, int]],
    fs: float,
) -> Dict[str, float]:
    start, peak, end = window_triplets[spike_idx]
    segment = fluorescence_trace[start : end + 1]
    baseline = float(np.percentile(segment, 10)) if segment.size else 0.0
    peak_value = float(fluorescence_trace[peak])
    width = float(max(end - start, 1))
    window_auc = float(np.trapz(segment))

    try:
        rise_slope, decay_tau = compute_spike_constants(fluorescence_trace, peak_idx=peak, fs=fs)
    except Exception:
        rise_slope, decay_tau = (np.nan, np.nan)

    return {
        "neuron_index": neuron_idx,
        "spike_number": spike_idx,
        "frame_index": int(peak),
        "prob_index": int(spike_prob_idx),
        "prob_height": float(spike_prob_value),
        "prominence": float(prominence),
        "fluorescence_peak": peak_value,
        "baseline": baseline,
        "baseline_delta": peak_value - baseline,
        "window_width": width,
        "window_auc": window_auc,
        "rise_slope": float(rise_slope),
        "decay_tau": float(decay_tau),
    }


def filter_spikes(
    fluorescence: np.ndarray,
    detection: List[NeuronSpikes],
    fs: float,
    config: SpikeFilteringConfig,
    spike_model=None,
) -> SpikeFilteringResult:
    """Compute spike features and apply spike-level classifier."""

    records: List[Dict[str, float]] = []
    for neuron_idx, neuron_spikes in enumerate(detection):
        spike_indices = neuron_spikes.fluorescence_indices.astype(int)
        if spike_indices.size == 0:
            continue
        windows = neuron_spikes.windows
        if windows is None or len(windows) != spike_indices.size:
            windows = window_spike_transients(fluorescence[neuron_idx], spike_indices)
        for spike_number, prob_idx in enumerate(neuron_spikes.spike_prob_indices):
            prob_val = neuron_spikes.spike_prob_values[spike_number]
            prom = neuron_spikes.left_prominences[spike_number]
            feature_row = _compute_single_spike_features(
                neuron_idx=neuron_idx,
                spike_idx=spike_number,
                spike_prob_idx=int(prob_idx),
                spike_prob_value=float(prob_val),
                prominence=float(prom),
                fluorescence_trace=fluorescence[neuron_idx],
                spike_indices=spike_indices,
                window_triplets=windows,
                fs=fs,
            )
            records.append(feature_row)

    if not records:
        return SpikeFilteringResult(
            table=pd.DataFrame(
                columns=[
                    "neuron_index",
                    "spike_number",
                    "frame_index",
                    "prob_index",
                    "prob_height",
                    "prominence",
                    "fluorescence_peak",
                    "baseline",
                    "baseline_delta",
                    "window_width",
                    "window_auc",
                    "rise_slope",
                    "decay_tau",
                    "accepted",
                    "probability",
                ]
            ),
            accepted_spike_indices={},
        )

    table = pd.DataFrame.from_records(records)

    probability = None
    accepted = np.ones(len(table), dtype=bool)
    if spike_model is not None and not table.empty:
        feature_columns = [col for col in config.feature_columns if col in table.columns]
        if not feature_columns:
            raise ValueError("No overlapping feature columns between configuration and spike table")
        model_input = table[feature_columns].to_numpy()
        nan_mask = np.isnan(model_input).any(axis=1)
        if np.any(nan_mask):
            diagnostic_cols = ["neuron_index", "spike_number", "frame_index"] + feature_columns
            diagnostic_df = table.loc[nan_mask, diagnostic_cols]
            preview = diagnostic_df.head().to_string(index=False)
            raise ValueError(
                "NaN detected in spike feature matrix prior to classification. "
                "Rows with NaNs:\n"
                f"{preview}\n"
                "Consider inspecting the saved diagnostic rows or pre-imputing values."
            )
        predictions = spike_model.predict(model_input)
        accepted = predictions.astype(bool)
        if hasattr(spike_model, "predict_proba"):
            probability = spike_model.predict_proba(model_input)[:, -1]
            if config.probability_threshold is not None:
                accepted = probability >= config.probability_threshold
    table["accepted"] = accepted
    if probability is not None:
        table["probability"] = probability

    accepted_spikes: Dict[int, List[int]] = {}
    for idx, row in table.iterrows():
        if not row["accepted"]:
            continue
        accepted_spikes.setdefault(int(row["neuron_index"]), []).append(int(row["frame_index"]))

    accepted_spike_indices = {k: np.array(v, dtype=int) for k, v in accepted_spikes.items()}

    return SpikeFilteringResult(table=table, accepted_spike_indices=accepted_spike_indices)
