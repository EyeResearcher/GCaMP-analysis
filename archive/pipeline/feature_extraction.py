"""Neuron-level feature aggregation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from utils.spike_utils import compute_area_under_curve

from .config import FeatureExtractionConfig
from .spike_filtering import SpikeFilteringResult


@dataclass(slots=True)
class FeatureExtractionResult:
    neuron_table: pd.DataFrame
    spike_trains: List[np.ndarray]


def _aggregate_neuron_metrics(
    neuron_idx: int,
    trace: np.ndarray,
    spikes: np.ndarray,
    neuron_rows: pd.DataFrame,
    duration_s: float,
    baseline_percentile: float,
) -> Dict[str, float]:
    baseline = float(np.percentile(trace, baseline_percentile)) if trace.size else 0.0
    num_spikes = spikes.size
    spike_frequency = num_spikes / duration_s if duration_s > 0 else 0.0

    accepted_rows = neuron_rows[neuron_rows["accepted"]]
    avg_peak = accepted_rows["fluorescence_peak"].mean() if not accepted_rows.empty else np.nan
    avg_prob = accepted_rows["prob_height"].mean() if "prob_height" in accepted_rows.columns and not accepted_rows.empty else np.nan
    avg_prom = accepted_rows["prominence"].mean() if "prominence" in accepted_rows.columns and not accepted_rows.empty else np.nan
    avg_rise = accepted_rows["rise_slope"].mean() if "rise_slope" in accepted_rows.columns and not accepted_rows.empty else np.nan
    avg_decay = accepted_rows["decay_tau"].mean() if "decay_tau" in accepted_rows.columns and not accepted_rows.empty else np.nan
    total_window_auc = accepted_rows["window_auc"].sum() if "window_auc" in accepted_rows.columns and not accepted_rows.empty else 0.0

    total_auc = compute_area_under_curve(trace)
    auc_per_spike = total_auc / num_spikes if num_spikes else np.nan

    return {
        "neuron_index": neuron_idx,
        "num_spikes": num_spikes,
        "spike_frequency": spike_frequency,
        "baseline": baseline,
        "avg_peak": avg_peak,
        "avg_prob": avg_prob,
        "avg_prominence": avg_prom,
        "avg_rise_slope": avg_rise,
        "avg_decay_tau": avg_decay,
        "total_window_auc": total_window_auc,
        "total_auc": total_auc,
        "auc_per_spike": auc_per_spike,
    }


def compute_neuron_features(
    fluorescence: np.ndarray,
    filtering_result: SpikeFilteringResult,
    ops: Dict,
    config: FeatureExtractionConfig,
) -> FeatureExtractionResult:
    """Aggregate neuron-level statistics and produce spike trains."""

    fs = float(ops.get("fs", 30.0))
    nframes = int(ops.get("nframes", fluorescence.shape[-1]))
    duration_s = nframes / fs if fs > 0 else 0.0

    spike_trains: List[np.ndarray] = []
    rows: List[Dict[str, float]] = []

    table = filtering_result.table if filtering_result.table is not None else pd.DataFrame()

    for neuron_idx in range(fluorescence.shape[0]):
        spikes = filtering_result.spikes_for_neuron(neuron_idx)
        neuron_rows = table[table["neuron_index"] == neuron_idx] if not table.empty else pd.DataFrame()
        metrics = _aggregate_neuron_metrics(
            neuron_idx,
            fluorescence[neuron_idx],
            spikes,
            neuron_rows,
            duration_s,
            config.baseline_percentile,
        )
        rows.append(metrics)
        spike_trains.append(spikes)

    neuron_table = pd.DataFrame.from_records(rows)
    neuron_table.index.name = "neuron_index"

    if config.min_spike_count > 0:
        neuron_table = neuron_table[neuron_table["num_spikes"] >= config.min_spike_count]

    return FeatureExtractionResult(neuron_table=neuron_table, spike_trains=spike_trains)
