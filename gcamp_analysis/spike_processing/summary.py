
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from collections import OrderedDict
import numpy as np
import pandas as pd

from gcamp_analysis.data_classes.neuron import Neuron


@dataclass
class NeuronSpikeSummary:
    """
    Turn per-spike stats into a per-neuron summary dict used for summary_df rows.
    This mirrors your current behavior: mean_{stat} and var_{stat} for each metric,
    plus metadata columns like neuron_idx, filtered_index, spike_frequency, number_of_spikes.
    """

    def summarize(self, neuron: Neuron, f_trace_raw: np.ndarray) -> Dict[str, Any]:
        # If no spike stats, return empty row (caller decides whether to keep neuron)
        if not neuron.all_spk_stats:
            neuron.summary_stats = {}
            return {}

        stats_df = pd.DataFrame(neuron.all_spk_stats)

        # Spike frequency (Hz) computed from neuron trace length
        f_trace = getattr(neuron, "f_trace", None)
        n_frames = len(f_trace) if f_trace is not None else 0
        fs = float(getattr(neuron, "fs", 30.0))
        spike_frequency = float(len(neuron.spikes) / (n_frames / fs)) if n_frames > 0 else 0.0

        # Build ordered row
        summary = OrderedDict()
        summary["neuron_idx"] = int(getattr(neuron, "index", -1))
        summary["filtered_index"] = int(getattr(neuron, "filtered_index", -1))
        summary["spike_frequency"] = spike_frequency
        summary["number_of_spikes"] = int(len(neuron.spikes))

        # Helpful raw lists (kept consistent with your current schema)
        summary["spike_indices"] = list(getattr(neuron, "peaks_filtered", []))
        summary["spike_values_normalized"] = [float(sp.f_value) for sp in neuron.spikes if sp.f_value is not None]

        raw = np.asarray(f_trace_raw, dtype=float).reshape(-1)
        summary["spike_values_raw"] = [
            float(raw[sp.sm_f_idx]) for sp in neuron.spikes
            if sp.sm_f_idx is not None and 0 <= int(sp.sm_f_idx) < raw.size
        ]

        # Mean/variance columns for each per-spike statistic
        for col in stats_df.columns:
            x = pd.to_numeric(stats_df[col], errors="coerce")
            summary[f"mean_{col}"] = float(x.mean())
            summary[f"var_{col}"] = float(x.var())  # ddof=1 default in pandas

        neuron.summary_stats = dict(summary)
        return neuron.summary_stats
