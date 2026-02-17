from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from gcamp_analysis.data_classes.neuron_group import NeuronGroup

def compute_group_summary_rows(
    groups: List[NeuronGroup],
    *,
    method: str,
    corr_matrix: np.ndarray | None,
    dtw_matrix: np.ndarray | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for g in groups:
        # Use neuron.summary_stats (stable contract)
        ss = [getattr(n, "summary_stats", {}) for n in g.neurons]
        df = pd.DataFrame(ss)

        rates = df.get("spike_frequency", pd.Series(dtype=float))
        num_spikes = df.get("number_of_spikes", pd.Series(dtype=float))

        mean_of_means = df.filter(like="mean_").mean(numeric_only=True).to_dict()

        row = {
            "group_id": g.group_id,
            "method": method,
            "number_neurons": int(g.size),
            "neuron_indices": list(getattr(g, "neuron_indices", [])),
            "filtered_idxs": list(getattr(g, "filtered_idxs", [])),
            "spike_rate": float(np.nanmean(rates)) if len(rates) else 0.0,
            "number_of_spikes": float(np.nanmean(num_spikes)) if len(num_spikes) else 0.0,
            **mean_of_means,
        }

        # group connectivity summaries
        try:
            row["mean_corr"] = g.group_mean_corr(corr_matrix) if corr_matrix is not None else np.nan
        except Exception:
            row["mean_corr"] = np.nan
        try:
            row["mean_dtw"] = g.group_mean_dtw(dtw_matrix) if dtw_matrix is not None else np.nan
        except Exception:
            row["mean_dtw"] = np.nan

        # attach thresholds if present on group
        if getattr(g, "t_win", None) is not None:
            row["time_window"] = g.t_win
        if getattr(g, "corr_thresh", None) is not None:
            row["corr_thresh"] = g.corr_thresh
        if getattr(g, "dtw_thresh", None) is not None:
            row["dtw_thresh"] = g.dtw_thresh

        rows.append(row)
    return rows
