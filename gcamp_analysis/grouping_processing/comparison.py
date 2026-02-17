from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
from gcamp_analysis.data_classes.neuron_group import NeuronGroup
from gcamp_analysis.grouping_processing.summary import compute_group_summary_rows

def compare_groupings(
    *,
    corr_groups: List[NeuronGroup],
    dtw_groups: List[NeuronGroup],
    corr_matrix: np.ndarray | None,
    dtw_matrix: np.ndarray | None,
    neurons: list,
) -> Dict[str, Any]:
    
    corr_membership = np.full(len(neurons), -1, dtype=int)
    for i, group in enumerate(corr_groups):
        for neuron in group.neurons:
            corr_membership[neurons.index(neuron)] = i

    if not dtw_groups:
        combined = compute_group_summary_rows(
            corr_groups, method="corr", corr_matrix=corr_matrix, dtw_matrix=dtw_matrix
        )
        return {
            "n_corr_groups": len(corr_groups),
            "corr_groups": corr_groups,
            "n_dtw_groups": 0,
            "dtw_groups": [],
            "agreement": 0.0,
            "combined_stats": combined,
        }

    dtw_membership = np.full(len(neurons), -1, dtype=int)
    for i, group in enumerate(dtw_groups):
        for neuron in group.neurons:
            dtw_membership[neurons.index(neuron)] = i

    agreement = float(np.mean(corr_membership == dtw_membership))

    corr_rows = compute_group_summary_rows(
        corr_groups, method="corr", corr_matrix=corr_matrix, dtw_matrix=dtw_matrix
    )
    dtw_rows = compute_group_summary_rows(
        dtw_groups, method="dtw", corr_matrix=corr_matrix, dtw_matrix=dtw_matrix
    )

    return {
        "n_corr_groups": len(corr_groups),
        "corr_groups": corr_groups,
        "n_dtw_groups": len(dtw_groups),
        "dtw_groups": dtw_groups,
        "agreement": agreement,
        "combined_stats": corr_rows + dtw_rows,
    }
