from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
from data_classes.neuron_group import NeuronGroup
from grouping_processing.summary import compute_group_summary_rows

def compare_groupings(
    *,
    sttc_groups: List[NeuronGroup],
    dtw_groups: List[NeuronGroup],
    sttc_matrix: np.ndarray | None,
    dtw_matrix: np.ndarray | None,
    neurons: list,
) -> Dict[str, Any]:
    # membership arrays for agreement
    sttc_membership = np.full(len(neurons), -1, dtype=int)
    for i, group in enumerate(sttc_groups):
        for neuron in group.neurons:
            sttc_membership[neurons.index(neuron)] = i

    if not dtw_groups:
        combined = compute_group_summary_rows(
            sttc_groups, method="sttc", sttc_matrix=sttc_matrix, dtw_matrix=dtw_matrix
        )
        return {
            "n_sttc_groups": len(sttc_groups),
            "sttc_groups": sttc_groups,
            "n_dtw_groups": 0,
            "dtw_groups": [],
            "agreement": 0.0,
            "combined_stats": combined,
        }

    dtw_membership = np.full(len(neurons), -1, dtype=int)
    for i, group in enumerate(dtw_groups):
        for neuron in group.neurons:
            dtw_membership[neurons.index(neuron)] = i

    agreement = float(np.mean(sttc_membership == dtw_membership))

    sttc_rows = compute_group_summary_rows(
        sttc_groups, method="sttc", sttc_matrix=sttc_matrix, dtw_matrix=dtw_matrix
    )
    dtw_rows = compute_group_summary_rows(
        dtw_groups, method="dtw", sttc_matrix=sttc_matrix, dtw_matrix=dtw_matrix
    )

    return {
        "n_sttc_groups": len(sttc_groups),
        "sttc_groups": sttc_groups,
        "n_dtw_groups": len(dtw_groups),
        "dtw_groups": dtw_groups,
        "agreement": agreement,
        "combined_stats": sttc_rows + dtw_rows,
    }
