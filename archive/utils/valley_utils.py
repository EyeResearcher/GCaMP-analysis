from typing import List
from data_classes.spike import Valley

def sort_and_rank_valleys(valleys:List[Valley]):
    """
    Sort valleys by various metrics, assign ranks and normalized ranks for each metric,
    and return sorted lists and key values for each metric.
    """
    sorting_methods = {
        "prev_auc": lambda v: v.previous_auc,
        "next_auc": lambda v: v.next_auc,    
        "sum_auc": lambda v: v.sum_auc,
        "prev_depth": lambda v: v.previous_depth,
        "next_depth": lambda v: v.next_depth,
        "sum_depth": lambda v: v.sum_depth,
        "prev_sharpness": lambda v: v.previous_depth_sharpness,
        "next_sharpness": lambda v: v.next_depth_sharpness,
        "sum_sharpness": lambda v: v.sum_depth_sharpness,
    }
    n = len(valleys)
    results = {}

    for style, key_func in sorting_methods.items():
        # Sort valleys by this metric
        sorted_valleys = sorted(valleys, key=key_func, reverse=False)
        key_values = [key_func(v) for v in sorted_valleys]
        results[style] = (sorted_valleys, key_values)
        # Assign ranks and normalized ranks
        for idx, valley in enumerate(sorted_valleys):
            if not hasattr(valley, 'ranks') or valley.ranks is None:
                valley.ranks = {}
            if not hasattr(valley, 'normal_ranks') or valley.normal_ranks is None:
                valley.normal_ranks = {}
            valley.ranks[style] = idx
            valley.normal_ranks[style] = idx / (n - 1) if n > 1 else 0.0

    return valleys, results