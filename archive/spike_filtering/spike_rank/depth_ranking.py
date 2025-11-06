from typing import List
from data_structures import Valley
import numpy as np

def compute_valley_depths(valleys: List[Valley], cascade) -> List[Valley]:
    """
    Uses PRE-ASSIGNED .normalized_value for peaks and valleys.

    For each valley v:
      prev_depth  = v.previous_peak.normalized_value - v.normalized_value        (if previous_peak else 0)
      next_depth  = v.next_peak.normalized_value     - v.normalized_value        (if next_peak else 0)
      sum_depth   = prev_depth + next_depth
      avg_depth   = sum_depth / 2

      prev_sharp  = prev_depth / (v.index - previous_peak.index)  if dist>0 else 0
      next_sharp  = next_depth / (next_peak.index - v.index)      if dist>0 else 0
      sum_sharp   = prev_sharp + next_sharp
      avg_sharp   = sum_sharp / 2

    After computing raw values, each attribute is rescaled to [0,1] by:
        value / (max value across all valleys for the SAME attribute)
    If the max is 0 (or all zeros), the attribute remains 0 for all valleys.
    """
    def require_norm(where, obj):
        if obj is None or obj.normalized_value is None:
            raise ValueError(f"{where} has normalized_value=None. Run normalization first.")
        return float(obj.normalized_value)

    # ---------- Pass 1: compute raw attributes ----------
    for v in valleys:
        v_norm = require_norm("Valley", v)

        # previous side
        if v.previous_peak is not None:
            prev_norm = require_norm("Previous peak", v.previous_peak)
            v.previous_depth = prev_norm - v_norm
            dist_prev = v.index - v.previous_peak.index
            v.previous_depth_sharpness = (v.previous_depth / dist_prev) if dist_prev > 0 else 0.0
            v.previous_auc = float(np.trapz(cascade[v.previous_peak.index:v.index+1])) if dist_prev > 0 else 0.0
        else:
            v.previous_depth = 0.0
            v.previous_depth_sharpness = 0.0
            v.previous_auc = 0.0

        # next side
        if v.next_peak is not None:
            next_norm = require_norm("Next peak", v.next_peak)
            v.next_depth = next_norm - v_norm
            dist_next = v.next_peak.index - v.index
            v.next_depth_sharpness = (v.next_depth / dist_next) if dist_next > 0 else 0.0
            v.next_auc = float(np.trapz(cascade[v.index:v.next_peak.index+1])) if dist_next > 0 else 0.0
        else:
            v.next_depth = 0.0
            v.next_depth_sharpness = 0.0
            v.next_auc = 0.0

        # aggregates
        v.sum_auc = v.previous_auc + v.next_auc
        v.average_auc = v.sum_auc / 2.0 
        v.sum_depth = v.previous_depth + v.next_depth
        v.average_depth = v.sum_depth / 2.0
        v.sum_depth_sharpness = v.previous_depth_sharpness + v.next_depth_sharpness
        v.average_depth_sharpness = v.sum_depth_sharpness / 2.0

    # ---------- Pass 2: rescale each attribute to [0,1] ----------
    # Collect maxima (floor negatives at 0 for safety; depths/sharpness should be >=0 given definitions)
    attrs = [
        "previous_depth", "next_depth", "sum_depth", "average_depth",
        "previous_depth_sharpness", "next_depth_sharpness", "sum_depth_sharpness", "average_depth_sharpness"
    ]

    # Compute per-attribute max (non-negative)
    max_by_attr = {
        "previous_auc": max((v.previous_auc for v in valleys), default=1),
        "next_auc": max((v.next_auc for v in valleys), default=1),
        "sum_auc": max((v.sum_auc for v in valleys), default=1),
        "average_auc": max((v.average_auc for v in valleys), default=1),
        "previous_depth": max((v.previous_depth for v in valleys), default=1),
        "next_depth": max((v.next_depth for v in valleys), default=1),
        "sum_depth": max((v.sum_depth for v in valleys), default=1),
        "average_depth": max((v.average_depth for v in valleys), default=1),
        "previous_depth_sharpness": max((v.previous_depth_sharpness for v in valleys), default=1),
        "next_depth_sharpness": max((v.next_depth_sharpness for v in valleys), default=1),
        "sum_depth_sharpness": max((v.sum_depth_sharpness for v in valleys), default=1),
        "average_depth_sharpness": max((v.average_depth_sharpness for v in valleys), default=1),
    }


    # Safe divide helper
    def safe_div(val, denom):
        if denom <= 0.0:
            return 0.0
        return max(0.0, val) / denom  # also clamp to >=0 before scaling

    # Apply normalization in-place
    for v in valleys:
        v.depths = {  # dictionary attribute instead of separate attributes
            "prev_auc": safe_div(v.previous_auc, max_by_attr["previous_auc"]),
            "next_auc": safe_div(v.next_auc, max_by_attr["next_auc"]),
            "sum_auc": safe_div(v.sum_auc, max_by_attr["sum_auc"]),
            "average_auc": safe_div(v.average_auc, max_by_attr["average_auc"]), 
            "prev_depth": safe_div(v.previous_depth, max_by_attr["previous_depth"]),
            "next_depth": safe_div(v.next_depth, max_by_attr["next_depth"]),
            "sum_depth": safe_div(v.sum_depth, max_by_attr["sum_depth"]),
            "avg_depth": safe_div(v.average_depth, max_by_attr["average_depth"]),
            "prev_sharpness": safe_div(v.previous_depth_sharpness, max_by_attr["previous_depth_sharpness"]),
            "next_sharpness": safe_div(v.next_depth_sharpness, max_by_attr["next_depth_sharpness"]),
            "sum_sharpness": safe_div(v.sum_depth_sharpness, max_by_attr["sum_depth_sharpness"]),
            "avg_sharpness": safe_div(v.average_depth_sharpness, max_by_attr["average_depth_sharpness"]),
        }

    return valleys

def sort_valleys_by_metrics(valleys):
    """
    Create sorted lists of valleys by different depth/ sharpness metrics.
    Also return lists of the corresponding key values used for sorting.

    Returns:
        dict: {
            "prev_depth": (sorted_valleys, key_values),
            "next_depth": (sorted_valleys, key_values),
            "sum_depth": (sorted_valleys, key_values),
            "prev_sharp": (sorted_valleys, key_values),
            "next_sharp": (sorted_valleys, key_values),
            "sum_sharp": (sorted_valleys, key_values)
        }
    """
    results = {}

    # Helper to sort and also extract key values
    def sort_and_keys(key_func, label):
        sorted_list = sorted(valleys, key=key_func, reverse=False)  # reverse=True → bigger = sharper
        key_values = [key_func(v) for v in sorted_list]
        results[label] = (sorted_list, key_values)

    #AUCs
    sort_and_keys(lambda v: v.previous_auc, "prev_auc")
    sort_and_keys(lambda v: v.next_auc, "next_auc")
    sort_and_keys(lambda v: v.sum_auc, "sum_auc")
    # Depths
    sort_and_keys(lambda v: v.previous_depth, "prev_depth")
    sort_and_keys(lambda v: v.next_depth, "next_depth")
    sort_and_keys(lambda v: v.sum_depth, "sum_depth")

    # Sharpness
    sort_and_keys(lambda v: v.previous_depth_sharpness, "prev_sharp")
    sort_and_keys(lambda v: v.next_depth_sharpness, "next_sharp")
    sort_and_keys(lambda v: v.sum_depth_sharpness, "sum_sharp")

    return results

def assign_valley_ranks(valleys : List[Valley]):
    """
    For each valley, assign a dictionary of ranks based on different sorting styles.
    Keys = sorting style names
    Values = index (rank) of the valley in the sorted list.
    """
    # Define sorting methods
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
    for style, key_func in sorting_methods.items():
        # Sort valleys by this style (deepest valleys ranked first)
        sorted_valleys = sorted(valleys, key=key_func, reverse=False)
        # Assign ranks
        for idx, valley in enumerate(sorted_valleys):
            if valley.ranks is None:
                valley.ranks = {}
            valley.ranks[style] = idx
            valley.normal_ranks[style] = idx / (n - 1) if n > 1 else 0.0
    return valleys

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