import numpy as np
from typing import List, Tuple

from scipy.stats import skew, zscore, kurtosis

def zscore_features(features):
    """
    Z-score the features for each spike.
    """
    if not isinstance(features, np.ndarray) and not isinstance(features, list):
        raise ValueError("Features must be a numpy array or a list of arrays.")
    features = np.array(features) if isinstance(features, list) else features
    return zscore(features, axis=0) if features.ndim > 1 else zscore(features)

def get_windowed_trace(raw_trace, trace, i_peak, edge = 32):
    """
    Get a windowed trace around a peak and its previous local minimum.
    Args:
        trace (np.ndarray): The trace from which to extract the window.
        i_peak (int): The index of the peak in the trace.
    Returns:
        window (np.ndarray): The windowed trace.
    """
    start = max(find_local_minimum(trace, i_peak, left = True), edge)
    end = min(find_local_minimum(trace, i_peak, right=True), len(trace) - edge)
    left_window = trace[start:i_peak]
    window = trace[start:end]
    right_window = trace[i_peak:end]
    left_window_raw, window_raw, right_window_raw = raw_trace[start:i_peak], raw_trace[start:end], raw_trace[i_peak:end]
    return (left_window, window, right_window), (left_window_raw, window_raw, right_window_raw)
    
def find_local_minimum(trace, i_peak, left = False, right = False):
    """ Find the previous local minimum in a trace before a given peak index.
    Args:
        trace (np.ndarray): The trace in which to find the local minimum.
        i_peak (int): The index of the peak in the trace.
    Returns:
        j (int): The index of the previous local minimum, or None if not found."""
    
    start, end, step = (i_peak - 1, 0, -1) if left else (i_peak + 1, len(trace), 1)
    for j in range(start, end, step):
        if trace[j] < trace[j-1] and trace[j] < trace[j+1]:
            return j
    return 0  # no local minimum found, default to start of trace

def find_max_second_derivative(window):
    d1 = np.diff(window) if len(window) > 1 else np.array([0])
    d2 = np.diff(d1) if len(d1) > 1 else np.array([0])
   
    max_second_derivative = np.max(d2)
    return max_second_derivative

def weight_valley_ranks_for_peak(valleys : List, peak , sigma):
    import math
    """
    For a single peak:
      - compute Gaussian weight per valley: exp(-0.5 * ((v.index - peak.index)/sigma)^2)
      - populate valley.weight
      - populate valley.weighted_ranks[style] = valley.ranks[style] * valley.weight

    Returns:
      valleys (updated in-place)
    """
    if sigma is None or sigma <= 0:
        raise ValueError("sigma must be > 0")

    p_idx = int(peak.idx_prob)

    for v in valleys:
        print(v.index, type(v.index))
        d = (int(v.index) - p_idx) / float(sigma)
        w = math.exp(-0.5 * (d * d))
        v.weight = w  # (overwritten each time you call this for a new peak)

        # Ensure dict exists
        if getattr(v, "weighted_ranks", None) is None:
            v.weighted_ranks = {}
        if getattr(v, "ranks", None) is None:
            # If ranks aren't present, treat as empty
            v.ranks = {}

        # Weight every available sorting style for this valley
        v.weighted_ranks.clear()
        for style, rank_val in v.ranks.items():
            # rank_val is the ordinal rank (int or float); multiply by the weight
            v.weighted_ranks[style] = float(rank_val) * w

    return valleys

def sum_preceding_valley_contributions(peak , valleys : List, styles):
    """
    Compute the sum of weights and weighted ranks for valleys
    that occur before the given peak (valley.index < peak.index).

    Parameters
    ----------
    peak : Peak
        Peak object with an 'index' attribute.
    valleys : list of Valley
        List of Valley objects, each with 'index', 'weight',
        and 'weighted_ranks' (dict of style → value).
    styles : set of styles

    Returns
    -------
    tuple
        (total_weight, summed_weighted_ranks)
        where summed_weighted_ranks is a dict of style → summed value.
    """
    total_weight = 0.0
    summed_weighted_ranks = dict(zip(styles, np.zeros(len(styles))))
    summed_weighted_next_depths = 0.0    
    summed_weighted_next_sharpness = 0.0
    summed_weighted_next_auc = 0.0
    for idx, valley in enumerate(valleys):
        print(idx)
        if valley.index < peak.idx_prob:
            # Add valley's weight
            total_weight += valley.weight

            # Add weighted rank contributions per style
            summed_weighted_next_depths += valley.weight * valley.next_depth if hasattr(valley, "sum_depth") and valley.sum_depth is not None else 0.0
            summed_weighted_next_sharpness += valley.weight * valley.next_depth_sharpness if hasattr(valley, "sum_depth_sharpness") and valley.sum_depth_sharpness is not None else 0.0
            summed_weighted_next_auc += valley.weight * valley.next_auc if hasattr(valley, "next_auc") and valley.next_auc is not None else 0.0
            if hasattr(valley, "weighted_ranks"):
                for style, value in valley.weighted_ranks.items():
                   summed_weighted_ranks[style] += value

    return total_weight, summed_weighted_ranks, summed_weighted_next_sharpness, summed_weighted_next_depths, summed_weighted_next_auc

def compute_peak_rank_score_from_weighted(valleys : List, peak  ):
    """
    Given valleys already weighted for *this* peak (via weight_valley_ranks_for_peak),
    compute per-style weighted average of ranks:

        peak.rank_score[style] = sum_v (v.weighted_ranks[style]) / sum_v (v.weight)

    Returns:
      rankn_scores (dict of style → rank score),
      peak (with no changes),
        valleys (with no changes)
    """

    # Collect styles from any valley that has weighted_ranks
    styles = set()
    for v in valleys:
        if getattr(v, "weighted_ranks", None):
            styles.update(v.weighted_ranks.keys())
    weights_pre, weighted_ranks_pre, weighted_sharp, weighted_depth, weighted_auc = sum_preceding_valley_contributions(peak, valleys, styles)
    print("weights_pre", weights_pre)
    print("weighted_ranks_pre", weighted_ranks_pre)
    # Sum of weights across valleys
    total_w = sum(getattr(v, "weight", 0.0) for v in valleys)

    rank_scores = {}
    if total_w == 0.0:
        # No weight support → assign None or 0.0 per your preference
        for style in styles:
            rank_scores[style] = 0.0
            rank_scores[f"{style}_preonly"] = 0.0
        return rank_scores, peak, valleys
    final_style_set = set(styles)
    for style in styles:
        weighted_sum = 0.0
        for v in valleys:
            if getattr(v, "weighted_ranks", None) and style in v.weighted_ranks:
                weighted_sum += v.weighted_ranks[style]
               
        rank_scores[style] = weighted_sum / total_w if total_w != 0.0 else 0.0
        rank_scores[f"{style}_preonly"] = weighted_ranks_pre[style]/weights_pre if weights_pre != 0.0 else 0.0
        final_style_set.add(f"{style}_preonly")
    rank_scores["next_sharpness_preonly_raw"] = weighted_sharp/weights_pre if weights_pre != 0.0 else 0.0
    rank_scores["next_depth_preonly_raw"] = weighted_depth/weights_pre if weights_pre != 0.0 else 0.0
    rank_scores["next_auc_preonly_raw"] = weighted_auc/weights_pre if weights_pre != 0.0 else 0.0
    final_style_set.add("next_auc_preonly_raw")
    final_style_set.add("next_sharpness_preonly_raw")
    final_style_set.add("next_depth_preonly_raw")
    return rank_scores, peak, valleys

def compute_spike_features(i, raw_trace, spike_prob_trace, all_left_base_proms, spike_idx_prob, neuron_prom_skew, peak , valleys, edge = 32):
    """
    Compute features for a single spike:
      1. Left-base prominence at spike index in spike_prob_trace.
      2. Value in spike_prob_trace at spike index.
      3. Change in skewness of prominence distribution when this spike is removed.
    """
    

    
    # Find the index of this spike in the peaks array
    """peak_idx = np.where(peaks == spike_idx_prob)[0][0]
    try:
        left_base_prom = spike_prob_trace[peaks[peak_idx]] - spike_prob_trace[left_bases[peak_idx]]
    except IndexError:
        raise IndexError(Possible reasons inclued: 
                         {peak_idx} is out of bounds for peaks array of length {len(peaks)}. 
                         {peak_idx} is out of bounds for left_bases array of length {len(left_bases)}.
                         {peaks} or {left_bases} contain unexpected values.) """
    #Step 0: Find window around the spike
    spike_idx_prob -= edge  # Adjust for earlier edge trimming
    left_window, window, right_window = get_windowed_trace(raw_trace, spike_prob_trace, spike_idx_prob)[0]
    left_window_raw, window_raw, right_window_raw = get_windowed_trace(raw_trace, spike_prob_trace, spike_idx_prob)[1]

    #Step 1: Find derivative and integral features
    max_d2 = find_max_second_derivative(left_window)
    max_d2_raw = find_max_second_derivative(left_window_raw)
    auc = np.trapz(window)

    #Step 2: Value in spike_prob_trace at spike index
    spike_prob_value = spike_prob_trace[spike_idx_prob]

    #Step 3: Change in skewness of prominence distribution if this spike is removed
    if all_left_base_proms.size > 1:
        proms_wo = np.delete(all_left_base_proms, i)
        new_skew = skew(proms_wo) if proms_wo.size > 1 else 0.0
        delta_skew = neuron_prom_skew - new_skew
        
    else:
        delta_skew = 0.0
    
    #Step 4: Compute window features
    (window_skew, window_kurtosis) = (skew(window), kurtosis(window)) if (len(window) > 4 and np.nanvar(window) > 0) else (0.0, 0.0)
    (window_raw_skew, window_raw_kurtosis) = (skew(window_raw), kurtosis(window_raw)) if (len(window_raw) > 4 and np.nanvar(window_raw) > 0) else (0.0, 0.0)
    #STep 5: Compute Rank Scores
    valleys = weight_valley_ranks_for_peak(valleys, peak, sigma = 2)
    rank_scores, _, _ = compute_peak_rank_score_from_weighted(valleys, peak)
    try:
        print("rank_scores keys", rank_scores.keys())
        return {"left_based_prom" : all_left_base_proms[i], 
            "spike_prob_value" : spike_prob_value, 
            "skew_contribution" : delta_skew,
            "auc" : auc,
            "max_second_derivative" : max_d2,
            "max_second_derivative_raw" : max_d2_raw,
            "window_kurtosis" : window_kurtosis,
            "window_skew" : window_skew,
            "window_raw_kurtosis" : window_raw_kurtosis,
            "window_raw_skew" : window_raw_skew, 
            "next_depth_preonly" : rank_scores["next_depth_preonly"], 
            "next_sharpness_preonly" : rank_scores["next_sharpness_preonly"],
            "next_auc_preonly" : rank_scores["next_auc_preonly"],
            "next_depth_preonly_raw" : rank_scores["next_depth_preonly_raw"],
            "next_sharpness_preonly_raw" : rank_scores["next_sharpness_preonly_raw"], 
            "next_auc_preonly_raw" : rank_scores["next_auc_preonly_raw"]}
    except KeyError as e:
        raise KeyError(f"Missing expected rank score key: {e}. Available keys: {list(rank_scores.keys())}")