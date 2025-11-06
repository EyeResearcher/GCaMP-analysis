import math
import numpy as np

def sum_preceding_valley_contributions(peak, valleys, styles):
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
    summed_weighted_valleys = dict(zip(styles, np.zeros(len(styles))))
    for valley in valleys:
        if valley.index < peak.index:
            # Add valley's weight
            total_weight += valley.weight

            # Add weighted rank contributions per style

            if hasattr(valley, "weighted_ranks"):
                for style, value in valley.weighted_ranks.items():
                   summed_weighted_ranks[style] += value
                   summed_weighted_valleys[style] += valley.weighted_depths[style]
            
    return total_weight, summed_weighted_ranks, summed_weighted_valleys
# ---------- 1) Per-peak weighting of valley ranks ----------
def weight_valley_ranks_for_peak(valleys, peak, sigma):
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

    p_idx = int(peak.index)

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
            v.weighted_depths[style] = float(v.depths[style]) * w

    return valleys


# ---------- 2) Aggregate to a rank score for that peak ----------
def compute_peak_rank_score_from_weighted(valleys, peak):
    """
    Given valleys already weighted for *this* peak (via weight_valley_ranks_for_peak),
    compute per-style weighted average of ranks:

        peak.rank_score[style] = sum_v (v.weighted_ranks[style]) / sum_v (v.weight)

    Returns:
      valleys (unchanged), peak (with rank_score updated)
    """

    # Collect styles from any valley that has weighted_ranks
    styles = set()
    for v in valleys:
        if getattr(v, "weighted_ranks", None):
            styles.update(v.weighted_ranks.keys())
    weights_pre, weighted_ranks_pre, summed_weighted_valleys_pre = sum_preceding_valley_contributions(peak, valleys, styles)
    print("weights_pre", weights_pre)
    print("weighted_ranks_pre", weighted_ranks_pre)
    # Sum of weights across valleys
    total_w = sum(getattr(v, "weight", 0.0) for v in valleys)

    if getattr(peak, "rank_score", None) is None:
        peak.rank_score = {}

    if total_w == 0.0:
        # No weight support → assign None or 0.0 per your preference
        for style in styles:
            peak.rank_score[style] = None
        return valleys, peak
    final_style_set = set(styles)
    for style in styles:
        weighted_sum = 0.0
        weighted_valley_sum = 0.0
        for v in valleys:
            if getattr(v, "weighted_ranks", None) and style in v.weighted_ranks:
                weighted_sum += v.weighted_ranks[style]
                weighted_valley_sum += v.weighted_depths[style]
        peak.rank_score[style] = weighted_sum / total_w
        peak.rank_score[f"{style}_raw"] = summed_weighted_valleys_pre[style]/weighted_valley_sum if weighted_valley_sum>0 else 0
        peak.rank_score[f"{style}_preonly"] = weighted_ranks_pre[style]/weights_pre if weights_pre>0 else 0
        peak.rank_score[f"{style}_preonly_raw"] = summed_weighted_valleys_pre[style]/weights_pre if weights_pre>0 else 0
        final_style_set.add(f"{style}_preonly")
        final_style_set.add(f"{style}_raw")
        final_style_set.add(f"{style}_preonly_raw")
    return valleys, peak, final_style_set


# ---------- Convenience: do it for all peaks in one go ----------
def compute_all_peak_rank_scores(valleys, peaks, sigma):
    """
    For each peak:
      - compute weights & weighted_ranks on valleys
      - compute the peak's rank_score dict (per sorting style)

    Returns:
      valleys (last-updated weights reflect the last peak processed),
      peaks   (each with .rank_score filled)
    """
    styles_set = set()
    for pk in peaks:
        weight_valley_ranks_for_peak(valleys, pk, sigma)
        _, _, styles = compute_peak_rank_score_from_weighted(valleys, pk)
        for style in styles:
            styles_set.update(styles)
    return valleys, peaks, styles_set