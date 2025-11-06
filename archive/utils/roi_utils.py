from data_classes.spike import Spike, Valley
from typing import List, Tuple

def compute_valley_depths(valleys : List[Valley]) ->  List[Valley]:
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
    """
    
    for v in valleys:
        # Valley must be normalized
        v_norm = v.normalized_value
        # ----- previous side -----
        if v.previous_peak is not None:
            prev_norm = v.previous_peak.val_prob_normalized
            v.previous_depth = prev_norm - v_norm
            dist_prev = v.index - v.previous_peak.idx_prob
            v.previous_depth_sharpness = (v.previous_depth / dist_prev) if dist_prev > 0 else 0.0
        else:
            v.previous_depth = 0.0
            v.previous_depth_sharpness = 0.0

        # ----- next side -----
        if v.next_peak is not None:
            next_norm = v.next_peak.val_prob_normalized
            v.next_depth = next_norm - v_norm
            dist_next = v.next_peak.idx_prob - v.index
            v.next_depth_sharpness = (v.next_depth / dist_next) if dist_next > 0 else 0.0
        else:
            v.next_depth = 0.0
            v.next_depth_sharpness = 0.0

        # ----- aggregates -----
        v.sum_depth = v.previous_depth + v.next_depth
        v.average_depth = v.sum_depth / 2.0
        v.sum_depth_sharpness = v.previous_depth_sharpness + v.next_depth_sharpness
        v.average_depth_sharpness = v.sum_depth_sharpness / 2.0

    return valleys

def assign_normalized_values(valleys : List[Valley], spikes : List[Spike]) -> Tuple[List[Valley], List[Spike]]:
        """
        Assign normalized_value to each Peak and Valley:

            normalized_value = (value - min_valley_value) / (max_peak_value - min_valley_value)

        Returns the updated lists and also the (max_peak_value, min_valley_value) used.
        """
        if len(spikes) == 0:
            raise ValueError("No peaks provided — cannot compute max_peak_value for normalization.")
        if len(valleys) == 0:
            raise ValueError("No valleys provided — cannot compute min_valley_value for normalization.")

        max_peak_value = max(p.val_prob for p in spikes)
        min_valley_value = min(v.value for v in valleys)
        denominator = (max_peak_value - min_valley_value)

        # Guard against degenerate case
        if denominator == 0:
            # If everything is flat, assign 0.0 to avoid NaNs — adjust if you prefer another convention
            for p in spikes:
                p.val_prob_normalized = 0.0
            for v in valleys:
                v.normalized_value = 0.0

        # Assign normalized values
        for p in spikes:
            p.val_prob_normalized = (p.val_prob - min_valley_value) / denominator
        for v in valleys:
            v.normalized_value = (v.value - min_valley_value) / denominator
        return valleys, spikes

def couple_peaks_to_valleys(valleys : List[Valley], spikes: List[Spike]) -> Tuple[List[Valley], List[Spike]]:
        """
        Couples each Valley to its previous and next Peak, following your rules.

        - Initialize offset=0, start=0
        - If the first valley occurs before the first peak:
            valleys[0].previous_peak = None
            valleys[0].next_peak = peaks[0]
            offset = -1
            start = 1
        - Then iterate i in range(start, len(valleys)-1) and assign:
            valleys[i].previous_peak = peaks[i + offset]
            valleys[i].next_peak = peaks[i + 1 + offset]  (except IndexError -> None)

        Returns the same peaks & valleys lists (in-place updated), just for convenience.
        """
        if not valleys:
            pass
        if not spikes:
            # No peaks: everything stays None on valleys
            for v in valleys:
                v.previous_peak = None
                v.next_peak = None
            

        offset = 0
        start = 0

        # Check whether the trace begins with a minimum (valley before first peak)
        if valleys[0].index < spikes[0].idx_prob:
            valleys[0].previous_peak = None
            valleys[0].next_peak = spikes[0]
            offset = -1
            start = 1

        # Couple previous/next spikes for the remaining valleys
        # (we follow your requested range: start .. len(valleys)-2 inclusive)
        for i in range(start, max(start, len(valleys) - 1)):
            try:
                valleys[i].previous_peak = spikes[i + offset]
            except IndexError:
                valleys[i].previous_peak = None
            try:
                valleys[i].next_peak = spikes[i + 1 + offset]
            except IndexError:
                valleys[i].next_peak = None

        # Optionally handle the last valley (len(valleys)-1) similarly:
        last_i = len(valleys) - 1
        if last_i >= 0:
            try:
                valleys[last_i].previous_peak = spikes[last_i + offset]
            except IndexError:
                valleys[last_i].previous_peak = None
            try:
                valleys[last_i].next_peak = spikes[last_i + 1 + offset]
            except IndexError:
                valleys[last_i].next_peak = None
        return valleys, spikes