
from scipy.signal import find_peaks
import numpy as np
from typing import Optional, List, Tuple
from data_structures import Peak, Valley


def find_peaks_and_valleys(
    smoothed_trace: np.ndarray,
    peak_kwargs: Optional[dict] = None,
    valley_kwargs: Optional[dict] = None,
) -> Tuple[List[Peak], List[Valley]]:
    """
    Args:
        smoothed_trace (np.ndarray): Smoothed trace
        peak_kwargs (dict): Keyword arguments passed to peak class
        valley_kwargs (dict): Keyword arguments passed to valley class
    """
    x = np.asarray(smoothed_trace, dtype=float)
    if x.ndim != 1:
        raise ValueError("smoothed_trace must be 1D")

    peak_kwargs = peak_kwargs or {}
    valley_kwargs = valley_kwargs or {}

    # Maxima

    peak_idx : np.ndarray  = find_peaks(x, **peak_kwargs)[0]
    peaks = [Peak(index=int(idx), value=float(x[idx])) for idx in peak_idx]

    # Minima: peaks of the inverted signal
    valley_idx : np.ndarray = find_peaks(-x, **valley_kwargs)[0]
    valleys = [Valley(index=int(i), value=float(x[i])) for i in valley_idx]

    return peaks, valleys


# ------------------------
# Step 2: normalize values across peaks & valleys using your global min/max rule
# ------------------------
def assign_normalized_values(
    peaks: List[Peak],
    valleys: List[Valley],
) -> Tuple[List[Peak], List[Valley], float, float]:
    """
    Assign normalized_value to each Peak and Valley:

        normalized_value = (value - min_valley_value) / (max_peak_value - min_valley_value)

    Returns the updated lists and also the (max_peak_value, min_valley_value) used.
    """
    if len(peaks) == 0:
        raise ValueError("No peaks provided — cannot compute max_peak_value for normalization.")
    if len(valleys) == 0:
        raise ValueError("No valleys provided — cannot compute min_valley_value for normalization.")

    max_peak_value = max(p.value for p in peaks)
    min_valley_value = min(v.value for v in valleys)
    denominator = (max_peak_value - min_valley_value)

    # Guard against degenerate case
    if denominator == 0:
        # If everything is flat, assign 0.0 to avoid NaNs — adjust if you prefer another convention
        for p in peaks:
            p.normalized_value = 0.0
        for v in valleys:
            v.normalized_value = 0.0
        return peaks, valleys, max_peak_value, min_valley_value

    # Assign normalized values
    for p in peaks:
        p.normalized_value = (p.value - min_valley_value) / denominator
    for v in valleys:
        v.normalized_value = (v.value - min_valley_value) / denominator

    return peaks, valleys, max_peak_value, min_valley_value

def couple_peaks_to_valleys(peaks: List[Peak], valleys: List[Valley]) -> Tuple[List[Peak], List[Valley]]:
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
        return peaks, valleys
    if not peaks:
        # No peaks: everything stays None on valleys
        for v in valleys:
            v.previous_peak = None
            v.next_peak = None
        return peaks, valleys

    offset = 0
    start = 0

    # Check whether the trace begins with a minimum (valley before first peak)
    if valleys[0].index < peaks[0].index:
        valleys[0].previous_peak = None
        valleys[0].next_peak = peaks[0]
        offset = -1
        start = 1

    # Couple previous/next peaks for the remaining valleys
    # (we follow your requested range: start .. len(valleys)-2 inclusive)
    for i in range(start, max(start, len(valleys) - 1)):
        try:
            valleys[i].previous_peak = peaks[i + offset]
        except IndexError:
            valleys[i].previous_peak = None
        try:
            valleys[i].next_peak = peaks[i + 1 + offset]
        except IndexError:
            valleys[i].next_peak = None

    # Optionally handle the last valley (len(valleys)-1) similarly:
    last_i = len(valleys) - 1
    if last_i >= 0:
        try:
            valleys[last_i].previous_peak = peaks[last_i + offset]
        except IndexError:
            valleys[last_i].previous_peak = None
        try:
            valleys[last_i].next_peak = peaks[last_i + 1 + offset]
        except IndexError:
            valleys[last_i].next_peak = None

    return peaks, valleys