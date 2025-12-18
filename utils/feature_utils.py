import numpy as np
from typing import Tuple, Optional
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew

def _compute_min_between(
    trace: np.ndarray, start: int, end: int
) -> int:
    """Return index of minimum value between start and end (exclusive of end)."""
    if start >= end:
        return start
    local_min = int(np.argmin(trace[start:end]))
    return start + local_min

def _create_large_window(
    valid_spike_prob: np.ndarray,
    peak_idx: int,
    left_base_idx: int,
    right_base_idx: int,
    start_idx: int = 0
) -> Tuple[np.ndarray, int, int, float]:
    """
    Create the large window (prominence-based) around a spike peak.
    
    Args:
        valid_spike_prob: Spike probability trace (valid region only)
        peak_idx: Peak index in valid region coordinates
        left_base_idx: Left base index in valid region coordinates
        right_base_idx: Right base index in valid region coordinates
        start_idx: Starting index of valid region in original array
    
    Returns:
        Tuple of (large_window, absolute_left_base, absolute_right_base, spike_prominence)
    """
    # Extract window in valid region coordinates
    large_window = valid_spike_prob[left_base_idx:right_base_idx]
    
    # Convert to absolute coordinates
    absolute_left_base = int(left_base_idx + start_idx)
    absolute_right_base = int(right_base_idx + start_idx)
    absolute_peak = int(peak_idx + start_idx)
    
    # Calculate spike prominence
    spike_prom = valid_spike_prob[peak_idx] - valid_spike_prob[left_base_idx]
    
    return large_window, absolute_left_base, absolute_right_base, float(spike_prom)


def _create_small_window(
    valid_spike_prob: np.ndarray,
    peak_idx: int,
    prev_peak_idx: int,
    next_peak_idx: int,
    start_idx: int = 0
) -> Tuple[np.ndarray, int, int]:
    """
    Create the small window (inter-peak distance) around a spike peak.
    
    Args:
        valid_spike_prob: Spike probability trace (valid region only)
        peak_idx: Current peak index in valid region coordinates
        prev_peak_idx: Previous peak index (or 0 if first peak)
        next_peak_idx: Next peak index (or len(trace) if last peak)
        start_idx: Starting index of valid region in original array
    
    Returns:
        Tuple of (small_window, absolute_prev_min, absolute_next_min)
    """
    # Find local minima between peaks
    prev_min = _compute_min_between(valid_spike_prob, prev_peak_idx, peak_idx)
    next_min = _compute_min_between(valid_spike_prob, peak_idx, next_peak_idx)
    

    # Ensure the small window is non-empty and ordered
    if next_min <= prev_min:
        next_min = prev_min + 1 if prev_min + 1 < len(valid_spike_prob) else len(valid_spike_prob)
    
    # Extract window in valid region coordinates
    small_window = valid_spike_prob[prev_min:next_min]
    
    # Convert to absolute coordinates
    absolute_prev_min = int(prev_min + start_idx)
    absolute_next_min = int(next_min + start_idx)
    
    return small_window, absolute_prev_min, absolute_next_min
def compute_spike_constants(
    window: np.ndarray,
    peak_idx_in_window: int,
    fs: float = 30.0,
    rise_fraction: float = 0.1,
    decay_fraction: float = 0.9,
    ) -> Tuple[float, float]:
    """
    Estimate rise slope and decay time constant for a single spike transient.
    
    The implementation avoids curve-fitting instability by:
    (1) normalizing the transient using the local baseline and peak amplitude,
    (2) fitting a simple linear model to the rising segment, and
    (3) computing the time to decay to 1/e of the peak using linear interpolation.
    
    Args:
        window: Spike probability window containing the spike transient
        peak_idx_in_window: Index of the peak within the window (relative to window start)
        fs: Sampling frequency in Hz (default: 30.0)
        rise_fraction: Fraction of peak amplitude to start rise fitting (default: 0.1)
        decay_fraction: Fraction of peak amplitude to start decay fitting (default: 0.9)
    
    Returns:
        Tuple of (rise_slope, decay_tau):
            - rise_slope: Linear slope of the rising phase (normalized units/second)
            - decay_tau: Decay time constant in seconds (time to reach 1/e of peak)
    """
    segment = np.asarray(window, dtype=float)
    
    # Validate input
    if segment.size < 3 or not np.isfinite(segment).all():
        return np.nan, np.nan
    
    # Normalize the segment
    baseline = float(np.nanmin(segment))
    peak_value = float(np.nanmax(segment))
    amplitude = peak_value - baseline
    
    if not np.isfinite(amplitude) or amplitude <= 1e-8:
        return np.nan, np.nan
    
    normed = (segment - baseline) / amplitude
    peak_rel = int(peak_idx_in_window)
    peak_rel = max(0, min(peak_rel, normed.size - 1))
    
    # ===== Rising slope via linear regression =====
    rise_segment = normed[:peak_rel + 1]
    if rise_segment.size < 2:
        rise_slope = np.nan
    else:
        t_rise = np.arange(rise_segment.size, dtype=float) / float(fs)
        try:
            slope, _ = np.polyfit(t_rise, rise_segment, 1)
        except Exception:
            slope = np.nan
        rise_slope = slope
    
    # ===== Decay constant estimated from time to reach 1/e of peak =====
    decay_segment = normed[peak_rel:]
    if decay_segment.size < 2:
        decay_tau = np.nan
    else:
        t_decay = np.arange(decay_segment.size, dtype=float) / float(fs)
        target = np.exp(-1.0)  # 1/e ≈ 0.368
        below = np.where(decay_segment <= target)[0]
        
        if below.size == 0:
            # Extrapolate using exponential fit if target not reached
            positive = decay_segment > 1e-6
            if np.count_nonzero(positive) >= 2:
                t_fit = t_decay[positive]
                y_fit = np.log(decay_segment[positive])
                try:
                    slope, _ = np.polyfit(t_fit, y_fit, 1)
                except Exception:
                    slope = np.nan
                
                if np.isfinite(slope) and slope < 0:
                    decay_tau = -1.0 / slope
                else:
                    decay_tau = np.nan
            else:
                decay_tau = np.nan
        else:
            # Interpolate to find exact crossing point
            idx = int(below[0])
            if idx == 0:
                decay_tau = 0.0
            else:
                y0 = decay_segment[idx - 1]
                y1 = decay_segment[idx]
                x0 = t_decay[idx - 1]
                x1 = t_decay[idx]
                
                if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
                    decay_tau = t_decay[idx]
                else:
                    frac = (target - y0) / (y1 - y0)
                    frac = np.clip(frac, 0.0, 1.0)
                    decay_tau = x0 + frac * (x1 - x0)
        
        # Fallback if decay_tau is still invalid
        if not np.isfinite(decay_tau):
            if decay_segment.size >= 2:
                decay_tau = t_decay[-1]
            else:
                decay_tau = 0.0
    
    # Final validation
    if not np.isfinite(rise_slope):
        rise_slope = np.nan
    
    return rise_slope, decay_tau
