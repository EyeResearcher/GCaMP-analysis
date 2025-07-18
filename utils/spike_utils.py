import sys


import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import curve_fit

def find_spikes(spike_prob, raw_fluorescence, smooth = True, sigma=4, window_size=10, edge = 32):
    """
    Find spikes using the spike probability trace.
    Returns indices and values of peaks in both smoothed spike probability and raw fluorescence.

    Returns:
        tuple: (smoothed_peak_indices, 
                values_at_smoothed_peaks_in_raw_prob,
                refined_peak_indices, 
                refined_peak_values)
    """
    # Step 0: Trim the spike probability trace to avoid edge effects 
    spike_prob_trimmed = spike_prob[edge:-edge]
    # Step 1: Smooth the spike probability signal and find peaks
    prob_input = gaussian_filter1d(spike_prob_trimmed, sigma=sigma) if smooth == True else spike_prob_trimmed
    prob_peak_indices, _ = find_peaks(prob_input)
    _ , left_bases, _ = peak_prominences(prob_input, prob_peak_indices)
    left_base_prominences = prob_input[prob_peak_indices] - prob_input[left_bases]

    # Step 2: Get spike probability values at smoothed peak indices frpm trimmed spike probability trace
    prob_peak_indices += int(edge)  # Adjust indices back to original trace
    prob_peak_values = spike_prob[prob_peak_indices] 
    

    # Step 3: Refine peak indices in the raw fluorescence trace
    fluorescence_peak_indices = []
    fluorescence_peak_values = []

    n_frames = len(raw_fluorescence)

    
    for peak_idx in prob_peak_indices:
        
        #Create a window around the smoothed peak index
        start = max(0, peak_idx - window_size/2)
        end = min(n_frames, peak_idx + window_size*3/2 + 1)
        window = raw_fluorescence[int(start):int(end)]

        # Find the local maximum in the window 
        local_max_idx = np.argmax(window)
        fluorescence_peak_index = start + local_max_idx
        fluorescence_peak_value = window[local_max_idx]

        # Add the refined peak index and value to the lists
        fluorescence_peak_indices.append(fluorescence_peak_index)
        fluorescence_peak_values.append(fluorescence_peak_value)

    return (
        np.array(prob_peak_indices),
        np.array(prob_peak_values),
        left_base_prominences,
        np.array(fluorescence_peak_indices),
        np.array(fluorescence_peak_values)
    )

# --- Windowing ---
def window_spike_transients(fluorescence, spike_indices):
    """
    For each spike peak index, find the window defined by local minima:
      - start: minimum between previous peak and current peak
      - peak: current peak index
      - end: minimum between current peak and next peak
    Returns a list of (start_idx, peak_idx, end_idx) tuples.
    """
    windows = []
    n = len(fluorescence)
    for i, peak in enumerate(spike_indices):
        prev_peak = spike_indices[i-1] if i > 0 else 0
        next_peak = spike_indices[i+1] if i < len(spike_indices)-1 else n - 1

        if peak > prev_peak:
            seg_b = fluorescence[prev_peak:peak]
            start = prev_peak + int(np.argmin(seg_b)) if seg_b.size else peak
        else:
            start = peak

        if next_peak > peak:
            seg_a = fluorescence[peak:next_peak]
            end = peak + int(np.argmin(seg_a)) if seg_a.size else peak
        else:
            end = peak

        windows.append((start, peak, end))
    return windows

# --- Models ---
def exponential_decay(t, A, tau, offset):
    """Exponential decay: A * exp(-t/tau) + offset"""
    return A * np.exp(-t / tau) + offset

def linear_rise(t, m, b):
    """Linear rise: m * t + b"""
    return m * t + b

# --- Fitting functions ---
def fit_rise_constants(normed_segments, fs=30, rise_fraction=0.1):
    """
    Fit linear rise on normalized segments.
    Returns list of (m, b) tuples.
    """
    params = []
    for seg in normed_segments:
        peak_norm = seg.max()
        thresh = rise_fraction * peak_norm
        idxs = np.where(seg >= thresh)[0]
        seg2 = seg[idxs[0]:] if idxs.size else seg
        t = np.arange(len(seg2)) / fs

        m0 = (seg2[-1] - seg2[0]) / (t[-1] if t[-1] != 0 else 1)
        b0 = float(seg2[0])
        try:
            popt, _ = curve_fit(linear_rise, t, seg2, p0=[m0, b0], maxfev=10000)
        except Exception:
            popt = (np.nan, np.nan)
        params.append(tuple(popt))
    return params


def fit_decay_constants(normed_segments, fs=30, decay_fraction=0.9):
    """
    Fit exponential decay on normalized segments.
    Returns list of (A, tau, offset) tuples.
    """
    params = []
    for seg in normed_segments:
        peak_norm = seg[0]
        end_norm = seg[-1]
        thresh = peak_norm - decay_fraction * (peak_norm - end_norm)
        idxs = np.where(seg <= thresh)[0]
        seg2 = seg[:idxs[0] + 1] if idxs.size else seg
        t = np.arange(len(seg2)) / fs

        A0 = float(seg2[0] - seg2[-1])
        tau0 = (len(seg2) / fs) / 2
        off0 = float(seg2[-1])
        try:
            popt, _ = curve_fit(exponential_decay, t, seg2, p0=[A0, tau0, off0], maxfev=10000)
        except Exception:
            popt = (np.nan, np.nan, np.nan)
        params.append(tuple(popt))
    return params


def compute_spike_constants(fluo, peak_idx, fs=30,
                             rise_fraction=0.1, decay_fraction=0.9):
    """
    Compute rise slope (m) and decay time constant (tau) for a single spike:
      1. Window around peak via local minima.
      2. Extract and normalize ΔF/F for that window.
      3. Fit linear rise and exponential decay.
    Returns:
        m: slope of rise fit
        tau: decay time constant
    """
    # Windowing
    start, peak, end = window_spike_transients(fluo, [peak_idx])[0]
    segment = fluo[start:end+1]
    # Normalize
    F0 = np.min(segment)
    normed = (segment - F0) / F0
    # Fit
    m, _ = fit_rise_constants([normed], fs, rise_fraction)[0]
    _, tau, _ = fit_decay_constants([normed], fs, decay_fraction)[0]
    return m, tau

# --- Area under curve ---
def compute_area_under_curve(fluo, fs=30, baseline=None):
    """
    Compute total area under the entire fluorescence trace minus baseline.

    Args:
        fluo (1D array): raw fluorescence trace.
        fs (float): sampling frequency (Hz).
        baseline (float or None): baseline to subtract. If None, use global min.

    Returns:
        total_area (float): AUC of full trace minus baseline.
    """
    if baseline is None:
        baseline = np.min(fluo)
    total_area = np.trapz(fluo - baseline, dx=1.0/fs)
    return total_area