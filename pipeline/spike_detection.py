"""Spike detection from Cascade probabilities."""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import logging

if TYPE_CHECKING:
    from data_classes import Spike

logger = logging.getLogger(__name__)

def detect_spikes_from_cascade(f_trace: np.ndarray,
                              cascade_prob: np.ndarray,
                              prob_sigma: float = 2.0,
                              window_size: int = 5,
                              min_prominence: float = 0.05,
                              min_distance: int = 8,
                              edge_trim: int = 32,
                              **kwargs) -> List[Spike]:
    """
    Detect spikes from Cascade probability and match to fluorescence peaks.
    
    Parameters:
        f_trace: Fluorescence trace
        cascade_prob: Cascade spike probability
        prob_sigma: Gaussian smoothing for probability
        window_size: Window for finding F peak (+/- frames)
        min_prominence: Minimum prominence for cascade peaks
        min_distance: Minimum distance between spikes (frames)
        edge_trim: Ignore spikes near edges
        
    Returns:
        List of Spike objects
    """
    from data_classes import Spike
    
    # Smooth cascade probability
    if prob_sigma > 0:
        smoothed_prob = gaussian_filter1d(cascade_prob, sigma=prob_sigma)
    else:
        smoothed_prob = cascade_prob
    
    # Find peaks in cascade probability
    cascade_peaks, properties = find_peaks(smoothed_prob,
                                          prominence=min_prominence,
                                          distance=min_distance)
    
    # Filter edge peaks
    cascade_peaks = cascade_peaks[(cascade_peaks >= edge_trim) & 
                                  (cascade_peaks < len(cascade_prob) - edge_trim)]
    
    spikes = []
    for cascade_idx in cascade_peaks:
        # Find corresponding fluorescence peak
        start_idx = max(0, cascade_idx - window_size)
        end_idx = min(len(f_trace), cascade_idx + window_size + 1)
        
        window = f_trace[start_idx:end_idx]
        if len(window) > 0:
            local_peak = np.argmax(window)
            f_peak_idx = start_idx + local_peak
            
            # Create spike object
            spike = Spike(
                frame_index=f_peak_idx,
                cascade_peak_idx=cascade_idx,
                prob_height=smoothed_prob[cascade_idx],
                f_value=f_trace[f_peak_idx]
            )
            spikes.append(spike)
    
    return spikes

def find_spike_peaks(trace: np.ndarray,
                    threshold: float = None,
                    prominence: float = 0.1,
                    distance: int = 5) -> Tuple[np.ndarray, Dict]:
    """
    Find peaks in a trace.
    
    Parameters:
        trace: Input trace
        threshold: Minimum height threshold
        prominence: Minimum prominence
        distance: Minimum distance between peaks
        
    Returns:
        Peak indices and properties
    """
    # Compute threshold if not provided
    if threshold is None:
        threshold = np.mean(trace) + 2 * np.std(trace)
    
    # Find peaks
    peaks, properties = find_peaks(trace,
                                  height=threshold,
                                  prominence=prominence,
                                  distance=distance)
    
    return peaks, properties

def create_spike_windows(spikes: List[Spike],
                        f_trace: np.ndarray,
                        window_before: int = 10,
                        window_after: int = 20) -> np.ndarray:
    """
    Extract windows around spikes for visualization or analysis.
    
    Parameters:
        spikes: List of Spike objects
        f_trace: Fluorescence trace
        window_before: Frames before spike
        window_after: Frames after spike
        
    Returns:
        Array of spike windows (n_spikes x window_size)
    """
    if len(spikes) == 0:
        return np.array([])
    
    window_size = window_before + window_after + 1
    windows = []
    
    for spike in spikes:
        start = spike.frame_index - window_before
        end = spike.frame_index + window_after + 1
        
        if start >= 0 and end <= len(f_trace):
            window = f_trace[start:end]
            windows.append(window)
    
    if windows:
        return np.array(windows)
    return np.array([])

def compute_spike_statistics(spikes: List[Spike], 
                            total_frames: int,
                            fs: float = 30.0) -> Dict:
    """
    Compute statistics for detected spikes.
    
    Parameters:
        spikes: List of Spike objects
        total_frames: Total number of frames
        fs: Sampling frequency
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    if len(spikes) == 0:
        return {
            'n_spikes': 0,
            'spike_rate': 0.0,
            'mean_isi': np.nan,
            'cv_isi': np.nan,
            'burst_index': 0.0
        }
    
    # Basic stats
    stats['n_spikes'] = len(spikes)
    stats['spike_rate'] = len(spikes) / (total_frames / fs)
    
    # Inter-spike intervals
    if len(spikes) > 1:
        spike_times = np.array([s.frame_index for s in spikes])
        isis = np.diff(spike_times) / fs
        stats['mean_isi'] = np.mean(isis)
        stats['cv_isi'] = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
        
        # Burst index (fraction of ISIs < 100ms)
        stats['burst_index'] = np.mean(isis < 0.1)
    else:
        stats['mean_isi'] = np.nan
        stats['cv_isi'] = np.nan
        stats['burst_index'] = 0.0
    
    return stats