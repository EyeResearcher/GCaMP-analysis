"""
Temporal analysis functions for GCaMP data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from scipy import signal
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from data_classes import Video, Neuron

def analyze_temporal_patterns(
    video: Video,
    bin_size: float = 1.0
) -> pd.DataFrame:
    """
    Analyze temporal patterns of neural activity.
    
    Parameters
    ----------
    video : Video
        Video to analyze
    bin_size : float
        Bin size in seconds for temporal binning
        
    Returns
    -------
    pd.DataFrame
        Time-binned activity statistics
    """
    frame_rate = video.frame_rate
    n_frames = video.F.shape[1]
    bin_frames = int(bin_size * frame_rate)
    n_bins = n_frames // bin_frames
    
    results = []
    
    for bin_idx in range(n_bins):
        start_frame = bin_idx * bin_frames
        end_frame = (bin_idx + 1) * bin_frames
        
        # Compute activity metrics for this bin
        bin_spike_counts = []
        bin_active_neurons = 0
        
        for neuron in video.neurons:
            spike_count = np.sum(neuron.binary_spike_train[start_frame:end_frame])
            bin_spike_counts.append(spike_count)
            if spike_count > 0:
                bin_active_neurons += 1
        
        # Compute fluorescence statistics
        bin_fluor = video.F[:, start_frame:end_frame]
        
        results.append({
            'bin_idx': bin_idx,
            'start_time': start_frame / frame_rate,
            'end_time': end_frame / frame_rate,
            'total_spikes': np.sum(bin_spike_counts),
            'mean_spikes_per_neuron': np.mean(bin_spike_counts),
            'active_neurons': bin_active_neurons,
            'fraction_active': bin_active_neurons / len(video.neurons) if video.neurons else 0,
            'mean_fluorescence': np.mean(bin_fluor),
            'std_fluorescence': np.std(bin_fluor)
        })
    
    return pd.DataFrame(results)


def compute_burst_statistics(
    neurons: List[Neuron],
    min_spikes: int = 3,
    max_isi: float = 0.5,
    min_burst_duration: float = 0.1
) -> Dict[str, float]:
    """
    Compute burst statistics for a list of neurons.
    
    Parameters
    ----------
    neurons : List[Neuron]
        Neurons to analyze
    min_spikes : int
        Minimum number of spikes to constitute a burst
    max_isi : float
        Maximum inter-spike interval (seconds) within a burst
    min_burst_duration : float
        Minimum burst duration in seconds
        
    Returns
    -------
    Dict[str, float]
        Dictionary of burst statistics
    """
    if not neurons:
        return {
            'n_bursts': 0,
            'burst_frequency': 0.0,
            'mean_burst_duration': 0.0,
            'mean_spikes_per_burst': 0.0,
            'burst_rate': 0.0
        }
    
    # Assume all neurons from same video with same frame rate
    frame_rate = neurons[0].video.frame_rate if neurons[0].video else 30.0
    max_isi_frames = int(max_isi * frame_rate)
    min_burst_frames = int(min_burst_duration * frame_rate)
    
    all_bursts = []
    
    for neuron in neurons:
        # Find spike indices
        spike_indices = np.where(neuron.binary_spike_train)[0]
        
        if len(spike_indices) < min_spikes:
            continue
        
        # Detect bursts
        bursts = []
        current_burst = [spike_indices[0]]
        
        for i in range(1, len(spike_indices)):
            isi = spike_indices[i] - spike_indices[i-1]
            
            if isi <= max_isi_frames:
                current_burst.append(spike_indices[i])
            else:
                # End current burst if criteria met
                if len(current_burst) >= min_spikes:
                    burst_duration = current_burst[-1] - current_burst[0]
                    if burst_duration >= min_burst_frames:
                        bursts.append({
                            'start': current_burst[0],
                            'end': current_burst[-1],
                            'n_spikes': len(current_burst),
                            'duration': burst_duration / frame_rate
                        })
                
                # Start new burst
                current_burst = [spike_indices[i]]
        
        # Check final burst
        if len(current_burst) >= min_spikes:
            burst_duration = current_burst[-1] - current_burst[0]
            if burst_duration >= min_burst_frames:
                bursts.append({
                    'start': current_burst[0],
                    'end': current_burst[-1],
                    'n_spikes': len(current_burst),
                    'duration': burst_duration / frame_rate
                })
        
        all_bursts.extend(bursts)
    
    # Compute statistics
    if all_bursts:
        n_frames = len(neurons[0].binary_spike_train)
        recording_duration = n_frames / frame_rate
        
        return {
            'n_bursts': len(all_bursts),
            'burst_frequency': len(all_bursts) / recording_duration,
            'mean_burst_duration': np.mean([b['duration'] for b in all_bursts]),
            'std_burst_duration': np.std([b['duration'] for b in all_bursts]),
            'mean_spikes_per_burst': np.mean([b['n_spikes'] for b in all_bursts]),
            'std_spikes_per_burst': np.std([b['n_spikes'] for b in all_bursts]),
            'burst_rate': np.sum([b['n_spikes'] for b in all_bursts]) / recording_duration
        }
    else:
        return {
            'n_bursts': 0,
            'burst_frequency': 0.0,
            'mean_burst_duration': 0.0,
            'std_burst_duration': 0.0,
            'mean_spikes_per_burst': 0.0,
            'std_spikes_per_burst': 0.0,
            'burst_rate': 0.0
        }


def analyze_synchrony_over_time(
    video: Video,
    window_size: int = 100,
    step_size: int = 50
) -> pd.DataFrame:
    """
    Analyze network synchrony over time using sliding windows.
    
    Parameters
    ----------
    video : Video
        Video to analyze
    window_size : int
        Window size in frames
    step_size : int
        Step size in frames
        
    Returns
    -------
    pd.DataFrame
        Time series of synchrony metrics
    """
    spike_raster = video.get_spike_raster()
    n_neurons, n_frames = spike_raster.shape
    
    if n_neurons < 2:
        return pd.DataFrame()
    
    n_windows = (n_frames - window_size) // step_size + 1
    results = []
    
    for win_idx in range(n_windows):
        start = win_idx * step_size
        end = start + window_size
        
        window_spikes = spike_raster[:, start:end]
        
        # Compute synchrony metrics
        # 1. Population spike rate
        pop_spike_rate = np.sum(window_spikes) / (window_size * n_neurons) * video.frame_rate
        
        # 2. Pairwise correlation
        correlations = []
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                if np.sum(window_spikes[i]) > 0 and np.sum(window_spikes[j]) > 0:
                    corr = np.corrcoef(window_spikes[i].astype(float), 
                                      window_spikes[j].astype(float))[0, 1]
                    correlations.append(corr)
        
        mean_corr = np.mean(correlations) if correlations else 0
        
        # 3. Synchronous events (frames where >X% of neurons spike)
        active_fraction = np.sum(window_spikes, axis=0) / n_neurons
        sync_events = np.sum(active_fraction > 0.2)  # Threshold at 20% of neurons
        
        # 4. Coefficient of variation of ISIs
        isis = []
        for i in range(n_neurons):
            spike_times = np.where(window_spikes[i])[0]
            if len(spike_times) > 1:
                neuron_isis = np.diff(spike_times)
                isis.extend(neuron_isis)
        
        cv_isi = np.std(isis) / (np.mean(isis) + 1e-10) if isis else 0
        
        results.append({
            'window_idx': win_idx,
            'start_frame': start,
            'end_frame': end,
            'start_time': start / video.frame_rate,
            'end_time': end / video.frame_rate,
            'pop_spike_rate': pop_spike_rate,
            'mean_correlation': mean_corr,
            'n_sync_events': sync_events,
            'sync_event_rate': sync_events / window_size * video.frame_rate,
            'cv_isi': cv_isi,
            'active_neurons': np.sum(np.sum(window_spikes, axis=1) > 0)
        })
    
    return pd.DataFrame(results)


def detect_network_events(
    video: Video,
    min_participating_fraction: float = 0.2,
    event_window: float = 0.5,
    min_event_duration: float = 0.1
) -> List[Dict]:
    """
    Detect network-wide synchronous events.
    
    Parameters
    ----------
    video : Video
        Video to analyze
    min_participating_fraction : float
        Minimum fraction of neurons that must be active
    event_window : float
        Time window (seconds) to consider as single event
    min_event_duration : float
        Minimum event duration in seconds
        
    Returns
    -------
    List[Dict]
        List of detected events with metadata
    """
    spike_raster = video.get_spike_raster()
    n_neurons, n_frames = spike_raster.shape
    frame_rate = video.frame_rate
    
    event_window_frames = int(event_window * frame_rate)
    min_event_frames = int(min_event_duration * frame_rate)
    min_neurons = int(min_participating_fraction * n_neurons)
    
    # Compute population activity
    pop_activity = np.sum(spike_raster, axis=0)
    
    # Smooth population activity
    pop_activity_smooth = gaussian_filter1d(pop_activity.astype(float), sigma=2)
    
    # Find peaks in population activity
    peaks, properties = signal.find_peaks(
        pop_activity_smooth,
        height=min_neurons,
        distance=event_window_frames,
        width=min_event_frames
    )
    
    # Extract event details
    events = []
    for i, peak in enumerate(peaks):
        # Get event window
        start = max(0, peak - event_window_frames // 2)
        end = min(n_frames, peak + event_window_frames // 2)
        
        # Count participating neurons
        event_spikes = spike_raster[:, start:end]
        participating_neurons = np.where(np.sum(event_spikes, axis=1) > 0)[0]
        
        # Compute event properties
        event_strength = np.sum(event_spikes) / len(participating_neurons) if len(participating_neurons) > 0 else 0
        
        events.append({
            'event_idx': i,
            'peak_frame': peak,
            'peak_time': peak / frame_rate,
            'start_frame': start,
            'end_frame': end,
            'duration': (end - start) / frame_rate,
            'n_participating_neurons': len(participating_neurons),
            'participating_fraction': len(participating_neurons) / n_neurons,
            'peak_height': properties['peak_heights'][i],
            'event_strength': event_strength,
            'participating_neuron_ids': participating_neurons.tolist()
        })
    
    return events


def compute_population_dynamics(
    video: Video,
    method: str = 'pca',
    n_components: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute low-dimensional representation of population dynamics.
    
    Parameters
    ----------
    video : Video
        Video to analyze
    method : str
        Dimensionality reduction method ('pca', 'ica')
    n_components : int
        Number of components to extract
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (transformed data, components)
    """
    # Get fluorescence traces (neurons x frames)
    traces = video.get_fluorescence_traces()
    
    if method == 'pca':
        from sklearn.decomposition import PCA
        model = PCA(n_components=n_components)
    elif method == 'ica':
        from sklearn.decomposition import FastICA
        model = FastICA(n_components=n_components)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit and transform (transpose to frames x neurons for sklearn)
    transformed = model.fit_transform(traces.T)
    components = model.components_
    
    return transformed, components


def compute_firing_rate_modulation(
    neuron: Neuron,
    bin_size: float = 1.0,
    smooth_sigma: float = 2.0
) -> np.ndarray:
    """
    Compute smoothed firing rate over time.
    
    Parameters
    ----------
    neuron : Neuron
        Neuron to analyze
    bin_size : float
        Bin size in seconds
    smooth_sigma : float
        Gaussian smoothing sigma in bins
        
    Returns
    -------
    np.ndarray
        Smoothed firing rate over time
    """
    frame_rate = neuron.video.frame_rate if neuron.video else 30.0
    spike_train = neuron.binary_spike_train
    
    bin_frames = int(bin_size * frame_rate)
    n_bins = len(spike_train) // bin_frames
    
    # Bin spikes
    binned_spikes = []
    for i in range(n_bins):
        start = i * bin_frames
        end = (i + 1) * bin_frames
        spike_count = np.sum(spike_train[start:end])
        binned_spikes.append(spike_count / bin_size)  # Convert to Hz
    
    binned_spikes = np.array(binned_spikes)
    
    # Smooth
    if smooth_sigma > 0:
        smoothed = gaussian_filter1d(binned_spikes, sigma=smooth_sigma)
    else:
        smoothed = binned_spikes
    
    return smoothed
