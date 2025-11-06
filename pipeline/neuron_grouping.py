"""Neuron grouping using STTC and DTW methods."""
from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING, Any
import logging

if TYPE_CHECKING:
    from data_classes import Neuron
    import torch

logger = logging.getLogger(__name__)

def compute_sttc_matrix(neurons: List[Neuron], 
                       n_frames: int,
                       time_window: float = 0.033,
                       fs: float = 30.0) -> np.ndarray:
    """
    Compute STTC (Spike Time Tiling Coefficient) matrix.
    
    Parameters:
        neurons: List of Neuron objects
        n_frames: Total number of frames
        time_window: Time window in seconds
        fs: Sampling frequency
        
    Returns:
        Symmetric STTC matrix
    """
    n_neurons = len(neurons)
    sttc_matrix = np.zeros((n_neurons, n_neurons))
    
    # Convert time window to frames
    window_frames = int(time_window * fs)
    
    for i in range(n_neurons):
        for j in range(i, n_neurons):
            if i == j:
                sttc_matrix[i, j] = 1.0
            else:
                # Get spike trains
                spikes_i = np.array([s.frame_index for s in neurons[i].spikes])
                spikes_j = np.array([s.frame_index for s in neurons[j].spikes])
                
                if len(spikes_i) == 0 or len(spikes_j) == 0:
                    sttc_matrix[i, j] = 0.0
                    sttc_matrix[j, i] = 0.0
                    continue
                
                # Calculate STTC
                # P_A: proportion of time with spikes from A
                tiles_a = np.zeros(n_frames, dtype=bool)
                for spike in spikes_i:
                    start = max(0, spike - window_frames)
                    end = min(n_frames, spike + window_frames + 1)
                    tiles_a[start:end] = True
                P_A = np.mean(tiles_a)
                
                # P_B: proportion of time with spikes from B
                tiles_b = np.zeros(n_frames, dtype=bool)
                for spike in spikes_j:
                    start = max(0, spike - window_frames)
                    end = min(n_frames, spike + window_frames + 1)
                    tiles_b[start:end] = True
                P_B = np.mean(tiles_b)
                
                # T_A: proportion of spikes from A that fall in B's tiles
                if len(spikes_i) > 0:
                    in_b_tiles = np.sum([tiles_b[spike] for spike in spikes_i])
                    T_A = in_b_tiles / len(spikes_i)
                else:
                    T_A = 0
                
                # T_B: proportion of spikes from B that fall in A's tiles
                if len(spikes_j) > 0:
                    in_a_tiles = np.sum([tiles_a[spike] for spike in spikes_j])
                    T_B = in_a_tiles / len(spikes_j)
                else:
                    T_B = 0
                
                # STTC calculation
                if P_A == 1:
                    term1 = 1.0
                else:
                    term1 = (T_A - P_B) / (1 - P_A * P_B) if (1 - P_A * P_B) > 0 else 0
                
                if P_B == 1:
                    term2 = 1.0
                else:
                    term2 = (T_B - P_A) / (1 - P_A * P_B) if (1 - P_A * P_B) > 0 else 0
                
                sttc = 0.5 * (term1 + term2)
                sttc_matrix[i, j] = sttc
                sttc_matrix[j, i] = sttc
    
    return sttc_matrix

def compute_dtw_matrix(neurons: List[Neuron], 
                      downsample_factor: int = 3,
                      use_gpu: bool = True) -> Optional[np.ndarray]:
    """
    Compute DTW (Dynamic Time Warping) distance matrix with GPU acceleration.
    Skips computation if GPU is not available to avoid hangups.
    
    Parameters:
        neurons: List of Neuron objects
        downsample_factor: Downsample traces for speed
        use_gpu: Use GPU acceleration if available
        
    Returns:
        Distance matrix, or None if GPU not available and use_gpu=True
    """
    try:
        import torch
    except (ImportError, OSError) as e:
        logger.warning(f"PyTorch not available ({e.__class__.__name__}) - skipping DTW computation")
        return None
    
    # Check GPU availability first
    if use_gpu and not torch.cuda.is_available():
        logger.warning("GPU not available - skipping DTW computation to avoid hangups")
        return None
    
    n_neurons = len(neurons)
    
    # Prepare downsampled traces
    traces = []
    for neuron in neurons:
        trace = neuron.raw_fluorescence
        # Downsample
        if downsample_factor > 1:
            trace = trace[::downsample_factor]
        # Z-score normalize
        trace = (trace - np.mean(trace)) / (np.std(trace) + 1e-8)
        traces.append(trace)
    
    # Check GPU availability
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    if use_gpu and torch.cuda.is_available():
        logger.info(f"Using GPU ({torch.cuda.get_device_name(0)}) for DTW computation")
    else:
        logger.info("Using CPU for DTW computation")
    
    # Convert to torch tensors and pad to same length
    max_len = max(len(t) for t in traces)
    traces_padded = []
    for trace in traces:
        padded = np.pad(trace, (0, max_len - len(trace)), mode='constant', constant_values=0)
        traces_padded.append(padded)
    
    traces_tensor = torch.tensor(traces_padded, dtype=torch.float32, device=device)
    
    # Compute DTW distance matrix on GPU
    distance_matrix = _compute_dtw_gpu(traces_tensor, device)
    
    return distance_matrix


def _compute_dtw_gpu(traces: Any, device: Any) -> np.ndarray:
    """
    GPU-accelerated DTW distance computation.
    
    Parameters:
        traces: Tensor of shape (n_neurons, max_len)
        device: torch device
        
    Returns:
        Distance matrix as numpy array
    """
    n_neurons, seq_len = traces.shape
    
    # Initialize distance matrix
    dist_matrix = torch.zeros((n_neurons, n_neurons), device=device)
    
    # Compute pairwise DTW distances
    # Process in batches to avoid memory issues
    batch_size = min(50, n_neurons)
    
    for i in range(0, n_neurons, batch_size):
        i_end = min(i + batch_size, n_neurons)
        batch_traces_i = traces[i:i_end]  # (batch_size, seq_len)
        
        for j in range(i, n_neurons, batch_size):
            j_end = min(j + batch_size, n_neurons)
            batch_traces_j = traces[j:j_end]  # (batch_size, seq_len)
            
            # Compute DTW for this batch pair
            batch_dist = _batch_dtw(batch_traces_i, batch_traces_j, device)
            
            # Fill in the distance matrix (symmetric)
            dist_matrix[i:i_end, j:j_end] = batch_dist
            if i != j:
                dist_matrix[j:j_end, i:i_end] = batch_dist.T
    
    return dist_matrix.cpu().numpy()


def _batch_dtw(traces_i: Any, traces_j: Any, device: Any) -> Any:
    """
    Compute DTW distances for a batch of trace pairs.
    Uses memory-efficient pairwise computation.
    
    Parameters:
        traces_i: Tensor of shape (n_i, seq_len)
        traces_j: Tensor of shape (n_j, seq_len)
        device: torch device
        
    Returns:
        Distance matrix of shape (n_i, n_j)
    """
    import torch
    
    n_i = traces_i.shape[0]
    n_j = traces_j.shape[0]
    
    # Initialize output distance matrix
    distances = torch.zeros((n_i, n_j), device=device)
    
    # Compute DTW for each pair
    for i in range(n_i):
        for j in range(n_j):
            distances[i, j] = _single_dtw_gpu(traces_i[i], traces_j[j], device)
    
    return distances


def _single_dtw_gpu(trace_i: Any, trace_j: Any, device: Any) -> Any:
    """
    Compute DTW distance between two sequences on GPU.
    
    Parameters:
        trace_i: Tensor of shape (seq_len,)
        trace_j: Tensor of shape (seq_len,)
        device: torch device
        
    Returns:
        Scalar DTW distance
    """
    import torch
    
    n = len(trace_i)
    m = len(trace_j)
    
    # Create cost matrix (only need current and previous row for memory efficiency)
    dtw_matrix = torch.full((2, m + 1), float('inf'), device=device)
    dtw_matrix[0, 0] = 0
    
    # Dynamic programming
    for i in range(1, n + 1):
        curr_row = i % 2
        prev_row = (i - 1) % 2
        
        dtw_matrix[curr_row, :] = float('inf')
        
        for j in range(1, m + 1):
            cost = torch.abs(trace_i[i - 1] - trace_j[j - 1])
            dtw_matrix[curr_row, j] = cost + torch.min(
                torch.stack([
                    dtw_matrix[prev_row, j],      # insertion
                    dtw_matrix[curr_row, j - 1],  # deletion
                    dtw_matrix[prev_row, j - 1]   # match
                ])
            )
    
    return dtw_matrix[n % 2, m]

def group_neurons_by_sttc(neurons: List[Neuron],
                         n_frames: int,
                         time_window: float = 0.033,
                         correlation_threshold: float = 0.1,
                         linkage_method: str = 'average',
                         distance_threshold: float = 0.3,
                         min_group_size: int = 2,
                         **kwargs) -> Tuple[List[List[Neuron]], np.ndarray]:
    """
    Group neurons using STTC correlation.
    
    Returns:
        List of neuron groups and STTC matrix
    """
    if len(neurons) < 2:
        return [neurons] if neurons else [], np.array([[1.0]])
    
    # Compute STTC matrix
    sttc_matrix = compute_sttc_matrix(neurons, n_frames, time_window)
    
    # Convert correlation to distance (1 - correlation)
    distance_matrix = 1 - sttc_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, method=linkage_method)
    
    # Get clusters
    clusters = fcluster(Z, distance_threshold, criterion='distance')
    
    # Organize into groups
    groups = []
    for cluster_id in np.unique(clusters):
        group = [neurons[i] for i in range(len(neurons)) if clusters[i] == cluster_id]
        if len(group) >= min_group_size:
            groups.append(group)
    
    return groups, sttc_matrix

def group_neurons_by_dtw(neurons: List[Neuron],
                        downsample_factor: int = 3,
                        linkage_method: str = 'average',
                        distance_percentile: int = 30,
                        min_group_size: int = 2,
                        **kwargs) -> Tuple[List[List[Neuron]], Optional[np.ndarray]]:
    """
    Group neurons using DTW distance.
    Returns empty groups if GPU not available.
    
    Returns:
        List of neuron groups and DTW distance matrix (or None if skipped)
    """
    if len(neurons) < 2:
        return [neurons] if neurons else [], np.array([[0.0]])
    
    # Compute DTW distance matrix (returns None if GPU not available)
    dtw_matrix = compute_dtw_matrix(neurons, downsample_factor)
    
    if dtw_matrix is None:
        logger.warning("DTW computation skipped - returning empty groups")
        return [], None
    
    # Use percentile for threshold
    distance_threshold = np.percentile(dtw_matrix[dtw_matrix > 0], distance_percentile)
    
    # Hierarchical clustering
    condensed_dist = squareform(dtw_matrix)
    Z = linkage(condensed_dist, method=linkage_method)
    
    # Get clusters
    clusters = fcluster(Z, distance_threshold, criterion='distance')
    
    # Organize into groups
    groups = []
    for cluster_id in np.unique(clusters):
        group = [neurons[i] for i in range(len(neurons)) if clusters[i] == cluster_id]
        if len(group) >= min_group_size:
            groups.append(group)
    
    return groups, dtw_matrix

def compare_groupings(sttc_groups: List[List[Neuron]], 
                     dtw_groups: List[List[Neuron]],
                     neurons: List[Neuron]) -> Dict:
    """
    Compare STTC and DTW groupings.
    
    Returns:
        Dict with comparison metrics
    """
    # Create membership arrays
    sttc_membership = np.zeros(len(neurons))
    for i, group in enumerate(sttc_groups):
        for neuron in group:
            idx = neurons.index(neuron)
            sttc_membership[idx] = i
    
    dtw_membership = np.zeros(len(neurons))
    for i, group in enumerate(dtw_groups):
        for neuron in group:
            idx = neurons.index(neuron)
            dtw_membership[idx] = i
    
    # Calculate agreement
    agreement = np.mean(sttc_membership == dtw_membership)
    
    # Log comparison
    logger.info(f"  Grouping comparison:")
    logger.info(f"    STTC groups: {len(sttc_groups)}")
    logger.info(f"    DTW groups: {len(dtw_groups)}")
    logger.info(f"    Agreement: {agreement:.2%}")
    
    return {
        'n_sttc_groups': len(sttc_groups),
        'n_dtw_groups': len(dtw_groups),
        'agreement': agreement
    }