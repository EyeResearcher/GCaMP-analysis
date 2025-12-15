"""Neuron grouping using STTC and DTW methods."""
from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING, Any
import logging
import torch
from joblib import Parallel, delayed
from data_classes.neuron_group import NeuronGroup

if TYPE_CHECKING:
    from data_classes import Neuron

logger = logging.getLogger(__name__)

def compute_sttc_matrix(neurons: List[Neuron], 
                       n_frames: int,
                       time_window: float = 0.033,
                       fs: float = 30.0) -> np.ndarray:
    """
    Compute STTC (Spike Time Tiling Coefficient) matrix - fully vectorized.
    
    Uses the Cutts & Eglen (2014) formula:
    STTC = 0.5 * ((P_A - T_B)/(1 - P_A*T_B) + (P_B - T_A)/(1 - P_B*T_A))
    
    Parameters:
        neurons: List of Neuron objects
        n_frames: Total number of frames
        time_window: Time window in seconds (dt)
        fs: Sampling frequency
        
    Returns:
        Symmetric STTC matrix with values in [-1, 1]
    """
    n = len(neurons)
    if n == 0:
        return np.array([[]])
    
    dt_frames = int(time_window * fs)
    
    # Build binary spike matrix (n_neurons x n_frames)
    spike_matrix = np.zeros((n, n_frames), dtype=np.float32)
    for i, neuron in enumerate(neurons):
        if hasattr(neuron, 'spikes') and neuron.spikes:
            times = [s.sm_f_idx for s in neuron.spikes]
            valid_times = [t for t in times if 0 <= t < n_frames]
            if valid_times:
                spike_matrix[i, valid_times] = 1.0
    
    # Build tiled matrix (dilate each spike by ±dt_frames)
    # Use convolution for fast dilation
    kernel = np.ones(2 * dt_frames + 1, dtype=np.float32)
    tiled_matrix = np.zeros((n, n_frames), dtype=np.float32)
    for i in range(n):
        if np.any(spike_matrix[i]):
            convolved = np.convolve(spike_matrix[i], kernel, mode='same')
            tiled_matrix[i] = (convolved > 0).astype(np.float32)
    
    # T values: fraction of time each neuron is "active" (within dt of any spike)
    T = tiled_matrix.sum(axis=1) / n_frames  # (n,)
    
    # Spike counts per neuron
    n_spikes = spike_matrix.sum(axis=1)  # (n,)
    
    # P matrix: P[i,j] = fraction of neuron i's spikes within ±dt of neuron j's spikes
    # = (spike_matrix[i] @ tiled_matrix[j].T) / n_spikes[i]
    # Vectorized: P = (spike_matrix @ tiled_matrix.T) / n_spikes[:, None]
    
    # spike_matrix @ tiled_matrix.T gives (n, n) where [i,j] = count of i's spikes in j's tiled region
    overlap_matrix = spike_matrix @ tiled_matrix.T  # (n, n)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        P = overlap_matrix / n_spikes[:, None]  # P[i,j] = P_i given j's tiling
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    
    # STTC formula: 0.5 * ((P[i,j] - T[j])/(1 - P[i,j]*T[j]) + (P[j,i] - T[i])/(1 - P[j,i]*T[i]))
    # Compute both terms
    T_row = T[None, :]  # (1, n) -> broadcast to (n, n) as T_B
    T_col = T[:, None]  # (n, 1) -> broadcast to (n, n) as T_A
    
    # Term A: (P[i,j] - T[j]) / (1 - P[i,j] * T[j])
    denom_A = 1.0 - P * T_row
    with np.errstate(divide='ignore', invalid='ignore'):
        term_A = (P - T_row) / denom_A
        term_A = np.nan_to_num(term_A, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Term B: (P[j,i] - T[i]) / (1 - P[j,i] * T[i])  -> use P.T
    denom_B = 1.0 - P.T * T_col
    with np.errstate(divide='ignore', invalid='ignore'):
        term_B = (P.T - T_col) / denom_B
        term_B = np.nan_to_num(term_B, nan=0.0, posinf=1.0, neginf=-1.0)
    
    sttc_matrix = 0.5 * (term_A + term_B)
    
    # Clamp to [-1, 1] and set diagonal to 1
    sttc_matrix = np.clip(sttc_matrix, -1.0, 1.0)
    np.fill_diagonal(sttc_matrix, 1.0)
    
    # Handle neurons with no spikes (set their row/col to 0 except diagonal)
    no_spikes = n_spikes == 0
    sttc_matrix[no_spikes, :] = 0.0
    sttc_matrix[:, no_spikes] = 0.0
    np.fill_diagonal(sttc_matrix, 1.0)
    
    return sttc_matrix.astype(np.float32)



def group_neurons_by_sttc(neurons: List[Neuron],
                         n_frames: int,
                         time_window: float = 0.033,
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
    
    # Compute STTC matrix using elephant
    fs = neurons[0].fs if neurons else 30.0
    sttc_matrix = compute_sttc_matrix(neurons, n_frames, time_window, fs)
    
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
            neuron_group = NeuronGroup(f"sttc_{cluster_id}", group, method='sttc', t_win=time_window, sttc_thresh=1-distance_threshold)
            groups.append(neuron_group)
    
    return groups, sttc_matrix

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
        trace = neuron.f_trace
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
    traces_padded = np.zeros((n_neurons, max_len), dtype=np.float32)
    for i, trace in enumerate(traces):
        traces_padded[i, :len(trace)] = trace
    
    traces_tensor = torch.from_numpy(traces_padded).to(device)
    
    # Compute DTW distance matrix using fast SoftDTW approximation
    distance_matrix = _compute_soft_dtw_matrix(traces_tensor, device, gamma=1.0)
    
    return distance_matrix


def _compute_soft_dtw_matrix(traces: torch.Tensor, device: torch.device, gamma: float = 1.0) -> np.ndarray:
    """
    Compute pairwise SoftDTW distances (fast, differentiable DTW approximation).
    
    SoftDTW replaces the min() in DTW with soft-min, enabling vectorized computation.
    As gamma -> 0, SoftDTW -> DTW.
    
    Parameters:
        traces: Tensor of shape (n_neurons, seq_len)
        device: torch device
        gamma: Smoothing parameter (smaller = closer to true DTW)
        
    Returns:
        Distance matrix as numpy array
    """
    n_neurons, seq_len = traces.shape
    
    # Compute pairwise squared Euclidean distance matrix for all timesteps
    dist_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    
    batch_size = 32  # Process this many pairs at once
    
    # Use vectorized soft-DTW computation
    for i in range(n_neurons):
        # Compute distances from neuron i to all neurons j >= i
        trace_i = traces[i:i+1]  # (1, T)
        
        # Process in batches
        for j_start in range(i, n_neurons, batch_size):
            j_end = min(j_start + batch_size, n_neurons)
            traces_j = traces[j_start:j_end]  # (batch, T)
            
            # Compute SoftDTW for this batch
            distances = _batch_soft_dtw(trace_i, traces_j, gamma, device)
            
            # Fill symmetric matrix
            dist_matrix[i, j_start:j_end] = distances.cpu().numpy()
            dist_matrix[j_start:j_end, i] = distances.cpu().numpy()
    
    return dist_matrix


def _batch_soft_dtw(trace_i: torch.Tensor, traces_j: torch.Tensor, gamma: float, device: torch.device) -> torch.Tensor:
    """
    Compute SoftDTW from one trace to a batch of traces.
    
    Parameters:
        trace_i: (1, T) single trace
        traces_j: (B, T) batch of traces
        gamma: smoothing parameter
        device: torch device
        
    Returns:
        (B,) distances
    """
    B, T = traces_j.shape
    
    # Compute cost matrix: |trace_i[t1] - traces_j[:, t2]|^2
    # trace_i: (1, T) -> (1, T, 1)
    # traces_j: (B, T) -> (B, 1, T)
    trace_i_exp = trace_i.unsqueeze(2)  # (1, T, 1)
    traces_j_exp = traces_j.unsqueeze(1)  # (B, 1, T)
    
    # Cost matrix: (B, T, T) where cost[b, i, j] = (trace_i[i] - traces_j[b, j])^2
    cost = (trace_i_exp - traces_j_exp) ** 2  # (B, T, T) via broadcasting
    
    # SoftDTW dynamic programming
    # R[b, i, j] = cost[b, i, j] + softmin(R[b, i-1, j], R[b, i, j-1], R[b, i-1, j-1])
    R = torch.full((B, T + 1, T + 1), float('inf'), device=device)
    R[:, 0, 0] = 0
    
    for i in range(1, T + 1):
        for j in range(1, T + 1):
            # Soft minimum of three predecessors
            r_prev = torch.stack([
                R[:, i-1, j],
                R[:, i, j-1],
                R[:, i-1, j-1]
            ], dim=1)  # (B, 3)
            
            soft_min = -gamma * torch.logsumexp(-r_prev / gamma, dim=1)
            R[:, i, j] = cost[:, i-1, j-1] + soft_min
    
    return R[:, T, T]


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
            groups.append(NeuronGroup(f"dtw_{cluster_id}", group, method='dtw'))
    
    return groups, dtw_matrix

def compare_groupings(sttc_groups: List[NeuronGroup], 
                     dtw_groups: List[NeuronGroup],
                        sttc_matrix: np.ndarray,
                        dtw_matrix: np.ndarray,
                     neurons: List[Neuron]) -> Dict:
    """
    Compare STTC and DTW groupings.
    
    Returns:
        Dict with comparison metrics
    """
    # Create membership arrays
    sttc_membership = np.zeros(len(neurons))
    sttc_mean_stats = []    
    for i, group in enumerate(sttc_groups):
        mean_stats = group.get_mean_spike_stats(sttc_matrix, dtw_matrix)
        mean_stats_ids = {"time window": group.t_win, "sttc_thresh": group.sttc_thresh,
                           "dtw_thresh": group.dtw_thresh, "group_id": group.group_id, "neuron_indices": group.neuron_indices, 
                           "number_neurons" : group.size, 
                           "method": "sttc", **mean_stats}
        sttc_mean_stats.append(mean_stats_ids)
        for neuron in group.neurons:
            idx = neurons.index(neuron)
            sttc_membership[idx] = i
    if not dtw_groups:
        logger.info(f"  Grouping comparison:")
        logger.info(f"    STTC groups: {len(sttc_groups)}")
        logger.info(f"    DTW groups: n/a (DTW grouping skipped)")
        logger.info(f"    Agreement: n/a (DTW grouping skipped)")
        return {
            'n_sttc_groups': len(sttc_groups),
            'n_dtw_groups': 0,
            'agreement': 0.0,
            'combined_stats' : sttc_mean_stats
        }
    dtw_mean_stats = []
    dtw_membership = np.zeros(len(neurons))
    for i, group in enumerate(dtw_groups):
        mean_stats = group.get_mean_spike_stats(sttc_matrix, dtw_matrix)
        mean_stats_ids = {"time window": group.t_win, "sttc_thresh": group.sttc_thresh, "dtw_thresh": group.dtw_thresh, "group_id": group.group_id,"method": "dtw", **mean_stats}
        dtw_mean_stats.append(mean_stats_ids)
        for neuron in group.neurons:
            idx = neurons.index(neuron)
            dtw_membership[idx] = i
    
    # Calculate agreement
    agreement = np.mean(sttc_membership == dtw_membership)
    
    # Log comparison
    logger.info(f"  Grouping comparison:")
    logger.info(f"    STTC groups: {len(sttc_groups)}")
    logger.info(f"    DTW groups: {len(dtw_groups)}")
    logger.info(f"    Agreement: {agreement:.2%}")
    
    # Combine stats (note: extend returns None, so use + concatenation)
    combined_stats = sttc_mean_stats + dtw_mean_stats
    
    return {
        'n_sttc_groups': len(sttc_groups),
        'n_dtw_groups': len(dtw_groups),
        'agreement': agreement,
        'combined_stats': combined_stats
    }