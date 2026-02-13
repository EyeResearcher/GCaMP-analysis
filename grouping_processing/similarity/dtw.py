from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import logging
import numpy as np

if TYPE_CHECKING:
    from data_classes.neuron import Neuron

logger = logging.getLogger(__name__)


def compute_dtw_matrix(neurons: List["Neuron"], 
                      downsample_factor: int = 3,
                      use_gpu: bool = True) -> Optional[np.ndarray]:
    """
    Compute DTW (Dynamic Time Warping) distance matrix with GPU acceleration.
    Skips computation if GPU is not available to avoid hangups.
    
    Parameters
    ----------
    neurons : list[Neuron]
        List of Neuron objects
    downsample_factor : int, optional
        Downsample traces for speed, by default 3
    use_gpu : bool, optional
        Use GPU acceleration if available, by default True
        
    Returns
    -------
    distance_matrix : np.ndarray or None
        Distance matrix, or None if GPU not available and use_gpu=True
    """
    try:
        import torch
    except (ImportError, OSError) as e:
        logger.warning(f"PyTorch not available ({e.__class__.__name__}) - skipping DTW computation")
        return None
    
    if use_gpu and not torch.cuda.is_available():
        logger.warning("GPU not available - skipping DTW computation to avoid hangups")
        return None
    
    n_neurons = len(neurons)
    
    # Prepare downsampled traces
    traces = []
    for neuron in neurons:
        trace = neuron.f_trace
        if downsample_factor > 1:
            trace = trace[::downsample_factor]
        trace = (trace - np.mean(trace)) / (np.std(trace) + 1e-8)
        traces.append(trace)
    
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
    
    distance_matrix = _compute_soft_dtw_matrix(traces_tensor, device, gamma=1.0)
    
    return distance_matrix


def _compute_soft_dtw_matrix(traces, device, gamma: float = 1.0) -> np.ndarray:
    """
    Compute pairwise SoftDTW distances (fast, differentiable DTW approximation).
    
    Parameters
    ----------
    traces : torch.Tensor
        Tensor of shape (n_neurons, seq_len)
    device : torch.device
        Torch device
    gamma : float, optional
        Smoothing parameter (smaller = closer to true DTW), by default 1.0
        
    Returns
    -------
    dist_matrix : np.ndarray
        Distance matrix
    """
    n_neurons, seq_len = traces.shape
    dist_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    
    batch_size = 32
    
    for i in range(n_neurons):
        trace_i = traces[i:i+1]
        
        for j_start in range(i, n_neurons, batch_size):
            j_end = min(j_start + batch_size, n_neurons)
            traces_j = traces[j_start:j_end]
            
            distances = _batch_soft_dtw(trace_i, traces_j, gamma, device)
            
            dist_matrix[i, j_start:j_end] = distances.cpu().numpy()
            dist_matrix[j_start:j_end, i] = distances.cpu().numpy()
    
    return dist_matrix


def _batch_soft_dtw(trace_i, traces_j, gamma: float, device):
    """
    Compute SoftDTW from one trace to a batch of traces.
    
    Parameters
    ----------
    trace_i : torch.Tensor
        (1, T) single trace
    traces_j : torch.Tensor
        (B, T) batch of traces
    gamma : float
        Smoothing parameter
    device : torch.device
        Torch device
        
    Returns
    -------
    distances : torch.Tensor
        (B,) distances
    """
    import torch
    
    B, T = traces_j.shape
    
    trace_i_exp = trace_i.unsqueeze(2)
    traces_j_exp = traces_j.unsqueeze(1)
    
    cost = (trace_i_exp - traces_j_exp) ** 2
    
    R = torch.full((B, T + 1, T + 1), float('inf'), device=device)
    R[:, 0, 0] = 0
    
    for i in range(1, T + 1):
        for j in range(1, T + 1):
            r_prev = torch.stack([
                R[:, i-1, j],
                R[:, i, j-1],
                R[:, i-1, j-1]
            ], dim=1)
            
            soft_min = -gamma * torch.logsumexp(-r_prev / gamma, dim=1)
            R[:, i, j] = cost[:, i-1, j-1] + soft_min
    
    return R[:, T, T]


@dataclass
class DTWSimilarity:
    """Produces distance matrix (>=0). May return None if skipped."""
    downsample_factor: int = 3
    use_gpu: bool = True

    def compute(self, neurons: List["Neuron"]) -> Optional[np.ndarray]:
        return compute_dtw_matrix(neurons, downsample_factor=int(self.downsample_factor), use_gpu=bool(self.use_gpu))
