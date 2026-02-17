from __future__ import annotations
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.neuron import Neuron


def compute_sttc_matrix(neurons: List["Neuron"], 
                       n_frames: int,
                       time_window: float = 0.033,
                       fs: float = 30.0) -> np.ndarray:
    """
    Compute STTC (Spike Time Tiling Coefficient) matrix - fully vectorized.
    
    Uses the Cutts & Eglen (2014) formula:
    STTC = 0.5 * ((P_A - T_B)/(1 - P_A*T_B) + (P_B - T_A)/(1 - P_B*T_A))
    
    Parameters
    ----------
    neurons : list[Neuron]
        List of Neuron objects
    n_frames : int
        Total number of frames
    time_window : float, optional
        Time window in seconds (dt), by default 0.033
    fs : float, optional
        Sampling frequency, by default 30.0
        
    Returns
    -------
    sttc_matrix : np.ndarray
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
    kernel = np.ones(2 * dt_frames + 1, dtype=np.float32)
    tiled_matrix = np.zeros((n, n_frames), dtype=np.float32)
    for i in range(n):
        if np.any(spike_matrix[i]):
            convolved = np.convolve(spike_matrix[i], kernel, mode='same')
            tiled_matrix[i] = (convolved > 0).astype(np.float32)
    
    # T values: fraction of time each neuron is "active"
    T = tiled_matrix.sum(axis=1) / n_frames
    
    # Spike counts per neuron
    n_spikes = spike_matrix.sum(axis=1)
    
    # P matrix: P[i,j] = fraction of neuron i's spikes within ±dt of neuron j's spikes
    overlap_matrix = spike_matrix @ tiled_matrix.T
    
    with np.errstate(divide='ignore', invalid='ignore'):
        P = overlap_matrix / n_spikes[:, None]
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    
    # STTC formula
    T_row = T[None, :]
    T_col = T[:, None]
    
    denom_A = 1.0 - P * T_row
    with np.errstate(divide='ignore', invalid='ignore'):
        term_A = (P - T_row) / denom_A
        term_A = np.nan_to_num(term_A, nan=0.0, posinf=1.0, neginf=-1.0)
    
    denom_B = 1.0 - P.T * T_col
    with np.errstate(divide='ignore', invalid='ignore'):
        term_B = (P.T - T_col) / denom_B
        term_B = np.nan_to_num(term_B, nan=0.0, posinf=1.0, neginf=-1.0)
    
    sttc_matrix = 0.5 * (term_A + term_B)
    
    sttc_matrix = np.clip(sttc_matrix, -1.0, 1.0)
    np.fill_diagonal(sttc_matrix, 1.0)
    
    # Handle neurons with no spikes
    no_spikes = n_spikes == 0
    sttc_matrix[no_spikes, :] = 0.0
    sttc_matrix[:, no_spikes] = 0.0
    np.fill_diagonal(sttc_matrix, 1.0)
    
    return sttc_matrix.astype(np.float32)


@dataclass
class STTCSimilarity:
    """Produces correlation matrix in [-1, 1]."""
    time_window: float = 0.033
    fs: float = 30.0

    def compute(self, neurons: List["Neuron"], n_frames: int) -> np.ndarray:
        fs = float(neurons[0].fs) if neurons else float(self.fs)
        return compute_sttc_matrix(neurons, n_frames, time_window=float(self.time_window), fs=fs)
