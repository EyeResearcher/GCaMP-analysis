"""Preprocessing functions for loading data and computing cascade."""
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)

def load_suite2p_data(suite2p_path: Path) -> Dict:
    """
    Load all Suite2p output files.
    
    Returns dict with keys: F, Fneu, spks, stat, ops, iscell
    """
    suite2p_path = Path(suite2p_path)
    data = {}
    
    # Required files
    required = ['F.npy', 'iscell.npy']
    for file in required:
        if not (suite2p_path / file).exists():
            raise FileNotFoundError(f"Required file {file} not found in {suite2p_path}")
    
    # Load arrays
    data['F'] = np.load(suite2p_path / 'F.npy')
    data['iscell'] = np.load(suite2p_path / 'iscell.npy')
    
    # Optional files
    if (suite2p_path / 'Fneu.npy').exists():
        data['Fneu'] = np.load(suite2p_path / 'Fneu.npy')
    else:
        logger.warning("Fneu.npy not found, using zeros")
        data['Fneu'] = np.zeros_like(data['F'])
        
    if (suite2p_path / 'spks.npy').exists():
        data['spks'] = np.load(suite2p_path / 'spks.npy')
        
    if (suite2p_path / 'stat.npy').exists():
        data['stat'] = np.load(suite2p_path / 'stat.npy', allow_pickle=True)
        
    if (suite2p_path / 'ops.npy').exists():
        data['ops'] = np.load(suite2p_path / 'ops.npy', allow_pickle=True).item()
        data['fs'] = data['ops'].get('fs', 30.0)
    else:
        data['fs'] = 30.0
        
    logger.info(f"Loaded Suite2p data: {data['F'].shape[0]} ROIs, {data['F'].shape[1]} frames")
    
    return data

def compute_cascade_probabilities(f_traces: np.ndarray, 
                                 cascade_model,
                                 batch_size: int = 64) -> np.ndarray:
    """
    Compute spike probabilities using Cascade model.
    
    Parameters:
        f_traces: (n_rois, n_frames) fluorescence traces
        cascade_model: Loaded cascade model
        batch_size: Process in batches for memory efficiency
        
    Returns:
        (n_rois, n_frames) spike probabilities
    """
    n_rois, n_frames = f_traces.shape
    probabilities = np.zeros_like(f_traces)
    
    # Process in batches
    for i in range(0, n_rois, batch_size):
        batch_end = min(i + batch_size, n_rois)
        batch = f_traces[i:batch_end]
        
        if cascade_model is not None:
            # Use actual model
            batch_prob = cascade_model.predict(batch)
        else:
            # Mock for testing
            batch_prob = np.random.random(batch.shape) * 0.1
            
        probabilities[i:batch_end] = batch_prob
        
    return probabilities

def smooth_cascade_prob(prob: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to cascade probability.
    
    Parameters:
        prob: 1D probability array
        sigma: Gaussian kernel width in frames
        
    Returns:
        Smoothed probability
    """
    if sigma > 0:
        return gaussian_filter1d(prob, sigma=sigma, mode='reflect')
    return prob