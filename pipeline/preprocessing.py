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