"""Preprocessing functions for loading data and computing cascade."""
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from scipy.ndimage import gaussian_filter1d
import logging

from utils.io_utils import load_suite2p_data          # noqa: F401  (re-export)

logger = logging.getLogger(__name__)



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