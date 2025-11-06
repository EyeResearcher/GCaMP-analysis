"""Spike class for detected events."""
import numpy as np
from typing import Optional

class Spike:
    """Represents a detected spike event."""
    
    def __init__(self,
                 frame_index: int,
                 cascade_peak_idx: int,
                 prob_height: float,
                 f_value: float):
        """
        Initialize Spike.
        
        Parameters:
            frame_index: Frame index of F peak
            cascade_peak_idx: Frame index of cascade peak
            prob_height: Cascade probability at peak
            f_value: Fluorescence value at peak
        """
        self.frame_index = frame_index
        self.cascade_peak_idx = cascade_peak_idx
        self.prob_height = prob_height
        self.f_value = f_value
        self.fluorescence_peak = f_value  # Alias for compatibility
        
        # Features for classification (populated later)
        self.prominence = 0.0
        self.baseline_delta = 0.0
        self.window_width = 0.0
        self.window_auc = 0.0
        self.rise_slope = 0.0
        self.decay_tau = 5.0
        
        # Classification result
        self.is_valid = None
        
    def __repr__(self):
        return f"Spike(frame={self.frame_index}, prob={self.prob_height:.3f}, F={self.f_value:.2f})"