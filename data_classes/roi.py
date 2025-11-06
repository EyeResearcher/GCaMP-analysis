"""ROI class for calcium imaging data."""
import numpy as np
from typing import Optional, Dict

class ROI:
    """Represents a Region of Interest from Suite2p."""
    
    def __init__(self, 
                 index: int,
                 f_trace: np.ndarray,
                 cascade_prob: np.ndarray,
                 spatial_footprint: Optional[Dict] = None,
                 fneu: Optional[np.ndarray] = None):
        """
        Initialize ROI.
        
        Parameters:
            index: Original row index in Suite2p arrays
            f_trace: Fluorescence trace
            cascade_prob: Cascade spike probability
            spatial_footprint: Suite2p stat dict
            fneu: Neuropil fluorescence
        """
        self.index = index
        self.f_trace = f_trace
        self.cascade_prob = cascade_prob
        self.spatial_footprint = spatial_footprint
        self.fneu = fneu
        
        # Features for classification
        self.features = {}
        
        # Classification result
        self.is_good = None
        
    def __repr__(self):
        return f"ROI(index={self.index}, frames={len(self.f_trace)}, is_good={self.is_good})"