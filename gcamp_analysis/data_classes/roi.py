"""ROI class for calcium imaging data."""
import numpy as np
from typing import Optional, Dict
from gcamp_analysis.roi_processing.features import compute_roi_features
from scipy.signal import find_peaks
class ROI:
    """Represents a Region of Interest from Suite2p."""
    
    def __init__(self, 
                 index: int,
                 f_trace: np.ndarray,
                 stats: Optional[Dict] = None,
                 fneu: Optional[np.ndarray] = None
                 ):
        """
        Initialize ROI.
        
        Parameters:
            index: Original row index in Suite2p arrays
            f_trace: Raw fluorescence trace
            cascade_prob: raw Cascade spike probability
            stats: Suite2p stat dict
            fneu: Neuropil fluorescence
        """
        self.index = index
        self.f_trace = f_trace
        self.stats = stats
        self.fneu = fneu
        
        # Features for classification
        self.features = {}
        self.peaks = []
        
        # Classification result
        self.is_good = None
        
    def __repr__(self):
        return f"ROI(index={self.index}, frames={len(self.f_trace)}, is_good={self.is_good})"
    
    def extract_features(self, sm_norm_f: np.ndarray) -> dict:
        """Extract features for this ROI using specified normalization."""
        self.peaks, _ = find_peaks(sm_norm_f)
        self.features, validity = compute_roi_features(sm_norm_f)
        self.features_validity = validity
        return self.features
   