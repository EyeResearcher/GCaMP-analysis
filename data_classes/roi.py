"""ROI class for calcium imaging data."""
import numpy as np
from typing import Optional, Dict
from roi_classifier.prepare_data import roi_feature_extraction
from scipy.ndimage import gaussian_filter1d
class ROI:
    """Represents a Region of Interest from Suite2p."""
    
    def __init__(self, 
                 index: int,
                 f_trace: np.ndarray,
                 cascade_prob: np.ndarray,
                 stats: Optional[Dict] = None,
                 fneu: Optional[np.ndarray] = None,
                 norm_f_trace: Optional[np.ndarray] = None,
                 norm_sp_trace: Optional[np.ndarray] = None):
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
        self.cascade_prob = cascade_prob
        self.stats = stats
        self.fneu = fneu
        
        # Features for classification
        self.features = {}
        
        # Classification result
        self.is_good = None
        
    def __repr__(self):
        return f"ROI(index={self.index}, frames={len(self.f_trace)}, is_good={self.is_good})"
    
    def extract_features(self, sm_norm_f: np.ndarray, sm_norm_sp: np.ndarray) -> dict:
        """Extract features for this ROI using specified normalization."""
        features, validity = roi_feature_extraction(sm_norm_f, sm_norm_sp)
        self.features = features
        if not validity['valid_deriv'] or not validity['valid_prom']:
            self.is_good = False
        return features
    
    def set_classification(self, prediction):
        """Classify ROI using provided classifier model."""
        if not self.features:
            raise ValueError("Features not extracted. Call extract_features() first.")
        if self.is_good is False:
            return 
        self.is_good = bool(prediction)