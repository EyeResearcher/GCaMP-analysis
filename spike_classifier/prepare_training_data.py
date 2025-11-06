"""Extract the 8 spike features for training."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def compute_spike_features(
    spike_key: str,
    f_trace: np.ndarray,
    spike_prob: np.ndarray,
    spike_info: Dict
) -> Dict:
    """
    Compute the 8 spike features for classification.
    
    Features:
    1. prob_height - Cascade probability at peak
    2. prominence - Left-base prominence
    3. fluorescence_peak - F value at peak  
    4. baseline_delta - Peak minus baseline
    5. window_width - Width of spike window
    6. window_auc - Area under curve
    7. rise_slope - Slope of rise phase
    8. decay_tau - Decay time constant
    """
    features = {'spike_key': spike_key}
    
    # Get spike indices
    prob_idx = spike_info.get('prob_idx', 0)
    f_idx = spike_info.get('f_idx', prob_idx)
    
    # 1. Probability height
    features['prob_height'] = spike_prob[prob_idx] if prob_idx < len(spike_prob) else 0.0
    
    # 2. Prominence (would need to be computed from context)
    features['prominence'] = spike_info.get('prominence', 0.0)
    
    # 3. Fluorescence peak
    features['fluorescence_peak'] = f_trace[f_idx] if f_idx < len(f_trace) else 0.0
    
    # 4. Baseline delta
    baseline_window = max(0, f_idx - 30)
    baseline = np.percentile(f_trace[baseline_window:f_idx], 10) if baseline_window < f_idx else 0.0
    features['baseline_delta'] = features['fluorescence_peak'] - baseline
    
    # 5-8. Window-based features
    window_start = spike_info.get('window_start', f_idx - 5)
    window_end = spike_info.get('window_end', f_idx + 15)
    
    if window_end > window_start:
        window = f_trace[window_start:window_end]
        features['window_width'] = float(window_end - window_start)
        features['window_auc'] = float(np.trapz(window))
        
        # Rise slope
        rise_end = f_idx
        if rise_end > window_start:
            rise_trace = f_trace[window_start:rise_end]
            features['rise_slope'] = (rise_trace[-1] - rise_trace[0]) / len(rise_trace) if len(rise_trace) > 1 else 0.0
        else:
            features['rise_slope'] = 0.0
            
        # Decay tau (simplified - would need proper fitting)
        features['decay_tau'] = spike_info.get('decay_tau', 1.0)
    else:
        features['window_width'] = 0.0
        features['window_auc'] = 0.0
        features['rise_slope'] = 0.0
        features['decay_tau'] = 0.0
    
    return features

def prepare_spike_training_data(
    annotations_path: Path,
    features_path: Path = None,
    output_path: Path = None
) -> pd.DataFrame:
    """
    Prepare spike training data with the 8 features.
    
    Args:
        annotations_path: Path to spike_annotations.csv
        features_path: Optional path to existing spike_features.csv
        output_path: Path to save prepared training data
        
    Returns:
        DataFrame with 8 features + label
    """
    # Load annotations
    annotations = pd.read_csv(annotations_path)
    
    # If features already exist, use them
    if features_path and features_path.exists():
        existing_features = pd.read_csv(features_path)
        # Select only our 8 features
        feature_cols = [
            'prob_height', 'prominence', 'fluorescence_peak', 
            'baseline_delta', 'window_width', 'window_auc', 
            'rise_slope', 'decay_tau'
        ]
        
        # Merge with annotations
        merged = pd.merge(
            existing_features[['spike_key'] + feature_cols],
            annotations[['spike_key', 'label']],
            on='spike_key'
        )
        
        if output_path:
            merged.to_csv(output_path, index=False)
            
        return merged
    
    else:
        logger.warning("No existing features found. You'll need to extract features from raw data.")
        return pd.DataFrame()