"""IO handlers for saving pipeline artefacts (filtered Suite2p files, etc.)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

from utils.visualization import visualize_neuron_groups          # noqa: F401  (re-export)

logger = logging.getLogger(__name__)



def save_filtered_suite2p(video_path: Path, 
                         good_roi_mask: np.ndarray,
                         suite2p_data: Dict,
                         cascade_prob: Optional[np.ndarray] = None) -> Path:
    """
    Save filtered Suite2p files with only good ROIs.
    
    Creates filtered_suite2p/plane0/ with filtered arrays.
    """
    # Create filtered_suite2p directory
    filtered_dir = video_path / 'filtered_suite2p' / 'plane0'
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to filter (2D arrays where first dimension is ROIs)
    files_to_filter = {
        'F': suite2p_data.get('F'),
        'Fneu': suite2p_data.get('Fneu'),
        'spks': suite2p_data.get('spks'),
        'iscell': suite2p_data.get('iscell')
    }
    
    # Save filtered arrays
    for name, data in files_to_filter.items():
        if data is not None:
            filtered_data = data[good_roi_mask]
            save_path = filtered_dir / f'{name}.npy'
            np.save(save_path, filtered_data)
            logger.debug(f"Saved filtered {name}: {data.shape} -> {filtered_data.shape}")
    
    # Handle stat (list of dicts)
    if 'stat' in suite2p_data and suite2p_data['stat'] is not None:
        filtered_stat = [suite2p_data['stat'][i] for i in np.where(good_roi_mask)[0]]
        np.save(filtered_dir / 'stat.npy', filtered_stat, allow_pickle=True)
        logger.debug(f"Saved filtered stat: {len(suite2p_data['stat'])} -> {len(filtered_stat)}")
    else: 
        raise FileNotFoundError(f"No stat data found in suite2p_data for video {video_path}")
  
    if 'ops' in suite2p_data and suite2p_data['ops'] is not None:
        np.save(filtered_dir / 'ops.npy', suite2p_data['ops'], allow_pickle=True)
        logger.debug("Copied ops.npy unchanged")
    else: 
        raise FileNotFoundError(f"No ops data found in suite2p_data for video {video_path}")
    # Save the indices of good and bad ROIs for reference
    good_indices = np.where(good_roi_mask)[0]
    bad_indices = np.where(~good_roi_mask)[0]
    
    np.save(filtered_dir / 'good_roi_indices.npy', good_indices)
    np.save(filtered_dir / 'bad_roi_indices.npy', bad_indices)
    
    # Save mapping file for cross-reference
    mapping_df = pd.DataFrame({
        'original_index': np.arange(len(good_roi_mask)),
        'is_good': good_roi_mask,
        'filtered_index': [-1] * len(good_roi_mask)
    })
    mapping_df.loc[good_roi_mask, 'filtered_index'] = np.arange(sum(good_roi_mask))
    mapping_df.to_csv(filtered_dir / 'roi_mapping.csv', index=False)
    
    logger.info(f"Filtered Suite2p saved: {len(good_indices)} good, {len(bad_indices)} bad ROIs")
    
    return filtered_dir

