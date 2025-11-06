"""Video class for managing individual recording sessions."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .timepoint import Timepoint

class Video:
    """Represents a single video recording session."""
    
    def __init__(self, path: Path, timepoint: Optional[Timepoint] = None):
        """
        Initialize Video.
        
        Parameters:
            path: Directory containing Suite2p output
            timepoint: Parent timepoint object
        """
        self.path = Path(path)
        self.video_id = path.name
        self.timepoint = timepoint
        
        # Parse metadata from folder name if possible
        self._parse_metadata()
        
        # Will be populated by pipeline
        self.neurons = []
        self.sttc_groups = []
        self.dtw_groups = []
        self.summary_df = None
        
    def _parse_metadata(self):
        """
        Parse metadata from directory structure.
        Expected: ex337/treatment/timepoint/video/suite2p/plane0/
        """
        # Get parent directories
        parts = self.path.parts
        
        # video is current folder name
        self.video_id = self.path.name
        
        # Go up: video -> timepoint -> treatment -> experiment
        if len(parts) >= 4:
            # parts[-1] = plane0, parts[-2] = suite2p, parts[-3] = video
            # parts[-4] = timepoint, parts[-5] = treatment, parts[-6] = experiment
            try:
                # If path includes suite2p/plane0, adjust indices
                if 'suite2p' in parts:
                    suite2p_idx = parts.index('suite2p')
                    # suite2p-1 = video, suite2p-2 = timepoint, suite2p-3 = treatment
                    if suite2p_idx >= 3:
                        self.timepoint_name = parts[suite2p_idx - 2]
                        self.treatment = parts[suite2p_idx - 3]
                        if suite2p_idx >= 4:
                            self.experiment_name = parts[suite2p_idx - 4]
                        else:
                            self.experiment_name = 'unknown'
                    else:
                        self.timepoint_name = 'unknown'
                        self.treatment = 'unknown'
                        self.experiment_name = 'unknown'
                else:
                    # Direct video path without suite2p
                    self.timepoint_name = parts[-2] if len(parts) >= 2 else 'unknown'
                    self.treatment = parts[-3] if len(parts) >= 3 else 'unknown'
                    self.experiment_name = parts[-4] if len(parts) >= 4 else 'unknown'
            except (IndexError, ValueError):
                self.timepoint_name = 'unknown'
                self.treatment = 'unknown'
                self.experiment_name = 'unknown'
        else:
            self.timepoint_name = 'unknown'
            self.treatment = 'unknown'
            self.experiment_name = 'unknown'
        
    def get_group_summary(self) -> pd.DataFrame:
        """Summarize grouping results."""
        rows = []
        
        # STTC groups
        for i, group in enumerate(self.sttc_groups):
            for neuron in group:
                rows.append({
                    'neuron_id': neuron.row_index,
                    'group_method': 'sttc',
                    'group_id': i,
                    'group_size': len(group)
                })
        
        # DTW groups
        for i, group in enumerate(self.dtw_groups):
            for neuron in group:
                rows.append({
                    'neuron_id': neuron.row_index,
                    'group_method': 'dtw',
                    'group_id': i,
                    'group_size': len(group)
                })
        
        return pd.DataFrame(rows)
    
    def __repr__(self):
        return f"Video(id={self.video_id}, neurons={len(self.neurons)})"