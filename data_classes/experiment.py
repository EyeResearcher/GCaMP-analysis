"""Experiment class for managing full experimental datasets."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .timepoint import Timepoint

class Experiment:
    """Represents a complete experiment with multiple timepoints."""
    
    def __init__(self, base_path: Path, name: Optional[str] = None, treatment: Optional[str] = None):
        """
        Initialize Experiment.
        
        Parameters:
            base_path: Root directory of experiment (e.g., ex337)
            name: Experiment name (default: uses base_path name)
            treatment: Treatment condition (e.g., 'GCaMP6s_EX_Plastic_CoCl')
        """
        self.base_path = Path(base_path)
        self.treatment = treatment
        
        # Create experiment name with treatment if provided
        if name:
            self.name = name
        elif treatment:
            self.name = f"{base_path.name}_{treatment}"
        else:
            self.name = base_path.name
            
        self.timepoints = []
        
    def add_timepoint(self, timepoint: Timepoint):
        """Add a timepoint to the experiment."""
        self.timepoints.append(timepoint)
        
    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame across all timepoints."""
        summaries = []
        for tp in self.timepoints:
            if tp.summary_df is not None:
                summary = tp.summary_df.copy()
                summary['timepoint'] = tp.name
                summaries.append(summary)
        
        if summaries:
            return pd.concat(summaries, ignore_index=True)
        return pd.DataFrame()
    
    def __repr__(self):
        return f"Experiment(name={self.name}, timepoints={len(self.timepoints)})"