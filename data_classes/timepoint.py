"""Timepoint class for managing data at specific time points."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .video import Video

class Timepoint:
    """Represents a single timepoint in an experiment."""
    
    def __init__(self, path: Path, name: Optional[str] = None, treatment: Optional[str] = None):
        """
        Initialize Timepoint.
        
        Parameters:
            path: Directory containing videos for this timepoint
            name: Timepoint name (e.g., 'Week1')
            treatment: Treatment condition
        """
        self.path = Path(path)
        self.name = name or path.name
        self.treatment = treatment
        self.videos = []
        self.summary_df = None
        
    def add_video(self, video: Video):
        """Add a video to this timepoint."""
        self.videos.append(video)
        
    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame across all videos."""
        summaries = []
        for video in self.videos:
            if hasattr(video, 'summary_df') and video.summary_df is not None:
                summary = video.summary_df.copy()
                summary['video_id'] = video.video_id
                # Add treatment if available
                if hasattr(video, 'treatment'):
                    summary['treatment'] = video.treatment
                summaries.append(summary)
        
        if summaries:
            self.summary_df = pd.concat(summaries, ignore_index=True)
            return self.summary_df
        return pd.DataFrame()
    
    def __repr__(self):
        return f"Timepoint(name={self.name}, videos={len(self.videos)})"