"""
I/O utilities for loading experiment data.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
import logging
from joblib import load
from archive.cascade_utils import load_cascade_model
import yaml
if TYPE_CHECKING:
    from data_classes import Experiment, Timepoint, Video

logger = logging.getLogger(__name__)


def load_npy_file(file_path: Path, allow_pickle: bool = False) -> np.ndarray:
    """
    Load a numpy file safely.
    
    Parameters
    ----------
    file_path : Path
        Path to .npy file
    allow_pickle : bool
        Whether to allow pickle (needed for stat.npy, ops.npy)
        
    Returns
    -------
    np.ndarray
        Loaded array
    """
    try:
        return np.load(file_path, allow_pickle=allow_pickle)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise


def find_suite2p_folders(base_path: Path, pattern: str = "suite2p") -> List[Path]:
    """
    Find all suite2p folders recursively.
    
    Parameters
    ----------
    base_path : Path
        Root directory to search
    pattern : str
        Folder name to search for
        
    Returns
    -------
    List[Path]
        List of paths to suite2p folders
    """
    suite2p_folders = []
    for path in base_path.rglob(pattern):
        if path.is_dir():
            # Check if it has plane0
            plane0 = path / "plane0"
            if plane0.exists() and (plane0 / "F.npy").exists():
                suite2p_folders.append(plane0)
    return suite2p_folders


def parse_experiment_path(plane0_path: Path) -> dict:
    """
    Parse experiment metadata from directory structure.
    Expected: ex337/treatment/timepoint/video/suite2p/plane0/
    
    Parameters
    ----------
    plane0_path : Path
        Path to suite2p/plane0 directory
        
    Returns
    -------
    dict
        Dictionary with keys: experiment, treatment, timepoint, video
    """
    parts = plane0_path.parts
    
    try:
        # Find suite2p in the path
        if 'suite2p' in parts:
            suite2p_idx = parts.index('suite2p')
            # suite2p-1 = video, suite2p-2 = timepoint, suite2p-3 = treatment, suite2p-4 = experiment
            metadata = {
                'video': parts[suite2p_idx - 1] if suite2p_idx >= 1 else 'unknown',
                'timepoint': parts[suite2p_idx - 2] if suite2p_idx >= 2 else 'unknown',
                'treatment': parts[suite2p_idx - 3] if suite2p_idx >= 3 else 'unknown',
                'experiment': parts[suite2p_idx - 4] if suite2p_idx >= 4 else 'unknown',
                'plane0_path': plane0_path,
                'video_path': plane0_path.parent.parent  # Go up from plane0/suite2p/
            }
            return metadata
    except (IndexError, ValueError) as e:
        logger.warning(f"Could not parse path {plane0_path}: {e}")
    
    # Fallback
    return {
        'video': 'unknown',
        'timepoint': 'unknown',
        'treatment': 'unknown',
        'experiment': 'unknown',
        'plane0_path': plane0_path,
        'video_path': plane0_path.parent.parent
    }


def load_experiment_structure(base_path: Path) -> Experiment:
    """
    Load entire experiment structure from directory.
    Expected structure: ex337/treatment/timepoint/video/suite2p/plane0/
    
    Parameters
    ----------
    base_path : Path
        Root experiment directory (e.g., ex337/treatment/)
        
    Returns
    -------
    Experiment
        Populated experiment object with all timepoints and videos
    """
    from data_classes import Experiment, Timepoint, Video
    
    base_path = Path(base_path)
    
    # Parse base path to get experiment and treatment
    parts = base_path.parts
    if len(parts) >= 2:
        experiment_name = parts[-2]  # e.g., 'ex337'
        treatment_name = parts[-1]    # e.g., 'GCaMP6s_EX_Plastic_CoCl'
    else:
        experiment_name = parts[-1] if parts else 'unknown'
        treatment_name = 'unknown'
    
    # Create experiment
    experiment = Experiment(
        base_path=base_path.parent if len(parts) >= 2 else base_path,
        name=experiment_name,
        treatment=treatment_name
    )
    
    # Find all suite2p/plane0 folders
    plane0_folders = find_suite2p_folders(base_path)
    logger.info(f"Found {len(plane0_folders)} videos in {base_path}")
    for plane0_path in plane0_folders:
        print(f"  - {plane0_path}")
    
    # Group by timepoint
    timepoint_dict = {}
    
    for plane0_path in plane0_folders:
        metadata = parse_experiment_path(plane0_path)
        
        timepoint_name = metadata['timepoint']
        video_name = metadata['video']
        
        # Get or create timepoint
        if timepoint_name not in timepoint_dict:
            timepoint_path = plane0_path.parent.parent.parent  # video/suite2p/plane0 -> timepoint
            timepoint = Timepoint(
                path=timepoint_path,
                name=timepoint_name,
                treatment=treatment_name
            )
            timepoint_dict[timepoint_name] = timepoint
            experiment.add_timepoint(timepoint)
        else:
            timepoint = timepoint_dict[timepoint_name]
        
        # Create video
        video = Video(
            path=metadata['video_path'],
            suite2p_path=plane0_path,
            timepoint=timepoint
        )
        timepoint.add_video(video)
        
        logger.debug(f"Added video: {experiment_name}/{treatment_name}/{timepoint_name}/{video_name}")
    
    logger.info(f"Loaded experiment '{experiment.name}' with {len(experiment.timepoints)} timepoints")
    
    return experiment


class SummaryFiles:
    """Helper class for managing output file paths."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize summary file manager.
        
        Parameters
        ----------
        output_dir : Path
            Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_video_summary_path(self, video_name: str) -> Path:
        """Get path for video summary Excel file."""
        return self.output_dir / f"{video_name}_summary.xlsx"
    
    def get_timepoint_summary_path(self, timepoint_name: str) -> Path:
        """Get path for timepoint summary Excel file."""
        return self.output_dir / f"{timepoint_name}_summary.xlsx"
    
    def get_experiment_summary_path(self, experiment_name: str) -> Path:
        """Get path for experiment summary Excel file."""
        return self.output_dir / f"{experiment_name}_summary.xlsx"
    
    def get_filtered_suite2p_dir(self, video_name: str) -> Path:
        """Get path for filtered suite2p output."""
        filtered_dir = self.output_dir / video_name / "filtered_suite2p" / "plane0"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        return filtered_dir
    
def load_config(config_path: Path = Path("config.yaml")) -> Dict:
    """Load configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(models_config: Dict, which : str) -> Dict:
    """Load all required models and normalize wrappers to sklearn estimators."""
    model 
    path = Path(models_config.get(f'{which}_model_path', ''))
    if path.exists():
        model = load(path)
        return model
    else:
        raise FileNotFoundError(f"{which} model cannot be found at {path}.")
    
    