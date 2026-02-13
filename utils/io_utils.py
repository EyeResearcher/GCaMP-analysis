"""
I/O utilities for loading experiment data.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
import logging
from joblib import load
import yaml
if TYPE_CHECKING:
    from data_classes import Video

logger = logging.getLogger(__name__)

def load_suite2p_data(suite2p_path: Path) -> dict:
    """
    Load all Suite2p output files from a plane0 directory.

    Parameters
    ----------
    suite2p_path : Path
        Path to suite2p/plane0 directory

    Returns
    -------
    dict
        Dictionary containing Suite2p data arrays.
        Keys: F, iscell (required); Fneu, spks, stat, ops, fs (optional).
    """
    suite2p_path = Path(suite2p_path)
    data = {}

    # Required files
    required = ['F.npy', 'iscell.npy']
    for file in required:
        if not (suite2p_path / file).exists():
            raise FileNotFoundError(f"Required file {file} not found in {suite2p_path}")

    # Load required arrays
    data['F'] = np.load(suite2p_path / 'F.npy')
    data['iscell'] = np.load(suite2p_path / 'iscell.npy')

    # Optional files
    if (suite2p_path / 'Fneu.npy').exists():
        data['Fneu'] = np.load(suite2p_path / 'Fneu.npy')
    else:
        logger.warning("Fneu.npy not found, using zeros")
        data['Fneu'] = np.zeros_like(data['F'])

    if (suite2p_path / 'spks.npy').exists():
        data['spks'] = np.load(suite2p_path / 'spks.npy')

    if (suite2p_path / 'stat.npy').exists():
        data['stat'] = np.load(suite2p_path / 'stat.npy', allow_pickle=True)

    if (suite2p_path / 'ops.npy').exists():
        data['ops'] = np.load(suite2p_path / 'ops.npy', allow_pickle=True).item()
        data['fs'] = data['ops'].get('fs', 30.0)
    else:
        data['fs'] = 30.0

    logger.info(f"Loaded Suite2p data: {data['F'].shape[0]} ROIs, {data['F'].shape[1]} frames")

    return data
    
def load_config(config_path: Path = Path("config.yaml")) -> Dict:
    """Load configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(models_config: Dict, which: str) -> tuple:
    """Load a model and its JSON config sidecar.

    Parameters
    ----------
    models_config : dict
        The ``models`` section of the pipeline YAML.  Expected keys:
        ``<which>_model_path`` and (optionally) ``<which>_config_path``.
    which : str
        ``"roi"`` or ``"spike"``.

    Returns
    -------
    tuple[Any, dict | None]
        ``(sklearn_model, config_dict)`` where *config_dict* is ``None``
        when no config path is given or the file does not exist.
    """
    import json

    model_path = Path(models_config.get(f"{which}_model_path", ""))
    if not model_path.exists():
        raise FileNotFoundError(f"{which} model cannot be found at {model_path}.")
    model = load(model_path)

    config_path_str = models_config.get(f"{which}_config_path")
    model_cfg: Optional[Dict] = None
    if config_path_str:
        config_path = Path(config_path_str)
        if config_path.exists():
            with open(config_path, "r") as fh:
                model_cfg = json.load(fh)
        else:
            logger.warning("%s config not found at %s — continuing without it.", which, config_path)

    return model, model_cfg


def create_backup(input_path: Path) -> Path:
    """
    Create timestamped backup of a file.

    Parameters
    ----------
    input_path : Path
        Path to file to backup

    Returns
    -------
    backup_path : Path
        Path to created backup
    """
    import shutil
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = input_path.with_suffix(f'.backup_{timestamp}.npy')
    shutil.copy(input_path, backup_path)
    return backup_path
