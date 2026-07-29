"""
I/O utilities for loading experiment data.
"""
from __future__ import annotations
import shutil
from datetime import datetime

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
from joblib import load
import yaml
if TYPE_CHECKING:
    from gcamp_analysis.data_classes import Video

def load_suite2p_data(suite2p_path: Path) -> dict:
    """
    Load the Suite2p inputs required by the analysis pipeline.

    Large numeric arrays are opened read-only with NumPy memory mapping so
    processing a video does not eagerly copy every Suite2p input into RAM.
    Suite2p ``ops.npy`` files may contain very large registration payloads
    (notably ``regPC``); the analysis pipeline only consumes ``fs``, ``Ly``,
    and ``Lx``, so only those fields are retained.

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

    required = ['F.npy', 'iscell.npy', ]
    for file in required:
        if not (suite2p_path / file).exists():
            raise FileNotFoundError(f"Required file {file} not found in {suite2p_path}")

    data['F'] = np.load(suite2p_path / 'F.npy', mmap_mode='r')
    data['iscell'] = np.load(suite2p_path / 'iscell.npy', mmap_mode='r')

    if (suite2p_path / 'Fneu.npy').exists():
        data['Fneu'] = np.load(suite2p_path / 'Fneu.npy', mmap_mode='r')
    else:
        data['Fneu'] = np.zeros_like(data['F'])

    if (suite2p_path / 'spks.npy').exists():
        data['spks'] = np.load(suite2p_path / 'spks.npy', mmap_mode='r')

    if (suite2p_path / 'stat.npy').exists():
        data['stat'] = np.load(suite2p_path / 'stat.npy', allow_pickle=True)

    if (suite2p_path / 'ops.npy').exists():
        full_ops = np.load(
            suite2p_path / 'ops.npy',
            allow_pickle=True,
        ).item()
        data['ops'] = {
            key: full_ops[key]
            for key in ('fs', 'Ly', 'Lx')
            if key in full_ops
        }
        del full_ops
        data['fs'] = data['ops'].get('fs', 15.0)
    else:
        data['ops'] = {}
        data['fs'] = 15.0

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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = input_path.with_suffix(f'.backup_{timestamp}.npy')
    shutil.copy(input_path, backup_path)
    return backup_path
