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

def load_suite2p_data(plane0_path: Path) -> dict:
    """
    Load Suite2p data from a given plane0 directory.
    
    Parameters
    ----------
    plane0_path : Path
        Path to suite2p/plane0 directory
        
    Returns
    -------
    dict
        Dictionary containing Suite2p data arrays
    """
    data = {}
    try:
        data['F'] = np.load(plane0_path / "F.npy")
        data['Fneu'] = np.load(plane0_path / "Fneu.npy")
        data['spks'] = np.load(plane0_path / "spks.npy")
        data['ops'] = np.load(plane0_path / "ops.npy", allow_pickle=True).item()
        data['stat'] = np.load(plane0_path / "stat.npy", allow_pickle=True)
    except Exception as e:
        logger.error(f"Error loading Suite2p data from {plane0_path}: {e}")
        raise
    return data
    
def load_config(config_path: Path = Path("config.yaml")) -> Dict:
    """Load configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(models_config: Dict, which : str) -> Dict:
    """Load all required models and normalize wrappers to sklearn estimators.""" 
    path = Path(models_config.get(f'{which}_model_path', ''))
    if path.exists():
        model = load(path)
        return model
    else:
        raise FileNotFoundError(f"{which} model cannot be found at {path}.")
    
def _safe_sheet_name(name: str) -> str:
    """
    Excel sheet name rules:
      - max 31 chars
      - cannot contain: : \ / ? * [ ]
    """
    bad = [":", "\\", "/", "?", "*", "[", "]"]
    for ch in bad:
        name = name.replace(ch, "_")
    return name[:31] if len(name) > 31 else name


def save_node_level_comparisons(
    sibling_tables: Dict[Path, pd.DataFrame],
    *,
    output_subdir: str = "metrics",
    filename: str = "sibling_comparisons.xlsx",
) -> None:
    """
    Writes each node's sibling comparison table into that node's directory:
        <node_path>/<output_subdir>/<filename>

    Example:
        Experiment337/metrics/sibling_comparisons.xlsx
        Experiment337/GABA/metrics/sibling_comparisons.xlsx
        Experiment337/GABA/Week1/metrics/sibling_comparisons.xlsx
        ...
    """
    for node_path, df in sibling_tables.items():
        if df is None or df.empty:
            continue

        node_path = Path(node_path)
        out_dir = node_path / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / filename

        # One sheet is enough because df already compares that node's children.
        # But we’ll name the sheet after the node for clarity.
        sheet = _safe_sheet_name(node_path.name or "node")

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet)
