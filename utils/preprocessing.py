import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

def normalize_minmax(f: np.ndarray, output_file: Path) -> np.ndarray:
    """Min-max normalize fluorescence traces and save to file."""
    scaler = MinMaxScaler()
    flat_f = f.reshape(-1, 1)
    scaled_flat = scaler.fit_transform(flat_f)
    scaled_f = scaled_flat.reshape(f.shape)
    np.save(output_file, scaled_f)
    return scaled_f