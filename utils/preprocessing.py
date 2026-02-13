import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_minmax(f: np.ndarray) -> np.ndarray:
    """Min-max normalize fluorescence traces.

    Parameters
    ----------
    f : np.ndarray
        Raw fluorescence array (n_rois x n_frames or flat).

    Returns
    -------
    scaled_f : np.ndarray
        Min-max scaled array with the same shape as *f*.
    """
    scaler = MinMaxScaler()
    flat_f = f.reshape(-1, 1)
    scaled_flat = scaler.fit_transform(flat_f)
    return scaled_flat.reshape(f.shape)