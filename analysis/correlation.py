
import numpy as np
import quantities as pq
from elephant.spike_train_correlation import spike_time_tiling_coefficient as sttc_elephant
def sttc_binary(spikes_a, spikes_b, fs, dt_ms=150):
    """
    Compute the Spike Time Tiling Coefficient (STTC) for two binarized spike trains.

    Parameters
    ----------
    spikes_a, spikes_b : np.ndarray
        1D arrays of 0s and 1s (same length), where 1 indicates a spike.
    fs : float
        Sampling frequency in Hz.
    dt_ms : float
        Time window in milliseconds (default 150 ms).

    Returns
    -------
    sttc : float
        The STTC value between the two spike trains.
    """
    assert spikes_a.shape == spikes_b.shape
    n = len(spikes_a)
    dt = int(round(dt_ms / 1000 * fs))  # window in samples

    # Find spike indices
    idx_a = np.flatnonzero(spikes_a)
    idx_b = np.flatnonzero(spikes_b)
    n_a = len(idx_a)
    n_b = len(idx_b)
    if n_a == 0 or n_b == 0:
        return np.nan

    # TA: Proportion of total recording within dt of any spike in A
    ta_mask = np.zeros(n, dtype=bool)
    for i in idx_a:
        ta_mask[max(0, i-dt):min(n, i+dt+1)] = True
    ta = ta_mask.sum() / n

    # TB: Proportion of total recording within dt of any spike in B
    tb_mask = np.zeros(n, dtype=bool)
    for i in idx_b:
        tb_mask[max(0, i-dt):min(n, i+dt+1)] = True
    tb = tb_mask.sum() / n

    # PA: Proportion of spikes in A within dt of any spike in B
    pa = np.any(np.abs(idx_a[:, None] - idx_b) <= dt, axis=1).sum() / n_a

    # PB: Proportion of spikes in B within dt of any spike in A
    pb = np.any(np.abs(idx_b[:, None] - idx_a) <= dt, axis=1).sum() / n_b

    sttc = 0.5 * ((pa - ta) / (1 - pa * ta) + (pb - tb) / (1 - pb * tb))
    return sttc

def compute_sttc_matrix(spike_trains, ops):
    """
    Compute the Spike Time Tiling Coefficient (STTC) matrix for a list of spike index arrays.

    Args:
        all_spike_idx (list of array-like): Each element is an array of spike frame indices for one neuron.
        ops (dict): Suite2p ops dictionary containing 'fs' (sampling rate) and 'nframes'.

    Returns:
        np.ndarray: Symmetric STTC matrix of shape (n_neurons, n_neurons).
    """
    fs = ops['fs']
    nframes = ops['nframes']
    n = len(spike_trains)
    sttc_matrix = np.zeros((n, n))

    # Compute STTC for each pair
    for i in range(n):
        for j in range(i, n):
            sttc = sttc_elephant(
                spike_trains[i], spike_trains[j], dt = 150 * pq.ms,
            )
            sttc_matrix[i, j] = sttc
            sttc_matrix[j, i] = sttc

    return sttc_matrix