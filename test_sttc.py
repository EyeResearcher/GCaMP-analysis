"""Validate vectorized STTC against elephant library."""
import numpy as np
import sys
sys.path.insert(0, r'c:\Users\mzinn1\Desktop\Scripts\GCaMP-analysis')

def compute_sttc_vectorized(spike_times_list, n_frames, dt_frames):
    """Vectorized STTC (copy of pipeline implementation)."""
    n = len(spike_times_list)
    
    # Build binary spike matrix
    spike_matrix = np.zeros((n, n_frames), dtype=np.float32)
    for i, times in enumerate(spike_times_list):
        valid_times = [t for t in times if 0 <= t < n_frames]
        if valid_times:
            spike_matrix[i, valid_times] = 1.0
    
    # Build tiled matrix
    kernel = np.ones(2 * dt_frames + 1, dtype=np.float32)
    tiled_matrix = np.zeros((n, n_frames), dtype=np.float32)
    for i in range(n):
        if np.any(spike_matrix[i]):
            convolved = np.convolve(spike_matrix[i], kernel, mode='same')
            tiled_matrix[i] = (convolved > 0).astype(np.float32)
    
    # T values
    T = tiled_matrix.sum(axis=1) / n_frames
    n_spikes = spike_matrix.sum(axis=1)
    
    # P matrix
    overlap_matrix = spike_matrix @ tiled_matrix.T
    with np.errstate(divide='ignore', invalid='ignore'):
        P = overlap_matrix / n_spikes[:, None]
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    
    # STTC formula
    T_row = T[None, :]
    T_col = T[:, None]
    
    denom_A = 1.0 - P * T_row
    with np.errstate(divide='ignore', invalid='ignore'):
        term_A = (P - T_row) / denom_A
        term_A = np.nan_to_num(term_A, nan=0.0, posinf=1.0, neginf=-1.0)
    
    denom_B = 1.0 - P.T * T_col
    with np.errstate(divide='ignore', invalid='ignore'):
        term_B = (P.T - T_col) / denom_B
        term_B = np.nan_to_num(term_B, nan=0.0, posinf=1.0, neginf=-1.0)
    
    sttc_matrix = 0.5 * (term_A + term_B)
    sttc_matrix = np.clip(sttc_matrix, -1.0, 1.0)
    np.fill_diagonal(sttc_matrix, 1.0)
    
    no_spikes = n_spikes == 0
    sttc_matrix[no_spikes, :] = 0.0
    sttc_matrix[:, no_spikes] = 0.0
    np.fill_diagonal(sttc_matrix, 1.0)
    
    return sttc_matrix


def compute_sttc_elephant(spike_times_list, n_frames, dt_seconds, fs):
    """Reference STTC using elephant library."""
    from elephant.spike_train_correlation import spike_time_tiling_coefficient
    from neo import SpikeTrain
    import quantities as pq
    
    n = len(spike_times_list)
    t_stop = n_frames / fs
    
    # Convert to SpikeTrain objects
    spike_trains = []
    for times in spike_times_list:
        spike_times_sec = np.array(times) / fs
        spike_trains.append(SpikeTrain(spike_times_sec * pq.s, t_stop=t_stop * pq.s))
    
    # Compute pairwise STTC
    sttc_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            try:
                val = float(spike_time_tiling_coefficient(
                    spike_trains[i], spike_trains[j], dt=dt_seconds * pq.s))
            except Exception:
                val = 0.0
            sttc_matrix[i, j] = val
            sttc_matrix[j, i] = val
    
    return sttc_matrix


def test_sttc_accuracy():
    """Compare vectorized vs elephant STTC."""
    np.random.seed(42)
    
    # Test parameters
    n_neurons = 10
    n_frames = 1000
    fs = 30.0
    dt_seconds = 0.033
    dt_frames = int(dt_seconds * fs)
    
    print(f"Test parameters:")
    print(f"  n_neurons: {n_neurons}")
    print(f"  n_frames: {n_frames}")
    print(f"  fs: {fs}")
    print(f"  dt_seconds: {dt_seconds}")
    print(f"  dt_frames: {dt_frames}")
    print()
    
    # Generate random spike times
    spike_times_list = []
    for i in range(n_neurons):
        n_spikes = np.random.randint(5, 30)
        times = np.sort(np.random.choice(n_frames, size=n_spikes, replace=False))
        spike_times_list.append(times.tolist())
        print(f"  Neuron {i}: {len(times)} spikes")
    
    print()
    
    # Compute with both methods
    print("Computing vectorized STTC...")
    sttc_vec = compute_sttc_vectorized(spike_times_list, n_frames, dt_frames)
    
    print("Computing elephant STTC...")
    sttc_eleph = compute_sttc_elephant(spike_times_list, n_frames, dt_seconds, fs)
    
    # Compare
    print()
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    # Check diagonal
    print(f"\nDiagonal (should be 1.0):")
    print(f"  Vectorized: {np.diag(sttc_vec)}")
    print(f"  Elephant:   {np.diag(sttc_eleph)}")
    
    # Check symmetry
    print(f"\nSymmetry check:")
    print(f"  Vectorized symmetric: {np.allclose(sttc_vec, sttc_vec.T)}")
    print(f"  Elephant symmetric:   {np.allclose(sttc_eleph, sttc_eleph.T)}")
    
    # Check range
    print(f"\nValue range:")
    print(f"  Vectorized: [{sttc_vec.min():.4f}, {sttc_vec.max():.4f}]")
    print(f"  Elephant:   [{sttc_eleph.min():.4f}, {sttc_eleph.max():.4f}]")
    
    # Compare values (exclude diagonal)
    mask = ~np.eye(n_neurons, dtype=bool)
    vec_vals = sttc_vec[mask]
    eleph_vals = sttc_eleph[mask]
    
    diff = np.abs(vec_vals - eleph_vals)
    
    print(f"\nOff-diagonal comparison:")
    print(f"  Max absolute difference: {diff.max():.6f}")
    print(f"  Mean absolute difference: {diff.mean():.6f}")
    print(f"  Correlation: {np.corrcoef(vec_vals, eleph_vals)[0,1]:.6f}")
    
    # Show a few specific pairs
    print(f"\nSample pairwise comparisons:")
    print(f"  {'Pair':<10} {'Vectorized':<12} {'Elephant':<12} {'Diff':<10}")
    print(f"  {'-'*44}")
    for i in range(min(5, n_neurons)):
        for j in range(i+1, min(i+3, n_neurons)):
            v = sttc_vec[i, j]
            e = sttc_eleph[i, j]
            d = abs(v - e)
            print(f"  ({i},{j}){' '*5} {v:>10.4f}   {e:>10.4f}   {d:>8.6f}")
    
    # Test edge cases
    print(f"\n" + "=" * 60)
    print("EDGE CASE TESTS")
    print("=" * 60)
    
    # Test 1: Identical spike trains (should be 1.0)
    print("\nTest 1: Identical spike trains")
    identical_spikes = [[10, 20, 30], [10, 20, 30]]
    sttc_ident_vec = compute_sttc_vectorized(identical_spikes, 100, dt_frames)
    sttc_ident_eleph = compute_sttc_elephant(identical_spikes, 100, dt_seconds, fs)
    print(f"  Vectorized [0,1]: {sttc_ident_vec[0,1]:.4f} (expected ~1.0)")
    print(f"  Elephant [0,1]:   {sttc_ident_eleph[0,1]:.4f}")
    
    # Test 2: No overlap (should be negative or ~0)
    print("\nTest 2: No temporal overlap")
    no_overlap = [[10, 20, 30], [60, 70, 80]]
    sttc_no_vec = compute_sttc_vectorized(no_overlap, 100, dt_frames)
    sttc_no_eleph = compute_sttc_elephant(no_overlap, 100, dt_seconds, fs)
    print(f"  Vectorized [0,1]: {sttc_no_vec[0,1]:.4f} (expected < 0)")
    print(f"  Elephant [0,1]:   {sttc_no_eleph[0,1]:.4f}")
    
    # Test 3: One neuron with no spikes
    print("\nTest 3: One neuron with no spikes")
    one_empty = [[10, 20, 30], []]
    sttc_empty_vec = compute_sttc_vectorized(one_empty, 100, dt_frames)
    sttc_empty_eleph = compute_sttc_elephant(one_empty, 100, dt_seconds, fs)
    print(f"  Vectorized [0,1]: {sttc_empty_vec[0,1]:.4f} (expected 0.0)")
    print(f"  Elephant [0,1]:   {sttc_empty_eleph[0,1]:.4f}")
    
    # Test 4: Partial overlap
    print("\nTest 4: Partial overlap")
    partial = [[10, 20, 30, 40], [12, 35, 60, 70]]  # 2 of 4 spikes overlap
    sttc_part_vec = compute_sttc_vectorized(partial, 100, dt_frames)
    sttc_part_eleph = compute_sttc_elephant(partial, 100, dt_seconds, fs)
    print(f"  Vectorized [0,1]: {sttc_part_vec[0,1]:.4f}")
    print(f"  Elephant [0,1]:   {sttc_part_eleph[0,1]:.4f}")
    print(f"  Difference:       {abs(sttc_part_vec[0,1] - sttc_part_eleph[0,1]):.6f}")
    
    # Final verdict
    print(f"\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    tolerance = 0.05  # 5% tolerance
    if diff.max() < tolerance:
        print(f"✓ PASS: Max difference ({diff.max():.4f}) < tolerance ({tolerance})")
    else:
        print(f"✗ FAIL: Max difference ({diff.max():.4f}) >= tolerance ({tolerance})")
        print("  Investigate discrepancies above.")
    
    return sttc_vec, sttc_eleph


if __name__ == "__main__":
    test_sttc_accuracy()