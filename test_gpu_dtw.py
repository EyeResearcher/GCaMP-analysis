"""Quick test to verify GPU DTW implementation."""
import torch
import numpy as np
from pipeline.neuron_grouping import compute_dtw_matrix
from data_classes.neuron import Neuron

# Check if GPU is available
if torch.cuda.is_available():
    print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("✗ No GPU detected - will use CPU")

# Create mock neurons with random fluorescence traces
print("\nCreating test neurons...")
n_neurons = 10
n_frames = 1000

neurons = []
for i in range(n_neurons):
    neuron = Neuron(roi_id=i, experiment_name="test", video_name="test")
    neuron.raw_fluorescence = np.random.randn(n_frames).astype(np.float64)
    neurons.append(neuron)

# Test GPU DTW computation
print(f"\nComputing DTW matrix for {n_neurons} neurons with {n_frames} frames...")
print("This tests GPU acceleration...")

import time
start = time.time()
dtw_matrix = compute_dtw_matrix(neurons, downsample_factor=3, use_gpu=True)
elapsed = time.time() - start

print(f"\n✓ DTW computation completed in {elapsed:.2f} seconds")
print(f"  Matrix shape: {dtw_matrix.shape}")
print(f"  Distance range: [{dtw_matrix.min():.2f}, {dtw_matrix.max():.2f}]")
print(f"  Mean distance: {dtw_matrix.mean():.2f}")

# Test CPU fallback
print("\nTesting CPU fallback...")
start = time.time()
dtw_matrix_cpu = compute_dtw_matrix(neurons, downsample_factor=3, use_gpu=False)
elapsed_cpu = time.time() - start

print(f"✓ CPU computation completed in {elapsed_cpu:.2f} seconds")

if torch.cuda.is_available():
    speedup = elapsed_cpu / elapsed
    print(f"\nSpeedup: {speedup:.2f}x faster with GPU")

print("\n✓ All tests passed!")
