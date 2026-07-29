"""Data-adaptive null tests for a WaveMiner-compatible component.

The tests intentionally avoid importing ex-vivo retinal-wave priors:

* spatial permutation moves each complete block trace to a random location,
  preserving the population time course and every trace's autocorrelation;
* circular shifts preserve each block's autocorrelation and activity rate but
  destroy its timing relative to neighboring blocks.

For each randomized movie, the maximum 3-D component voxel count is retained,
so the resulting p value controls selection over the whole recording for this
specific size statistic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, generate_binary_structure, label


def _remove_isolated(active: np.ndarray) -> np.ndarray:
    kernel = np.ones((1, 3, 3), dtype=np.int8)
    kernel[0, 1, 1] = 0
    neighbor_count = convolve(
        active.astype(np.int8), kernel, mode="constant", cval=0
    )
    return active & (neighbor_count >= 1)


def _maximum_component_size(active: np.ndarray) -> int:
    labels, count = label(active, structure=generate_binary_structure(3, 3))
    if count == 0:
        return 0
    sizes = np.bincount(labels.ravel())
    return int(sizes[1:].max(initial=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument("--component", type=int, default=120)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    z = np.load(args.results / "phasic_z_level.npy", mmap_mode="r")
    observed_labels = np.load(
        args.results / f"labels_level_z{args.threshold:.1f}.npy",
        mmap_mode="r",
    )
    observed_size = int(np.count_nonzero(observed_labels == args.component))
    active = np.asarray(z >= args.threshold)
    time_count, height, width = active.shape
    flat = active.reshape(time_count, height * width)
    rng = np.random.default_rng(args.seed)
    rows = []

    for repeat in range(args.permutations):
        spatial_order = rng.permutation(flat.shape[1])
        spatial_null = _remove_isolated(
            flat[:, spatial_order].reshape(active.shape)
        )
        spatial_max = _maximum_component_size(spatial_null)

        shifts = rng.integers(1, time_count, size=flat.shape[1])
        shifted = np.empty_like(flat)
        for pixel, shift in enumerate(shifts):
            shifted[:, pixel] = np.roll(flat[:, pixel], int(shift))
        temporal_null = _remove_isolated(shifted.reshape(active.shape))
        temporal_max = _maximum_component_size(temporal_null)
        rows.append(
            {
                "permutation": repeat + 1,
                "spatial_permutation_max_voxels": spatial_max,
                "circular_shift_max_voxels": temporal_max,
            }
        )
        if (repeat + 1) % 20 == 0:
            print(f"{repeat + 1}/{args.permutations}", flush=True)

    nulls = pd.DataFrame(rows)
    spatial_p = (
        1
        + int(
            np.count_nonzero(
                nulls["spatial_permutation_max_voxels"] >= observed_size
            )
        )
    ) / (args.permutations + 1)
    temporal_p = (
        1
        + int(
            np.count_nonzero(
                nulls["circular_shift_max_voxels"] >= observed_size
            )
        )
    ) / (args.permutations + 1)
    nulls.to_csv(args.results / "invitro_null_max_component_sizes.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "threshold": args.threshold,
                "component": args.component,
                "observed_voxels": observed_size,
                "permutations": args.permutations,
                "spatial_permutation_max_p_fwer": spatial_p,
                "circular_shift_max_p_fwer": temporal_p,
                "spatial_null_95pct": np.percentile(
                    nulls["spatial_permutation_max_voxels"], 95
                ),
                "circular_null_95pct": np.percentile(
                    nulls["circular_shift_max_voxels"], 95
                ),
            }
        ]
    )
    summary.to_csv(args.results / "invitro_null_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
