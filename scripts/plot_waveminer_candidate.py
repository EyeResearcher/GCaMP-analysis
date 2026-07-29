"""Plot a threshold-stable WaveMiner-compatible candidate."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def _arrival_map(labels: np.ndarray, component_id: int) -> np.ndarray:
    member = labels == component_id
    result = np.full(member.shape[1:], np.nan)
    spatial = member.any(axis=0)
    result[spatial] = np.argmax(member, axis=0)[spatial]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--high-component", type=int, default=33)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "candidate_frame270_validation.png"

    thresholds = [2.0, 2.5, 3.0]
    tables = {
        z: pd.read_csv(
            args.results / f"components_level_z{z:.1f}.csv"
        ).set_index("component_id")
        for z in thresholds
    }
    labels = {
        z: np.load(args.results / f"labels_level_z{z:.1f}.npy")
        for z in thresholds
    }
    high_mask = labels[3.0] == args.high_component
    component_ids = {}
    for z in thresholds:
        ids = labels[z][high_mask]
        ids = ids[ids > 0]
        component_ids[z] = int(np.argmax(np.bincount(ids)))

    middle_row = tables[2.5].loc[component_ids[2.5]]
    start, end = int(middle_row.start_frame), int(middle_row.end_frame)
    processed = np.load(args.results / "processed_dff.npy")
    middle_mask = labels[2.5] == component_ids[2.5]
    snapshots = np.linspace(start, end, 6).round().astype(int)

    fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.05])
    vmax = float(np.nanpercentile(processed[start : end + 1], 99.5))
    vmin = float(np.nanpercentile(processed[start : end + 1], 5))
    for column, frame in enumerate(snapshots):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(processed[frame], cmap="gray", vmin=vmin, vmax=vmax)
        ax.contour(
            middle_mask[frame],
            levels=[0.5],
            colors=["#ff3b30"],
            linewidths=1.2,
        )
        ax.set_title(f"frame {frame}\n{frame / 15:.2f} s")
        ax.set_xticks([])
        ax.set_yticks([])
        if column == 0:
            ax.set_ylabel("dF/F + z=2.5 component")

    arrival_maps = {
        z: _arrival_map(labels[z], component_ids[z]) for z in thresholds
    }
    all_arrivals = np.concatenate(
        [m[np.isfinite(m)] for m in arrival_maps.values()]
    )
    norm = Normalize(vmin=float(all_arrivals.min()), vmax=float(all_arrivals.max()))
    for column, z in enumerate(thresholds):
        ax = fig.add_subplot(grid[1, column * 2 : column * 2 + 2])
        image = ax.imshow(arrival_maps[z], cmap="turbo", norm=norm)
        row = tables[z].loc[component_ids[z]]
        speed = row["leading_edge_speed_um_s"]
        speed_text = "not estimable" if np.isnan(speed) else f"{speed:.0f} µm/s"
        ax.set_title(
            f"z ≥ {z:.1f}: component {component_ids[z]}\n"
            f"{int(row.start_frame)}–{int(row.end_frame)}, "
            f"{row.area_um2 / 1e6:.3f} mm², {speed_text}"
        )
        ax.set_xlabel("movie block x (19.9 µm/block)")
        if column == 0:
            ax.set_ylabel("movie block y")
        else:
            ax.set_yticklabels([])
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("first-active frame")

    fig.suptitle(
        "1-1 Day10 candidate near frame 270: threshold persistence and motion",
        fontsize=16,
    )
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
