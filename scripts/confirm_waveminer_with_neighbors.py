"""Confirm WaveMiner-compatible components with the Suite2p neighbor graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gcamp_analysis.waves.neighbor_xcorr import (
    NeighborXcorrConfig,
    analyze_episode_table,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument("--component", type=int, default=120)
    parser.add_argument("--pixel-size-um", type=float, default=1.242961138804478)
    parser.add_argument("--block-size-pixels", type=int, default=16)
    parser.add_argument("--null-repeats", type=int, default=999)
    parser.add_argument(
        "--participant-episodes",
        type=Path,
        help="Optional independently detected ROI episode table; restrict the "
        "graph to its participant ROI indices.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    labels = np.load(
        args.results / f"labels_level_z{args.threshold:.1f}.npy",
        mmap_mode="r",
    )
    table = pd.read_csv(
        args.results / f"components_level_z{args.threshold:.1f}.csv"
    ).set_index("component_id")
    component = table.loc[args.component]
    footprint = np.any(labels == args.component, axis=0)

    suite2p = args.recording / "suite2p" / "plane0"
    stat = np.load(suite2p / "stat.npy", allow_pickle=True)
    coords = np.asarray(
        [[float(item["med"][1]), float(item["med"][0])] for item in stat]
    )
    block_x = np.clip(
        (coords[:, 0] // args.block_size_pixels).astype(int),
        0,
        footprint.shape[1] - 1,
    )
    block_y = np.clip(
        (coords[:, 1] // args.block_size_pixels).astype(int),
        0,
        footprint.shape[0] - 1,
    )
    roi_indices = np.flatnonzero(footprint[block_y, block_x])
    participant_source_frame = np.nan
    if args.participant_episodes is not None:
        episodes = pd.read_csv(args.participant_episodes)
        nearest_index = (
            episodes["center_frame"] - component["center_frame"]
        ).abs().idxmin()
        nearest = episodes.loc[nearest_index]
        participants = np.asarray(
            json.loads(nearest["roi_indices"]), dtype=int
        )
        roi_indices = np.intersect1d(roi_indices, participants)
        participant_source_frame = int(nearest["center_frame"])
    episode = pd.DataFrame(
        [
            {
                "day": 10,
                "treatment": "in-vitro stem-cell culture",
                "recording": args.recording.name,
                "recording_path": str(args.recording),
                "pixel_size_um": args.pixel_size_um,
                "center_frame": int(component["center_frame"]),
                "center_seconds": float(component["center_frame"] / 15.0),
                "roi_indices": json.dumps(roi_indices.tolist()),
                "waveminer_threshold": args.threshold,
                "waveminer_component": args.component,
                "waveminer_start_frame": int(component["start_frame"]),
                "waveminer_end_frame": int(component["end_frame"]),
                "waveminer_footprint_roi_count": int(roi_indices.size),
                "participant_source_frame": participant_source_frame,
            }
        ]
    )
    result = analyze_episode_table(
        episode,
        config=NeighborXcorrConfig(
            half_window_frames=20,
            max_lag_frames=8,
            propagation_null_repeats=args.null_repeats,
        ),
    )
    output = args.output or args.results / "candidate_frame270_neighbor_xcorr.csv"
    result.to_csv(output, index=False)
    columns = [
        "center_frame",
        "waveminer_start_frame",
        "waveminer_end_frame",
        "waveminer_footprint_roi_count",
        "xcorr_status",
        "xcorr_n_component_nodes",
        "xcorr_peak_correlation_median",
        "xcorr_edge_consistency_r2",
        "xcorr_model",
        "xcorr_propagation_r2",
        "xcorr_speed_um_s",
        "xcorr_lag_permutation_p",
        "xcorr_propagation_q",
        "xcorr_significant_wave",
    ]
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
