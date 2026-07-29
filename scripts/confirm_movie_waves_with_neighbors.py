"""Confirm movie-derived wave candidates using local ROI cross-correlations."""
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
from gcamp_analysis.waves.raw_movie import RawMovieWaveConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("temporal_gradient", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixel-size-um", type=float, default=1.242961138804478)
    parser.add_argument("--null-repeats", type=int, default=499)
    parser.add_argument(
        "--include-borderline",
        action="store_true",
        help="Also include p<=0.01, R2>=0.15 movie candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates)
    selected = candidates["movie_significant_wave"].astype(bool)
    if args.include_borderline:
        selected |= (
            (candidates["movie_propagation_p"] <= 0.01)
            & (candidates["movie_propagation_r2"] >= 0.15)
        )
    candidates = candidates[selected].copy()
    temporal = np.load(args.temporal_gradient, mmap_mode="r")
    suite2p = args.recording / "suite2p" / "plane0"
    stat = np.load(suite2p / "stat.npy", allow_pickle=True)
    coordinates = np.asarray(
        [
            [float(item["med"][1]), float(item["med"][0])]
            for item in stat
        ]
    )
    movie_config = RawMovieWaveConfig()
    block_columns = np.clip(
        (coordinates[:, 0] // movie_config.block_size_pixels).astype(int),
        0,
        temporal.shape[2] - 1,
    )
    block_rows = np.clip(
        (coordinates[:, 1] // movie_config.block_size_pixels).astype(int),
        0,
        temporal.shape[1] - 1,
    )
    rows = []
    for _, candidate in candidates.iterrows():
        center = int(candidate["center_frame"])
        start = max(0, center - movie_config.fit_half_window_frames)
        stop = min(
            temporal.shape[0],
            center + movie_config.fit_half_window_frames + 1,
        )
        window = np.maximum(np.asarray(temporal[start:stop]), 0.0)
        amplitude = window.max(axis=0)
        relative_arrival = np.argmax(window, axis=0)
        threshold = float(
            np.quantile(amplitude, movie_config.active_block_quantile)
        )
        active = amplitude >= threshold
        active &= relative_arrival > 0
        active &= relative_arrival < window.shape[0] - 1
        selected = np.flatnonzero(active[block_rows, block_columns])
        rows.append(
            {
                **candidate.to_dict(),
                "day": 10,
                "treatment": "BP",
                "recording": args.recording.name,
                "recording_path": str(args.recording),
                "pixel_size_um": args.pixel_size_um,
                "roi_indices": json.dumps(selected.tolist()),
                "movie_footprint_roi_count": int(selected.size),
            }
        )
    table = pd.DataFrame(rows)
    confirmed = analyze_episode_table(
        table,
        config=NeighborXcorrConfig(
            propagation_null_repeats=args.null_repeats
        ),
    )
    confirmed["movie_xcorr_same_model"] = (
        confirmed["movie_model"] == confirmed["xcorr_model"]
    )
    direction_difference = np.abs(
        (
            confirmed["movie_direction_degrees"]
            - confirmed["xcorr_direction_degrees"]
            + 180.0
        )
        % 360.0
        - 180.0
    )
    confirmed["movie_xcorr_angle_difference_degrees"] = direction_difference
    confirmed.to_csv(args.output, index=False)
    columns = [
        "center_frame",
        "movie_model",
        "movie_propagation_r2",
        "movie_propagation_q",
        "movie_footprint_roi_count",
        "xcorr_model",
        "xcorr_propagation_r2",
        "xcorr_propagation_p",
        "xcorr_propagation_q",
        "movie_xcorr_same_model",
        "movie_xcorr_angle_difference_degrees",
        "xcorr_significant_wave",
    ]
    print(confirmed[columns].to_string(index=False))


if __name__ == "__main__":
    main()
