"""Run neighbor-graph cross-correlation on saved candidate wave episodes."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gcamp_analysis.waves.neighbor_xcorr import (
    NeighborXcorrConfig,
    analyze_episode_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--null-repeats", type=int, default=499)
    parser.add_argument("--half-window-frames", type=int, default=18)
    parser.add_argument("--max-lag-frames", type=int, default=8)
    args = parser.parse_args()
    episodes = pd.read_csv(args.episodes_csv)
    config = NeighborXcorrConfig(
        propagation_null_repeats=args.null_repeats,
        half_window_frames=args.half_window_frames,
        max_lag_frames=args.max_lag_frames,
    )
    output = analyze_episode_table(episodes, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    columns = [
        "day",
        "treatment",
        "recording",
        "center_seconds",
        "significant_wave",
        "movie_validated",
        "xcorr_status",
        "xcorr_n_component_nodes",
        "xcorr_peak_correlation_median",
        "xcorr_edge_consistency_r2",
        "xcorr_model",
        "xcorr_propagation_r2",
        "xcorr_propagation_q",
        "xcorr_significant_wave",
    ]
    print(output[columns].to_string(index=False))


if __name__ == "__main__":
    main()
