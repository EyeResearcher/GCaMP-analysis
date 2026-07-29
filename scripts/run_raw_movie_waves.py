"""Run movie-first wave discovery for one TIFF recording."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gcamp_analysis.waves.raw_movie import (
    RawMovieWaveConfig,
    analyze_block_movie,
    load_block_movie,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("tiff", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fs", type=float, default=15.0)
    parser.add_argument("--pixel-size-um", type=float, default=1.242961138804478)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--null-repeats", type=int, default=499)
    parser.add_argument(
        "--range",
        action="append",
        default=[],
        metavar="START:STOP",
        help="Restrict candidates to a half-open frame range; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranges = [
        tuple(int(value) for value in specification.split(":", maxsplit=1))
        for specification in args.range
    ]
    config = RawMovieWaveConfig(
        block_size_pixels=args.block_size,
        propagation_null_repeats=args.null_repeats,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "block_movie.npy"
    if cache.exists():
        movie = np.load(cache, mmap_mode="r")
    else:
        movie = load_block_movie(
            args.tiff,
            block_size=config.block_size_pixels,
        )
        np.save(cache, movie)
    episodes, metrics, temporal = analyze_block_movie(
        movie,
        fs=args.fs,
        pixel_size_um=args.pixel_size_um,
        config=config,
        ranges=ranges or None,
    )
    episodes.to_csv(args.output / "movie_wave_candidates.csv", index=False)
    metrics.to_csv(args.output / "movie_gradient_metrics.csv", index_label="frame")
    np.save(args.output / "movie_temporal_gradient.npy", temporal)
    print(
        f"Tested {len(episodes)} movie-derived candidates; "
        f"{int(episodes.get('movie_significant_wave', []).sum())} significant."
    )
    if not episodes.empty:
        columns = [
            "center_frame",
            "center_seconds",
            "discovery_score",
            "movie_model",
            "movie_propagation_r2",
            "movie_propagation_p",
            "movie_propagation_q",
            "movie_significant_wave",
        ]
        print(
            episodes.sort_values(
                ["movie_significant_wave", "movie_propagation_q"],
                ascending=[False, True],
            )[columns].to_string(index=False)
        )


if __name__ == "__main__":
    main()
