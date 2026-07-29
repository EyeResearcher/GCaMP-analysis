"""Run movie-first wave discovery across selected recording days."""
from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from gcamp_analysis.waves.raw_movie import (
    RawMovieWaveConfig,
    analyze_block_movie,
    load_block_movie,
)


def pixel_size(recording: Path) -> float:
    xml_path = next(recording.glob("*.ome.xml"))
    root = ElementTree.parse(xml_path).getroot()
    pixels = next(
        element for element in root.iter() if element.tag.endswith("Pixels")
    )
    return float(pixels.attrib["PhysicalSizeX"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, nargs="+", default=[10])
    parser.add_argument("--null-repeats", type=int, default=499)
    args = parser.parse_args()
    config = RawMovieWaveConfig(
        propagation_null_repeats=args.null_repeats
    )
    recordings = []
    for treatment in ("BP", "IOBP"):
        for day in args.days:
            for recording in sorted(
                (args.root / treatment).glob(f"*_Day{day}")
            ):
                tiffs = [
                    path
                    for path in recording.glob("*.tif*")
                    if "_snap" not in path.name
                ]
                if tiffs and (
                    recording / "suite2p" / "plane0" / "F.npy"
                ).exists():
                    recordings.append((treatment, recording, tiffs[0]))
    for treatment, recording, tiff in recordings:
        output = args.output / f"{treatment}_{recording.name}"
        output.mkdir(parents=True, exist_ok=True)
        cache = output / "block_movie.npy"
        if cache.exists():
            movie = np.load(cache, mmap_mode="r")
        else:
            movie = load_block_movie(
                tiff, block_size=config.block_size_pixels
            )
            np.save(cache, movie)
        episodes, metrics, temporal = analyze_block_movie(
            movie,
            fs=15.0,
            pixel_size_um=pixel_size(recording),
            config=config,
        )
        episodes.insert(0, "treatment", treatment)
        episodes.insert(1, "recording", recording.name)
        episodes.insert(2, "recording_path", str(recording))
        episodes.insert(3, "pixel_size_um", pixel_size(recording))
        episodes.to_csv(output / "movie_wave_candidates.csv", index=False)
        metrics.to_csv(output / "movie_gradient_metrics.csv", index_label="frame")
        np.save(output / "movie_temporal_gradient.npy", temporal)
        count = int(episodes["movie_significant_wave"].sum())
        print(
            f"{treatment}/{recording.name}: "
            f"{len(episodes)} candidates, {count} movie-positive",
            flush=True,
        )


if __name__ == "__main__":
    main()
