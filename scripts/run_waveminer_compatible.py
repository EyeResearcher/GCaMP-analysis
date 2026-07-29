"""Run the documented WaveMiner x/y/t flood-fill method on a calcium movie."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gcamp_analysis.waves.raw_movie import load_block_movie
from gcamp_analysis.waves.waveminer_compatible import (
    WaveMinerCompatibleConfig,
    analyze_waveminer_compatible,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tiff", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[2.5, 3.0, 3.5]
    )
    parser.add_argument(
        "--phasic-mode",
        choices=["level", "derivative"],
        default="level",
    )
    parser.add_argument("--pixel-size-um", type=float, default=1.242961138804478)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "block_movie.npy"
    if cache.exists():
        movie = np.load(cache, mmap_mode="r")
    else:
        movie = load_block_movie(args.tiff, block_size=16)
        np.save(cache, movie)
    all_components = []
    for threshold in args.thresholds:
        config = WaveMinerCompatibleConfig(
            pixel_size_um=args.pixel_size_um,
            phasic_z_threshold=threshold,
            phasic_mode=args.phasic_mode,
        )
        components, labels, processed, phasic_z = (
            analyze_waveminer_compatible(movie, config)
        )
        components.to_csv(
            args.output
            / f"components_{args.phasic_mode}_z{threshold:.1f}.csv",
            index=False,
        )
        np.save(
            args.output / f"labels_{args.phasic_mode}_z{threshold:.1f}.npy",
            labels.astype(np.int32),
        )
        all_components.append(components)
        print(
            f"z={threshold:.1f}: {len(components)} components; "
            f"{int(components.get('waveminer_propagating_component', []).sum())} "
            "large propagating components; "
            f"{int(components.get('strict_front_candidate', []).sum())} "
            "strict single-front candidates",
            flush=True,
        )
    np.save(args.output / "processed_dff.npy", processed.astype(np.float32))
    np.save(
        args.output / f"phasic_z_{args.phasic_mode}.npy",
        phasic_z.astype(np.float32),
    )


if __name__ == "__main__":
    main()
