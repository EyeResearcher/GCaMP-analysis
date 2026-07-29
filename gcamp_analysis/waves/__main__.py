"""Command-line entry point for retinal wave analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import WaveAnalysisConfig, analyze_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("wave_analysis_results"))
    parser.add_argument("--days", type=int, nargs="+", default=[10])
    parser.add_argument("--population-nulls", type=int, default=200)
    parser.add_argument("--propagation-nulls", type=int, default=499)
    args = parser.parse_args()
    config = WaveAnalysisConfig(
        population_null_repeats=args.population_nulls,
        propagation_null_repeats=args.propagation_nulls,
    )
    _, summary = analyze_dataset(args.dataset_root, args.output_dir, args.days, config)
    if summary.empty:
        print("No matching recordings found.")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
