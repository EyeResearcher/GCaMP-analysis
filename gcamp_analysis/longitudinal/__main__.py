"""Command-line entry point for longitudinal group tracking."""
from __future__ import annotations

import argparse
from pathlib import Path

from .tracking import LongitudinalTracker, discover_recordings


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the longitudinal-tracking CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Register same-region Suite2p masks across days and track the "
            "largest anchor-day functional groups."
        )
    )
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--region", required=True, help="Base region ID, e.g. 1-1")
    parser.add_argument(
        "--treatment",
        help="One treatment folder. Omit to process every treatment containing the region.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strategy", default="combined")
    parser.add_argument("--anchor-day", type=int)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--max-registration-shift", type=int, default=80)
    parser.add_argument("--max-centroid-distance", type=float, default=10.0)
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--min-match-score", type=float, default=0.24)
    return parser


def main() -> None:
    """Parse CLI arguments and run longitudinal tracking per treatment/region."""
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.experiment_root / "metrics" / "longitudinal"
    tracker = LongitudinalTracker(
        experiment_root=args.experiment_root,
        strategy=args.strategy,
        max_registration_shift=args.max_registration_shift,
        max_centroid_distance=args.max_centroid_distance,
        min_iou=args.min_iou,
        min_match_score=args.min_match_score,
    )
    if args.treatment:
        treatments = [args.treatment]
    else:
        treatments = sorted(
            {
                recording.treatment
                for recording in discover_recordings(args.experiment_root)
                if recording.region == args.region
            }
        )
    if not treatments:
        raise SystemExit(f"No recordings found for region {args.region!r}.")
    for treatment in treatments:
        manifest = tracker.run(
            treatment=treatment,
            region=args.region,
            output_dir=output_dir,
            anchor_day=args.anchor_day,
            top_fraction=args.top_fraction,
            top_n=args.top_n,
        )
        print(f"{treatment}/{args.region}")
        for name, path in manifest.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

