"""Recording-only command-line interface."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze independent GCaMP recordings")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("recordings_root", type=Path)
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--sensor")
    analyze.add_argument("--dry-run", action="store_true")
    analyze.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    from gcamp_analysis.api import analyze_recordings
    records = analyze_recordings(args.recordings_root, args.config, sensor=args.sensor,
                                 dry_run=args.dry_run, verbose=not args.quiet)
    if not args.quiet:
        print(f"Analyzed {len(records)} recording(s)." if not args.dry_run else
              f"Dry run analyzed {len(records)} recording(s); no outputs written.")


if __name__ == "__main__":
    main()
