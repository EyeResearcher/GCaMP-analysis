"""
Reverse the concatenation performed by ``concat_videos.py``.

Recursively searches an input directory for concatenated TIFF stacks
(``*_concat.tiff``) and their companion order CSV (``*_concat_order.csv``).
For each concatenated video, the stack is split back into its original
segments using the recorded 0-based, end-exclusive frame ranges, and every
segment is written to its **own folder** named after the source video stem::

    <output_root>/<relative_parent>/<source_stem>/<source_file_name>

For example, given ``BP/1-3/1-3_concat.tiff`` with segments ``1-3.tiff`` and
``1-3_Day2.tiff``, the outputs are::

    BP/1-3/1-3/1-3.tiff
    BP/1-3/1-3_Day2/1-3_Day2.tiff

The order CSV columns match what ``concat_videos.py`` writes::

    index, source file name, section type, start frame, end frame

Usage
-----
    python deconcat_videos.py <input_path> <output_path> [--overwrite] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import tifffile


@dataclass(frozen=True)
class DeconcatSegment:
    """One segment to extract from a concatenated stack."""

    index: int
    source_file_name: str
    section_type: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class DeconcatSet:
    """A concatenated TIFF paired with its parsed segments."""

    concat_tiff: Path
    order_csv: Path
    segments: list[DeconcatSegment]


def _find_order_csv(concat_tiff: Path) -> Path:
    """Resolve the ``*_concat_order.csv`` companion for *concat_tiff*."""
    # e.g. "1-3_concat.tiff" -> "1-3_concat_order.csv"
    stem = concat_tiff.name
    for suffix in (".tiff", ".tif", ".TIFF", ".TIF"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidate = concat_tiff.with_name(f"{stem}_order.csv")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find order CSV for '{concat_tiff.name}'. Expected '{candidate.name}'."
    )


def _parse_order_csv(order_csv: Path) -> list[DeconcatSegment]:
    """Parse the concat order CSV into a list of segments."""
    segments: list[DeconcatSegment] = []
    with order_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"index", "source file name", "section type", "start frame", "end frame"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Order CSV '{order_csv}' is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            segments.append(
                DeconcatSegment(
                    index=int(row["index"]),
                    source_file_name=row["source file name"].strip(),
                    section_type=row["section type"].strip(),
                    start_frame=int(row["start frame"]),
                    end_frame=int(row["end frame"]),
                )
            )
    if not segments:
        raise ValueError(f"Order CSV '{order_csv}' contains no segments.")
    segments.sort(key=lambda seg: seg.start_frame)
    return segments


def find_sets(root: Path) -> list[DeconcatSet]:
    """Return concatenated TIFF sets found under *root*."""
    sets: list[DeconcatSet] = []
    for concat_tiff in sorted(root.rglob("*_concat.tiff")):
        if not concat_tiff.is_file():
            continue
        try:
            order_csv = _find_order_csv(concat_tiff)
            segments = _parse_order_csv(order_csv)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [warn] Skipping '{concat_tiff}': {exc}")
            continue
        sets.append(DeconcatSet(concat_tiff=concat_tiff, order_csv=order_csv, segments=segments))
    return sets


def _segment_output_path(
    concat_tiff: Path, root: Path, output_root: Path, segment: DeconcatSegment
) -> Path:
    """Compute the per-segment output path, giving each video its own folder."""
    rel_parent = concat_tiff.parent.relative_to(root)
    source_stem = Path(segment.source_file_name).stem
    return output_root / rel_parent / source_stem / segment.source_file_name


def deconcat_set(
    ds: DeconcatSet,
    root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> None:
    """Split one concatenated stack into per-segment TIFFs."""
    print(f"  Reading {ds.concat_tiff.name:<32} {ds.concat_tiff}")
    frames = tifffile.imread(str(ds.concat_tiff))
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
    total = frames.shape[0]

    for segment in ds.segments:
        if not (0 <= segment.start_frame < segment.end_frame <= total):
            print(
                f"  !! Segment '{segment.source_file_name}' has invalid range "
                f"[{segment.start_frame}, {segment.end_frame}) for stack of {total} frames -- skipped"
            )
            continue

        out_path = _segment_output_path(ds.concat_tiff, root, output_root, segment)
        if out_path.exists() and not overwrite:
            print(f"    - {segment.source_file_name:<32} skipped (exists): {out_path}")
            continue

        chunk = frames[segment.start_frame : segment.end_frame]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(out_path), chunk, photometric="minisblack")
        print(
            f"    - {segment.source_file_name:<32} "
            f"frames [{segment.start_frame}:{segment.end_frame}] "
            f"({chunk.shape[0]}) -> {out_path}"
        )


def dry_run_set(ds: DeconcatSet, root: Path, output_root: Path) -> None:
    """Print what would happen for *ds* without reading frames or writing files."""
    print("  [dry-run] Would split into per-segment folders:")
    for segment in ds.segments:
        out_path = _segment_output_path(ds.concat_tiff, root, output_root, segment)
        print(
            f"    - {segment.source_file_name:<32} "
            f"frames [{segment.start_frame}:{segment.end_frame}] "
            f"({segment.end_frame - segment.start_frame}) -> {out_path}"
        )


def deconcat_all_sets(
    sets: list[DeconcatSet],
    root: Path,
    output_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    for ds in sets:
        tag = " [DRY RUN]" if dry_run else ""
        print(f"> {ds.concat_tiff.relative_to(root)}  [{len(ds.segments)} segments]{tag}")
        try:
            if dry_run:
                dry_run_set(ds, root, output_root)
            else:
                deconcat_set(ds, root, output_root, overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001 -- report and continue with next set
            print(f"  !! Error: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reverse concat_videos.py: split concatenated TIFF stacks back into "
        "per-segment videos, each in its own folder."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Root directory to recursively search for '*_concat.tiff' sets.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Root directory for split outputs (mirrors input structure).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing per-segment TIFFs instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be done without reading frames or writing files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_root: Path = args.input_path
    output_root: Path = args.output_path

    if not input_root.exists():
        print(f"!! Input path does not exist: {input_root}")
        sys.exit(1)

    if args.dry_run:
        print("*** DRY RUN -- no files will be read or written ***")
    print(f"Scanning: {input_root}")
    sets = find_sets(input_root)

    if not sets:
        print("No concatenated ('*_concat.tiff') sets found.")
        sys.exit(0)

    print(f"Found {len(sets)} concatenated set(s) to split.\n")
    deconcat_all_sets(sets, input_root, output_root, overwrite=args.overwrite, dry_run=args.dry_run)
    print("\nDone." + ("  (dry run -- nothing was changed)" if args.dry_run else ""))


# EXAMPLE USAGE:
# python preprocessing/deconcat_videos.py "C:\Users\mzinn1\Desktop\GCaMP6s_EX37x_Days_Repeating" "C:\Users\mzinn1\Desktop\GCaMP6s_EX37x_Days_Repeating" --dry-run
if __name__ == "__main__":
    main()
