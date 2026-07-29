"""Concatenated-video metadata models, validation, and loading.

Parsing and normalization functions are pure: callers provide a DataFrame and
video dimensions and receive validated section descriptors. Filesystem access
is isolated to ``find_concat_summary`` and ``load_concat_metadata`` so I/O is
explicit at the call site.

To support a new section kind, update ``validate_section_kind`` and the
initial counters in ``parse_concat_sections``. Frame validation and stable key
generation should remain centralized here rather than on ``Video``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ConcatSection:
    """Normalized description of one concatenated-video section."""

    index: int
    source_file_name: str
    section_kind: str
    section_key: str
    start_frame: int
    end_frame: int

    @property
    def frame_slice(self) -> slice:
        return slice(self.start_frame, self.end_frame)

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class ConcatMetadata:
    """Validated metadata loaded for one concatenated video."""

    summary_path: Path
    summary_df: pd.DataFrame
    sections: tuple[ConcatSection, ...]

    @property
    def sections_by_key(self) -> dict[str, ConcatSection]:
        return {section.section_key: section for section in self.sections}


def normalize_section_key(section_type: str) -> str:
    """Normalize a section label into a stable dictionary key."""
    normalized = (
        section_type.strip().lower().replace(" ", "_").replace("-", "_")
    )
    if not normalized:
        raise ValueError(f"Could not normalize section type '{section_type}'.")
    return normalized


def validate_section_kind(section_type: str) -> str:
    """Return a supported canonical section kind."""
    normalized = normalize_section_key(section_type)
    allowed = {"baseline", "treatment", "recovery"}
    if normalized not in allowed:
        raise ValueError(
            "Concatenation summary CSV section type must be one of "
            f"{sorted(allowed)}. Got '{section_type}'."
        )
    return normalized


def find_concat_summary(video_path: Path) -> Path:
    """Resolve the unique concat summary CSV in a video directory."""
    video_path = Path(video_path)
    candidates = sorted(video_path.glob("*_concat_order.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Concatenated video '{video_path}' is missing the required "
            "'*_concat_order.csv' file."
        )
    raise ValueError(
        f"Concatenated video '{video_path}' has multiple "
        "'*_concat_order.csv' files."
    )


def parse_concat_sections(
    summary_df: pd.DataFrame,
    *,
    n_frames: int,
    video_path: Path,
) -> list[ConcatSection]:
    """Validate a concat summary table and return normalized sections."""
    expected_columns = [
        "index",
        "source file name",
        "section type",
        "start frame",
        "end frame",
    ]
    normalized_columns = [
        str(column).strip().lower()
        for column in summary_df.columns.tolist()
    ]
    if normalized_columns[: len(expected_columns)] != expected_columns:
        raise ValueError(
            "Concatenation summary CSV must start with columns: "
            f"{expected_columns}. Got {summary_df.columns.tolist()}."
        )

    if summary_df.empty:
        raise ValueError(
            "Concatenation summary CSV must contain at least one section row."
        )

    video_path = Path(video_path)
    sections: list[ConcatSection] = []
    previous_end = 0
    seen_keys: set[str] = set()
    kind_counts = {"baseline": 0, "treatment": 0, "recovery": 0}

    for row_number, row in enumerate(
        summary_df.itertuples(index=False, name=None),
        start=1,
    ):
        index_value = int(row[0])
        source_file_name = str(row[1]).strip()
        section_kind = validate_section_kind(str(row[2]).strip())
        start_frame = int(row[3])
        end_frame = int(row[4])

        kind_counts[section_kind] += 1
        if section_kind == "baseline":
            if kind_counts[section_kind] > 1:
                raise ValueError(
                    f"Concatenated video '{video_path}' must define exactly "
                    "one baseline section."
                )
            section_key = "baseline"
        else:
            section_key = f"{section_kind}_{kind_counts[section_kind]}"

        if section_key in seen_keys:
            raise ValueError(
                f"Duplicate section key '{section_key}' in concat summary."
            )
        seen_keys.add(section_key)

        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError(
                f"Invalid frame range for concat row {row_number}: "
                f"start={start_frame}, end={end_frame}."
            )
        if end_frame > n_frames:
            raise ValueError(
                f"Concat row {row_number} ends at frame {end_frame}, past "
                f"video length {n_frames}."
            )
        if start_frame < previous_end:
            raise ValueError(
                "Concat rows must be non-overlapping and ordered. "
                f"Row {row_number} starts at {start_frame} after previous "
                f"end {previous_end}."
            )
        previous_end = end_frame

        sections.append(
            ConcatSection(
                index=index_value,
                source_file_name=source_file_name,
                section_kind=section_kind,
                section_key=section_key,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )

    if kind_counts["baseline"] != 1:
        raise ValueError(
            f"Concatenated video '{video_path}' must define an explicit "
            "baseline section."
        )

    return sections


def load_concat_metadata(
    video_path: Path,
    *,
    n_frames: int,
) -> ConcatMetadata:
    """Load and validate concat metadata from a video directory."""
    summary_path = find_concat_summary(video_path)
    summary_df = pd.read_csv(summary_path)
    sections = parse_concat_sections(
        summary_df,
        n_frames=n_frames,
        video_path=video_path,
    )
    return ConcatMetadata(
        summary_path=summary_path,
        summary_df=summary_df,
        sections=tuple(sections),
    )
