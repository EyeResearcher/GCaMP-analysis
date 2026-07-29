"""Concatenated-video metadata models, parsing, and explicit loading."""

from gcamp_analysis.concatenation.metadata import (
    ConcatMetadata,
    ConcatSection,
    find_concat_summary,
    load_concat_metadata,
    normalize_section_key,
    parse_concat_sections,
    validate_section_kind,
)

__all__ = [
    "ConcatMetadata",
    "ConcatSection",
    "find_concat_summary",
    "load_concat_metadata",
    "normalize_section_key",
    "parse_concat_sections",
    "validate_section_kind",
]
