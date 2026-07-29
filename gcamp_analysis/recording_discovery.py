"""Shared conventions for recording-folder naming and discovery.

A recording folder name encodes a region identity and an optional day. The
complete base name is the region identity (``1-1`` and ``1-2`` are different
regions), and a ``_DayN`` suffix marks repeated observations of that region;
a folder without the suffix is day 1.

Both the longitudinal and waves analyses rely on this convention. Keep the
single source of truth here so the two entry points cannot drift.
"""
from __future__ import annotations

import re

DAY_SUFFIX_RE = re.compile(r"_Day(?P<day>\d+)$", re.IGNORECASE)
REGION_DAY_RE = re.compile(r"^(?P<region>\d+-\d+)(?:_Day(?P<day>\d+))?$", re.IGNORECASE)


def parse_day(name: str) -> int:
    """Return the day encoded in a recording name.

    Folders without a ``_DayN`` suffix are day 1.
    """
    match = DAY_SUFFIX_RE.search(name)
    return int(match.group("day")) if match else 1


def parse_region_day(name: str) -> tuple[str, int] | None:
    """Return ``(region, day)`` for a ``<region>[_DayN]`` folder, or ``None``.

    The region must look like ``\\d+-\\d+`` (for example ``1-1``). Names that do
    not match this convention return ``None`` so callers can skip them.
    """
    match = REGION_DAY_RE.fullmatch(name)
    if match is None:
        return None
    return match.group("region"), int(match.group("day") or 1)
