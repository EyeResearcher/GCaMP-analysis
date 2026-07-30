"""Compatibility imports for experiment reporting writers.

New code should import these functions from ``gcamp_analysis.reporting``.
This module remains to avoid breaking notebooks and external callers.
"""

from gcamp_analysis.reporting.experiment_writers import (
    build_comparison_legend,
    save_comparisons,
)

# Preserve the previous private helper for callers that used it despite its
# private name.
_build_legend = build_comparison_legend

__all__ = [
    "build_comparison_legend",
    "save_comparisons",
]
