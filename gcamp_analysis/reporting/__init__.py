"""Reporting snapshots and filesystem writers for processed videos.

Reporting is intentionally separate from ``data_classes.video``. The
``Video`` class owns loaded data and mutable pipeline state; this package owns
immutable output snapshots and side effects such as Excel, NumPy, CSV, and
figure writing.
"""

from gcamp_analysis.reporting.video_statistics import VideoStatistics
from gcamp_analysis.reporting.video_writers import (
    VideoFiguresWriter,
    VideoStatisticsWriter,
)
from gcamp_analysis.reporting.experiment_writers import (
    build_comparison_legend,
    save_comparisons,
)

__all__ = [
    "VideoStatistics",
    "VideoStatisticsWriter",
    "VideoFiguresWriter",
    "build_comparison_legend",
    "save_comparisons",
]
