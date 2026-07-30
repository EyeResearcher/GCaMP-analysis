"""Immutable reporting snapshots derived from processed video state.

``VideoStatistics`` is the boundary between the mutable processing context
stored by ``Video`` and filesystem writers. Add a field here when a completed
per-video result must be persisted or consumed by reporting code. Populate it
in ``from_video`` and teach the appropriate writer how to serialize it.

Computation should remain in processing services. This module may project or
organize completed results, but it should not run the analysis pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.video import Video


@dataclass(frozen=True)
class VideoStatistics:
    """Pure in-memory container for per-video reporting outputs."""

    video_name: str

    per_neuron_spike_summaries: pd.DataFrame
    grouping_stats: pd.DataFrame
    bad_rois_features: pd.DataFrame

    # All strategy matrices keyed by strategy name
    matrices: dict = field(default_factory=dict)

    # Per-group light-evoked detail DataFrames keyed by group_id
    light_evoked_details: dict = field(default_factory=dict)

    @classmethod
    def from_video(cls, video: Video) -> VideoStatistics:
        """Create a reporting snapshot from a fully processed video."""
        from gcamp_analysis.grouping_processing.light_evoked_detail import (
            build_light_evoked_detail,
        )

        matrices = {
            name: result.matrix
            for name, result in video.grouping_results.items()
            if result.matrix is not None
        }

        light_evoked_details: dict = {}
        light_evoked_result = video.grouping_results.get("light-evoked")
        if light_evoked_result is not None:
            light_evoked_details = build_light_evoked_detail(
                light_evoked_result,
                fs=float(video.fs),
            )

        return cls(
            video_name=video.path.name,
            per_neuron_spike_summaries=video.summary_df,
            grouping_stats=video.grouping_stats,
            bad_rois_features=video.bad_rois_features,
            matrices=matrices,
            light_evoked_details=light_evoked_details,
        )
