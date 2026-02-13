from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from data_classes import video
import numpy as np
import pandas as pd

from pipeline.reports import GroupingReport

# NEW modular imports (your new framework)
from grouping_processing.strategies.dtw_strategy import DTWStrategy
from grouping_processing.strategies.corr_strategy import CorrelationStrategy
from grouping_processing.comparison import compare_groupings

if TYPE_CHECKING:
    from data_classes.video import Video


@dataclass
class GroupingService:
    """
    Compute-only grouping service.

    Side effects on `video`:
      - sttc_groups, sttc_matrix
      - dtw_groups, dtw_matrix (if enabled)
      - agreement
      - grouping_stats (DataFrame)

    NOTE: Visualization is intentionally NOT handled here anymore.
    Use VideoFiguresWriter to render and save figures.
    """
    enable_dtw: bool = False

    def run(self, video: "Video", grouping_cfg: dict) -> Optional[GroupingReport]:
        if len(video.neurons) < 2:
            # Keep fields consistent so downstream writers don't crash
            video.sttc_groups, video.sttc_matrix = [], np.asarray([])
            video.dtw_groups, video.dtw_matrix = [], np.asarray([])
            video.agreement = None
            video.grouping_stats = pd.DataFrame()
            return None

        dtw_cfg = grouping_cfg.get("dtw", {}) or {}
        #0) Compute correlatin grouping
        corr_cfg = grouping_cfg.get("corr", grouping_cfg.get("sttc", {})) or {}
        corr_res = CorrelationStrategy().compute(video, corr_cfg)
        video.sttc_groups = corr_res.get("groups", [])
        video.sttc_matrix = corr_res.get("matrix", np.asarray([]))

        # 2) compute DTW grouping (optional)
        if self.enable_dtw:
            dtw_res = DTWStrategy().compute(video, dtw_cfg)
            video.dtw_groups = dtw_res.get("groups", [])
            video.dtw_matrix = dtw_res.get("matrix", np.asarray([]))
            dtw_label = dtw_res.get("config_label", "dtw")
        else:
            video.dtw_groups, video.dtw_matrix = [], np.asarray([])
            dtw_label = "dtw_disabled"

        # 3) compare / combine summaries
        grouping_summary = compare_groupings(
            sttc_groups=video.sttc_groups,
            dtw_groups=video.dtw_groups,
            sttc_matrix=video.sttc_matrix if isinstance(video.sttc_matrix, np.ndarray) else None,
            dtw_matrix=video.dtw_matrix if isinstance(video.dtw_matrix, np.ndarray) else None,
            neurons=video.neurons,
        )

        # Preserve the same outputs your pipeline expects
        video.sttc_groups = grouping_summary.get("sttc_groups", video.sttc_groups)
        video.dtw_groups = grouping_summary.get("dtw_groups", video.dtw_groups)
        video.agreement = grouping_summary.get("agreement", None)

        combined_stats = grouping_summary.get("combined_stats", None)

        # Your old code wrapped combined_stats in a single-row df.
        # In the new modular version, combined_stats may be a list of rows.
        if combined_stats is None:
            video.grouping_stats = pd.DataFrame()
        elif isinstance(combined_stats, list):
            video.grouping_stats = pd.DataFrame(combined_stats)
        elif isinstance(combined_stats, dict):
            video.grouping_stats = pd.DataFrame([combined_stats])
        else:
            video.grouping_stats = pd.DataFrame()

        method = "corr" + ("+dtw" if self.enable_dtw else "")
        n_groups = len(video.sttc_groups) if video.sttc_groups else 0

        return GroupingReport(
            method=method,
            n_groups=n_groups,
            agreement=video.agreement,
        )
