from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Any
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pipeline.neuron_grouping import group_neurons_by_sttc, group_neurons_by_dtw, compare_groupings
from pipeline.reports import GroupingReport
from utils.visualization import visualize_neuron_groups
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data_classes.video import Video

class GroupingStrategy(Protocol):
    name: str
    def compute(self, video: "Video", config: dict) -> dict:
        ...

@dataclass
class STTCStrategy:
    name: str = "sttc"
    def compute(self, video: "Video", config: dict) -> dict:
        tw = config.get("time_window", 0.4)
        dt = config.get("distance_threshold", 0.2)
        groups, matrix = group_neurons_by_sttc(video.neurons, video.n_frames, time_window=tw, distance_threshold=dt)
        return {"groups": groups, "matrix": matrix, "config_label": f"sttc_tw{tw}_dt{dt}"}

@dataclass
class DTWStrategy:
    name: str = "dtw"
    def compute(self, video: "Video", config: dict) -> dict:
        groups, matrix = group_neurons_by_dtw(video.neurons, **config)
        return {"groups": groups, "matrix": matrix, "config_label": "dtw"}

@dataclass
class GroupingService:
    enable_dtw: bool = False  # you currently have DTW commented out :contentReference[oaicite:7]{index=7}

    def run(self, video: "Video", grouping_cfg: dict) -> GroupingReport:
        if len(video.neurons) < 2:
            video.grouping_stats = pd.DataFrame()
            return

        sttc_cfg = grouping_cfg.get("sttc", {})
        dtw_cfg = grouping_cfg.get("dtw", {})

        sttc_res = STTCStrategy().compute(video, sttc_cfg)
        video.sttc_groups = sttc_res["groups"]
        video.sttc_matrix = sttc_res["matrix"]

        if self.enable_dtw:
            dtw_res = DTWStrategy().compute(video, dtw_cfg)
            video.dtw_groups = dtw_res["groups"]
            video.dtw_matrix = dtw_res["matrix"]
        else:
            video.dtw_groups, video.dtw_matrix = [], np.ndarray([])

        # Combine/compare like your current logic
        grouping_summary = compare_groupings(video.sttc_groups, video.dtw_groups, video.sttc_matrix, video.dtw_matrix, video.neurons)

        video.sttc_groups = grouping_summary["sttc_groups"]
        video.dtw_groups = grouping_summary["dtw_groups"]
        video.agreement = grouping_summary["agreement"]
        video.grouping_stats = pd.DataFrame([grouping_summary["combined_stats"]])
        return GroupingReport(method = "sttc" + ("+dtw" if self.enable_dtw else ""),
                              n_groups= len(video.sttc_groups),
                              agreement= video.agreement)

    def visualize(self, video: "Video", which: str = "sttc") -> Optional[Figure]:
        if which == "sttc":
            groups = video.sttc_groups
            label = "sttc_grouping"
        else:
            groups = video.dtw_groups
            label = "dtw_grouping"

        if not groups:
            return None

        img_size = (
            video.suite2p_data.get("ops", {}).get("Ly", 1024),
            video.suite2p_data.get("ops", {}).get("Lx", 1024),
        )
        fig, _ = visualize_neuron_groups(
            neuron_groups=groups,
            stat=video.suite2p_data["stat"] if "stat" in video.suite2p_data else np.array([]),
            img_size=img_size,
            video_path=video.path,
            config_label=label,
        )

        if which == "sttc":
            video.sttc_fig = fig
        else:
            video.dtw_fig = fig

        return fig
