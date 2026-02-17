from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, TYPE_CHECKING
import numpy as np

from grouping_processing.similarity.sttc import STTCSimilarity
from grouping_processing.clustering.hierarchical import HierarchicalClusterer

if TYPE_CHECKING:
    from data_classes.video import Video

@dataclass
class STTCStrategy:
    name: str = "sttc"
    similarity: STTCSimilarity = field(default_factory=STTCSimilarity)
    clusterer: HierarchicalClusterer = field(default_factory=HierarchicalClusterer)

    def compute(self, video: "Video", config: Dict[str, Any]) -> Dict[str, Any]:
        tw = float(config.get("time_window", self.similarity.time_window))
        dt = float(config.get("distance_threshold", self.clusterer.distance_threshold))
        linkage_method = config.get("linkage_method", self.clusterer.linkage_method)
        min_group_size = int(config.get("min_group_size", self.clusterer.min_group_size))

        sim = STTCSimilarity(time_window=tw, fs=float(getattr(video, "fs", 30.0)))
        sttc = sim.compute(video.neurons, video.n_frames)

        # correlation -> distance
        dist = 1.0 - sttc

        clusterer = HierarchicalClusterer(
            linkage_method=str(linkage_method),
            distance_threshold=dt,
            min_group_size=min_group_size,
        )

        groups = clusterer.cluster_from_distance(
            video.neurons,
            dist,
            group_id_prefix="sttc",
            method="sttc",
            meta={"t_win": tw, "sttc_thresh": 1.0 - dt},
        )

        return {"groups": groups, "matrix": sttc, "config_label": f"sttc_tw{tw}_dt{dt}"}
