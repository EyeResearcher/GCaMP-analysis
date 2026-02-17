from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, TYPE_CHECKING, Optional
import numpy as np

from grouping_processing.similarity.dtw import DTWSimilarity
from grouping_processing.clustering.hierarchical import HierarchicalClusterer

if TYPE_CHECKING:
    from data_classes.video import Video

@dataclass
class DTWStrategy:
    name: str = "dtw"
    similarity: DTWSimilarity = field(default_factory=DTWSimilarity)
    clusterer: HierarchicalClusterer = field(default_factory=lambda: HierarchicalClusterer(distance_threshold=0.3))

    def compute(self, video: "Video", config: Dict[str, Any]) -> Dict[str, Any]:
        down = int(config.get("downsample_factor", self.similarity.downsample_factor))
        use_gpu = bool(config.get("use_gpu", self.similarity.use_gpu))
        linkage_method = config.get("linkage_method", self.clusterer.linkage_method)
        dist_percentile = int(config.get("distance_percentile", 30))
        min_group_size = int(config.get("min_group_size", self.clusterer.min_group_size))

        sim = DTWSimilarity(downsample_factor=down, use_gpu=use_gpu)
        dtw = sim.compute(video.neurons)

        if dtw is None:
            return {"groups": [], "matrix": None, "config_label": "dtw_skipped"}

        dtw = np.asarray(dtw, dtype=float)

        # choose threshold by percentile (preserve your old behavior), then cluster
        nonzero = dtw[dtw > 0]
        if nonzero.size == 0:
            return {"groups": [], "matrix": dtw, "config_label": "dtw_empty"}

        thresh = float(np.percentile(nonzero, dist_percentile))

        clusterer = HierarchicalClusterer(
            linkage_method=str(linkage_method),
            distance_threshold=thresh,
            min_group_size=min_group_size,
        )

        groups = clusterer.cluster_from_distance(
            video.neurons,
            dtw,
            group_id_prefix="dtw",
            method="dtw",
            meta={"dtw_thresh": thresh},
        )

        return {"groups": groups, "matrix": dtw, "config_label": "dtw"}
