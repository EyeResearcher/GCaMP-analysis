from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from experiments.tree import TreeNode, is_video_dir
from data_classes.video import Video, VideoStatistics, VideoStatisticsWriter
from pipeline.video_runner import VideoPipelineRunner
@dataclass(frozen=True)
class VideoRunRecord:
    video_dir: Path
    metrics_dir: Path

    n_rois_total: int
    n_rois_good: int
    n_neurons: int
    n_spikes_kept: int
    n_groups: int
class ExperimentProcessor:
    def __init__(self, runner, models, config, output_root: Path):
        self.runner = runner
        self.models = models
        self.config = config
        self.output_root = Path(output_root)

    def process_tree(self, root: TreeNode, verbose: bool = True) -> None:
        for node in root.iter_nodes():
            if is_video_dir(node.path):
                node.payload = self._process_one_video(node.path, verbose=verbose)

    def _process_one_video(self, video_dir: Path, verbose: bool) -> VideoRunRecord:
        suite2p_plane0 = video_dir / "suite2p" / "plane0"

        video = Video(path=video_dir, suite2p_path=suite2p_plane0)
        self.runner.run(video, models=self.models, config=self.config, verbose=verbose)

        stats = VideoStatistics.from_video(video)
        writer = VideoStatisticsWriter()

        # choose where you save; either per-video or centralized
        manifest = writer.write(stats, output_root=video_dir)

        # Pull summary counts from video fields (already populated by services)
        n_rois_total = int(video.n_rois)
        n_rois_good = int(video.n_good_rois)
        n_neurons = int(len(video.neurons))
        n_spikes_kept = int(sum(len(getattr(n, "peaks_filtered", [])) for n in video.neurons))
        n_groups = int(len(video.sttc_groups))  # or whatever you use

        metrics_dir = video_dir / "metrics"
        return VideoRunRecord(
            video_dir=video_dir,
            metrics_dir=metrics_dir,
            n_rois_total=n_rois_total,
            n_rois_good=n_rois_good,
            n_neurons=n_neurons,
            n_spikes_kept=n_spikes_kept,
            n_groups=n_groups,
        )
