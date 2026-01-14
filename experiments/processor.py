# experiments/processor.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from experiments.tree import TreeNode, is_video_dir
from experiments.summary_utils import (
    StatSummary,
    combine_neuron_level_to_video,
    aggregate_children,
)
from data_classes.video import Video, VideoFiguresWriter, VideoStatistics, VideoStatisticsWriter
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

    # leaf summaries (over neurons in this video)
    kin_unweighted: StatSummary = field(default_factory=StatSummary)
    kin_weighted_spikes: StatSummary = field(default_factory=StatSummary)
    freq_unweighted: StatSummary = field(default_factory=StatSummary)


class ExperimentProcessor:
    def __init__(self, runner: VideoPipelineRunner, models: dict, config: dict, output_root: Path):
        self.runner = runner
        self.models = models
        self.config = config
        self.output_root = Path(output_root)

    def process_tree(self, root: TreeNode, verbose: bool = True) -> None:
        # 1) process leaves
        for node in root.iter_nodes():
            if is_video_dir(node.path):
                node.payload = self._process_one_video(node.path, verbose=verbose)

        # 2) compute bottom-up node summaries + counts
        self._compute_bottom_up_summaries(root)

    def _process_one_video(self, video_dir: Path, verbose: bool) -> VideoRunRecord:
        suite2p_plane0 = video_dir / "suite2p" / "plane0"

        video = Video(path=video_dir, suite2p_path=suite2p_plane0)
        self.runner.run(video, models=self.models, config=self.config, verbose=verbose)

        stats = VideoStatistics.from_video(video)
        stat_writer = VideoStatisticsWriter()
        stat_writer.write(stats, output_root=video_dir)
        figure_writer = VideoFiguresWriter()
        figure_writer.write(video)

        n_rois_total = int(video.n_rois)
        n_rois_good = int(video.n_good_rois)
        n_neurons = int(len(video.neurons))
        n_spikes_kept = int(sum(len(getattr(n, "peaks_filtered", [])) for n in video.neurons))
        n_groups = int(len(video.sttc_groups))

        # Leaf summaries from per-neuron summary_df:
        # - kinetics: unweighted + spike-weighted (number_of_spikes)
        # - frequency: unweighted (never spike-weighted)
        kin_unw, kin_wspk, freq_unw = combine_neuron_level_to_video(
            video.summary_df,
            spike_count_col="number_of_spikes",
            spike_freq_col="spike_frequency",
        )

        metrics_dir = video_dir / "metrics"
        return VideoRunRecord(
            video_dir=video_dir,
            metrics_dir=metrics_dir,
            n_rois_total=n_rois_total,
            n_rois_good=n_rois_good,
            n_neurons=n_neurons,
            n_spikes_kept=n_spikes_kept,
            n_groups=n_groups,
            kin_unweighted=kin_unw,
            kin_weighted_spikes=kin_wspk,
            freq_unweighted=freq_unw,
        )

    def _compute_bottom_up_summaries(self, root: TreeNode) -> None:
        """
        Post-order traversal:
          - Leaf nodes take summaries from their VideoRunRecord
          - Internal nodes aggregate children with:
              * unweighted: equal weight per immediate child
              * weighted:
                  - if children are videos (leaf nodes), weight by child.n_neurons
                  - else weight by child.n_videos
        """
        def post(node: TreeNode) -> None:
            for ch in node.children.values():
                post(ch)

            if node.is_leaf():
                if isinstance(node.payload, VideoRunRecord):
                    node.n_videos = 1
                    node.n_neurons = node.payload.n_neurons

                    # For comparisons, define:
                    # - kin_unweighted: unweighted across neurons inside the video
                    # - kin_weighted: spike-weighted across neurons inside the video
                    node.kin_unweighted = node.payload.kin_unweighted
                    node.kin_weighted = node.payload.kin_weighted_spikes

                    # Frequency: only unweighted at neuron level
                    node.freq_unweighted = node.payload.freq_unweighted
                    node.freq_weighted = node.payload.freq_unweighted  # same at leaf
                return

            # internal node: counts
            node.n_videos = sum(ch.n_videos for ch in node.children.values())
            node.n_neurons = sum(ch.n_neurons for ch in node.children.values())

            kids = list(node.children.values())

            # Unweighted: each immediate child counts equally
            node.kin_unweighted = aggregate_children([(ch.kin_weighted, 1.0) for ch in kids])
            node.freq_unweighted = aggregate_children([(ch.freq_weighted, 1.0) for ch in kids])

            # Weighted: choose weight basis by level
            children_are_videos = all(ch.is_leaf() for ch in kids)
            if children_are_videos:
                # e.g., Week1 comparing videos => weight by neurons per video
                w = [(ch.kin_weighted, float(ch.n_neurons)) for ch in kids]
                wf = [(ch.freq_weighted, float(ch.n_neurons)) for ch in kids]
            else:
                # e.g., GABA comparing timepoints => weight by number of videos per timepoint
                w = [(ch.kin_weighted, float(ch.n_videos)) for ch in kids]
                wf = [(ch.freq_weighted, float(ch.n_videos)) for ch in kids]

            node.kin_weighted = aggregate_children(w)
            node.freq_weighted = aggregate_children(wf)

        post(root)
