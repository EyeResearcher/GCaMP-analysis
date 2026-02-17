"""Process an experiment tree: run the per-video pipeline, aggregate
summary statistics bottom-up, and compare sibling nodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from gcamp_analysis.experiments.tree import TreeNode, is_video_dir
from gcamp_analysis.experiments.summary_utils import (
    StatSummary,
    combine_neuron_level_to_video,
    aggregate_children,
)
from gcamp_analysis.data_classes.video import Video, VideoFiguresWriter, VideoStatistics, VideoStatisticsWriter
from gcamp_analysis.video_runner import VideoPipelineRunner


@dataclass(frozen=True)
class VideoRunRecord:
    """Immutable summary produced after a single video is processed.

    Attributes
    ----------
    video_dir : Path
        Root directory of the video.
    metrics_dir : Path
        Directory where per-video metrics are saved.
    n_rois_total : int
        Total ROIs detected by Suite2p.
    n_rois_good : int
        ROIs that passed the classifier.
    n_neurons : int
        Neurons retained after spike detection.
    n_spikes_kept : int
        Total spikes across all neurons after filtering.
    n_groups : int
        Number of neuron groups found by clustering.
    kin_unweighted : StatSummary
        Kinetics averaged equally across neurons.
    kin_weighted_spikes : StatSummary
        Kinetics weighted by per-neuron spike count.
    freq_unweighted : StatSummary
        Spike frequency averaged equally across neurons.
    """

    video_dir: Path
    metrics_dir: Path

    n_rois_total: int
    n_rois_good: int
    n_neurons: int
    n_spikes_kept: int
    n_groups: int

    kin_unweighted: StatSummary = field(default_factory=StatSummary)
    kin_weighted_spikes: StatSummary = field(default_factory=StatSummary)
    freq_unweighted: StatSummary = field(default_factory=StatSummary)


class ExperimentProcessor:
    """Walk the experiment tree, process every video leaf, and propagate
    summary statistics upward through the hierarchy.

    Parameters
    ----------
    runner : VideoPipelineRunner
        Pre-configured pipeline runner (holds models and config).
    output_root : Path
        Top-level experiment directory used for output paths.
    """

    def __init__(self, runner: VideoPipelineRunner, output_root: Path):
        self.runner = runner
        self.output_root = Path(output_root)

    def process_tree(self, root: TreeNode, verbose: bool = True) -> None:
        """Run the full pipeline on every video leaf, then aggregate.

        Parameters
        ----------
        root : TreeNode
            Root of the experiment tree.
        verbose : bool, optional
            If ``True``, print per-video progress (default ``True``).
        """
        # 1) process leaves
        for node in root.iter_nodes():
            if is_video_dir(node.path):
                node.payload = self._process_one_video(node.path, verbose=verbose)

        # 2) compute bottom-up node summaries + counts
        self._compute_bottom_up_summaries(root)

    def _process_one_video(self, video_dir: Path, verbose: bool) -> VideoRunRecord:
        """Run the pipeline on a single video and return a record.

        Parameters
        ----------
        video_dir : Path
            Directory containing the ``suite2p/plane0/`` output.
        verbose : bool
            Passed through to ``runner.run``.

        Returns
        -------
        VideoRunRecord
        """
        suite2p_plane0 = video_dir / "suite2p" / "plane0"

        video = Video(path=video_dir, suite2p_path=suite2p_plane0)
        self.runner.run(video, verbose=verbose)

        stats = VideoStatistics.from_video(video)
        stat_writer = VideoStatisticsWriter()
        stat_writer.write(stats, output_root=video_dir)
        figure_writer = VideoFiguresWriter()
        figure_writer.write(video)

        n_rois_total = int(video.n_rois)
        n_rois_good = int(video.n_good_rois)
        n_neurons = int(len(video.neurons))
        n_spikes_kept = int(sum(len(getattr(n, "peaks_filtered", [])) for n in video.neurons))
        n_groups = int(len(video.corr_groups))

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
        """Propagate counts and statistics from leaves to root.

        Uses a post-order traversal so that every node is visited after
        all of its descendants.

        Weighting rules
        ---------------
        * **Unweighted** — each immediate child counts equally.
        * **Weighted** — if children are video leaves, weight by
          ``n_neurons``; otherwise weight by ``n_videos``.
        """
        def post(node: TreeNode) -> None:
            for ch in node.children.values():
                post(ch)

            if node.is_leaf():
                if isinstance(node.payload, VideoRunRecord):
                    node.n_videos = 1
                    node.n_neurons = node.payload.n_neurons
                    node.n_groups = node.payload.n_groups
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
            node.n_groups = sum(ch.n_groups for ch in node.children.values())
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

    # ------------------------------------------------------------------
    # Sibling comparison
    # ------------------------------------------------------------------

    def compare_siblings(self, root: TreeNode) -> dict[Path, pd.DataFrame]:
        """Compare children at every internal node of *root*.

        Parameters
        ----------
        root : TreeNode
            Root of a processed experiment tree (``process_tree`` must
            have been called first).

        Returns
        -------
        dict[Path, pd.DataFrame]
            One entry per internal node that has >= 2 children with
            data.  Keyed by the parent node's filesystem path.
        """
        results: dict[Path, pd.DataFrame] = {}
        for node in root.iter_nodes():
            if not node.children or len(node.children) < 2:
                continue
            df = self._compare_one(node)
            if df is not None:
                results[node.path] = df
        return results

    @staticmethod
    def _compare_one(parent: TreeNode) -> Optional[pd.DataFrame]:
        """Build a comparison DataFrame for children of *parent*."""
        rows = []
        for child in parent.children.values():
            if child.n_videos <= 0:
                continue

            row: dict = {
                "child": child.name,
                "n_videos": child.n_videos,
                "n_neurons": child.n_neurons,
                "n_groups": child.n_groups,
            }

            # Flatten kinetics (unweighted + weighted)
            for summary, scheme in [
                (child.kin_unweighted, "unweighted"),
                (child.kin_weighted, "weighted"),
            ]:
                for stat in sorted(summary.means):
                    row[f"{stat}_mean_{scheme}"] = summary.means[stat]
                    row[f"{stat}_var_{scheme}"] = summary.vars_total.get(stat, 0.0)
                    row[f"{stat}_within_{scheme}"] = summary.vars_within.get(stat, 0.0)
                    row[f"{stat}_between_{scheme}"] = summary.vars_between.get(stat, 0.0)

            # Flatten frequency (unweighted + weighted)
            for summary, scheme in [
                (child.freq_unweighted, "unweighted"),
                (child.freq_weighted, "weighted"),
            ]:
                for stat in sorted(summary.means):
                    key = f"{stat}_{{}}_{scheme}"
                    row[key.format("mean")] = summary.means[stat]
                    row[key.format("var")] = summary.vars_total.get(stat, 0.0)
                    row[key.format("within")] = summary.vars_within.get(stat, 0.0)
                    row[key.format("between")] = summary.vars_between.get(stat, 0.0)

            rows.append(row)

        if len(rows) < 2:
            return None
        return pd.DataFrame(rows).sort_values("child")
