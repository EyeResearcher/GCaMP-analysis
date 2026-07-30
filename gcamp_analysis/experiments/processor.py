"""Orchestrate per-video processing and hierarchical experiment summaries.

This module connects the video-analysis pipeline to the experiment-tree
summary model. Its responsibilities are deliberately limited to:

1. Run the pipeline for each video leaf.
2. Extract per-video values into a :class:`VideoRunRecord`.
3. Traverse the tree in post-order and assign summaries produced by the
   pure functions in ``summary_utils``.
4. Flatten child summaries into sibling-comparison tables.

Aggregation rules do not belong in ``ExperimentProcessor``. Leaf conversion
is handled by ``summary_from_video_record`` and parent aggregation is handled
by ``aggregate_node_summaries``. Keeping those computations pure makes them
testable without constructing a pipeline runner or filesystem tree.

Adding an aggregated statistic
------------------------------
For a value first calculated from a ``Video``:

1. Calculate or extract it in ``_process_one_video`` (or a focused helper).
2. Add the leaf-level field to ``VideoRunRecord`` in ``models.py``.
3. Add the tree-level field to ``NodeSummary`` in ``summary_utils.py``.
4. Map the record field in ``summary_from_video_record``.
5. Define its weighting/merge rule in ``aggregate_node_summaries``.
6. Add a ``TreeNode`` compatibility property only if existing callers need
   direct ``node.<field>`` access.
7. Add structural fields to ``comparison_utils.summary_to_comparison_row`` or
   an output writer only when they should appear in exported results. New keys
   inside existing ``StatSummary`` fields are flattened automatically.

Statistics already represented inside ``StatSummary`` usually require no new
dataclass field. Add the statistic to the per-neuron input columns consumed by
``summarize_video`` and it will propagate by key through ``aggregate_children``.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from gcamp_analysis.experiments.tree import TreeNode, is_video_dir
from gcamp_analysis.experiments.summary_utils import (
    StatSummary,
    summarize_video,
    aggregate_node_summaries,
    summary_from_video_record,
)
from gcamp_analysis.experiments.models import VideoRunRecord
from gcamp_analysis.experiments.comparison_utils import build_sibling_comparison
from gcamp_analysis.data_classes.video import Video
from gcamp_analysis.reporting import (
    VideoFiguresWriter,
    VideoStatistics,
    VideoStatisticsWriter,
)
from gcamp_analysis.video_runner import VideoPipelineRunner


class _GroupedPartition(NamedTuple):
    """Result of partitioning neurons into grouped vs. ungrouped."""
    n_grouped: int
    n_ungrouped: int
    kin_grouped: StatSummary
    kin_ungrouped: StatSummary
    freq_grouped: StatSummary
    freq_ungrouped: StatSummary


class ExperimentProcessor:
    """Walk the experiment tree, process every video leaf, and propagate
    summary statistics upward through the hierarchy.

    Parameters
    ----------
    runner : VideoPipelineRunner
        Pre-configured pipeline runner (holds models and config).
    output_root : Path
        Top-level experiment directory used for output paths.
    dry_run : bool, optional
        Compute all analysis results without invoking filesystem writers.
    """

    def __init__(
        self,
        runner: VideoPipelineRunner,
        output_root: Path,
        dry_run: bool = False,
    ):
        self.runner = runner
        self.output_root = Path(output_root)
        self.dry_run = dry_run

    def process_tree(self, root: TreeNode, verbose: bool = True) -> None:
        """Run the full pipeline on every video leaf, then aggregate.

        Parameters
        ----------
        root : TreeNode
            Root of the experiment tree.
        verbose : bool, optional
            If ``True``, print per-video progress (default ``True``).
        """
        for node in root.iter_nodes():
            if is_video_dir(node.path):
                node.payload = self._process_one_video(node.path, verbose=verbose)
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

        video = Video.from_suite2p(
            path=video_dir,
            suite2p_path=suite2p_plane0,
        )
        stats = None
        try:
            self.runner.run(video, verbose=verbose)

            stats = VideoStatistics.from_video(video)
            if not self.dry_run:
                stat_writer = VideoStatisticsWriter()
                stat_writer.write(stats, output_root=video_dir)
                figure_writer = VideoFiguresWriter()
                figure_writer.write(video)

            n_rois_total = int(video.n_rois)
            n_rois_good = int(video.n_good_rois)
            n_neurons = int(len(video.neurons))
            n_spikes_kept = int(sum(len(getattr(n, "peaks_filtered", [])) for n in video.neurons))
            n_groups_per_strategy = {
                name: len(result.groups)
                for name, result in video.grouping_results.items()
            }

            group_stats = self._compute_group_stats(video)

            kin_unw, kin_wspk, freq_unw = summarize_video(
                video.summary_df,
                spike_count_col="number_of_spikes",
                spike_freq_col="spike_frequency",
            )

            part = self._partition_grouped_ungrouped(video.summary_df, video.grouping_results)

            metrics_dir = video_dir / "metrics"
            return VideoRunRecord(
                video_dir=video_dir,
                metrics_dir=metrics_dir,
                n_rois_total=n_rois_total,
                n_rois_good=n_rois_good,
                n_neurons=n_neurons,
                n_spikes_kept=n_spikes_kept,
                n_neurons_grouped=part.n_grouped,
                n_neurons_ungrouped=part.n_ungrouped,
                n_groups_per_strategy=n_groups_per_strategy,
                group_stats=group_stats,
                kin_unweighted=kin_unw,
                kin_weighted_spikes=kin_wspk,
                freq_unweighted=freq_unw,
                kin_grouped=part.kin_grouped,
                kin_ungrouped=part.kin_ungrouped,
                freq_grouped=part.freq_grouped,
                freq_ungrouped=part.freq_ungrouped,
                light_evoked_details=stats.light_evoked_details,
            )
        finally:
            # Release array views held by ROIs/neurons before dropping the
            # memory-mapped Suite2p inputs. This also runs when a notebook
            # cell fails, preventing IPython's retained traceback from
            # keeping a complete video resident.
            stats = None
            video.clear_results()
            video.suite2p_data.clear()

    @staticmethod
    def _compute_group_stats(video: Video) -> dict[str, dict[str, float]]:
        """Compute per-strategy group-level scalar statistics.

        For each grouping strategy, computes the following: 

            mean/median group size
            mean intra-group correlation 
            mean spikes per group

        For the ``light-evoked`` strategy, it adds the following: 

            {subtype} per response 
            {subtype} total

        Parameters
        ----------
        video : Video
            A fully processed video with ``grouping_results`` populated.

        Returns
        -------
        dict[str, dict[str, float]]
            Outer key is the strategy name, inner dict holds the stats.
        """
        group_stats: dict[str, dict[str, float]] = {}
        for name, r in video.grouping_results.items():
            sizes = [g.size for g in r.groups] if r.groups else []
            corrs = [g.group_mean_similarity(r.matrix) for g in r.groups] if r.groups else []
            corrs = [c for c in corrs if np.isfinite(c)]
            spikes_per_group = [
                sum(len(n.spikes) for n in g.neurons)
                for g in r.groups
            ] if r.groups else []

            group_stats[name] = {
                "mean_group_size": float(np.mean(sizes)) if sizes else 0.0,
                "median_group_size": float(np.median(sizes)) if sizes else 0.0,
                "mean_group_corr": float(np.mean(corrs)) if corrs else 0.0,
                "mean_spikes_per_group": float(np.mean(spikes_per_group)) if spikes_per_group else 0.0,
            }

            if name == "light-evoked" and r.groups:
                ExperimentProcessor._add_light_evoked_cell_counts(
                    group_stats[name], r.groups,
                )

        return group_stats

    @staticmethod
    def _partition_grouped_ungrouped(
        summary_df: pd.DataFrame,
        grouping_results: dict,
    ) -> _GroupedPartition:
        """Split *summary_df* into grouped/ungrouped neurons and summarize each."""
        grouped_indices: set[int] = set()
        for result in grouping_results.values():
            for group in result.groups:
                grouped_indices.update(getattr(group, "neuron_indices", []))

        if not summary_df.empty and grouped_indices:
            mask = summary_df.index.isin(grouped_indices)
            grouped_df = summary_df.loc[mask]
            ungrouped_df = summary_df.loc[~mask]
        else:
            grouped_df = pd.DataFrame()
            ungrouped_df = summary_df

        kin_grp, _, freq_grp = summarize_video(
            grouped_df,
            spike_count_col="number_of_spikes",
            spike_freq_col="spike_frequency",
        )
        kin_ungrp, _, freq_ungrp = summarize_video(
            ungrouped_df,
            spike_count_col="number_of_spikes",
            spike_freq_col="spike_frequency",
        )

        return _GroupedPartition(
            n_grouped=len(grouped_df),
            n_ungrouped=len(ungrouped_df),
            kin_grouped=kin_grp,
            kin_ungrouped=kin_ungrp,
            freq_grouped=freq_grp,
            freq_ungrouped=freq_ungrp,
        )

    @staticmethod
    def _add_light_evoked_cell_counts(
        stats: dict[str, float],
        groups: list,
    ) -> None:
        """
        Mutates *stats* in place to add total counts for each subtype.
        It requires that the prefix for the group ID describe the subtype.
        """
        type_totals: dict[str, int] = {}
        for g in groups:
            gid = str(g.group_id)
            subtype = gid.split("_")[0]
            type_totals[subtype] = type_totals.get(subtype, 0) + g.size
            stats[f"n_cells_{gid}"] = g.size

        for subtype, total in type_totals.items():
            stats[f"total_{subtype}_cells"] = total

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
                    node.summary = summary_from_video_record(
                        node.payload,
                        source=node.name,
                    )
                return

            kids = list(node.children.values())
            node.summary = aggregate_node_summaries(
                (child.summary for child in kids),
                children_are_videos=all(child.is_leaf() for child in kids),
            )

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
    def _compare_one(parent: TreeNode) -> pd.DataFrame | None:
        """Compatibility wrapper around the pure comparison formatter."""
        return build_sibling_comparison(
            (child.name, child.summary)
            for child in parent.children.values()
        )
