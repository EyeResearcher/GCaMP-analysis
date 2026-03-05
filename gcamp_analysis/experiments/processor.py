"""Process an experiment tree: run the per-video pipeline, aggregate
summary statistics bottom-up, and compare sibling nodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
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
    n_groups : dict[str, int]
        Number of neuron groups per grouping strategy.
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
    n_neurons_grouped: int = 0
    n_neurons_ungrouped: int = 0
    n_groups: dict[str, int] = field(default_factory=dict)

    # Per-strategy group-level scalar stats
    # {strategy: {"mean_group_size": ..., "median_group_size": ..., "mean_group_corr": ...}}
    group_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    kin_unweighted: StatSummary = field(default_factory=StatSummary)
    kin_weighted_spikes: StatSummary = field(default_factory=StatSummary)
    freq_unweighted: StatSummary = field(default_factory=StatSummary)

    # grouped vs ungrouped neuron summaries
    kin_grouped: StatSummary = field(default_factory=StatSummary)
    kin_ungrouped: StatSummary = field(default_factory=StatSummary)
    freq_grouped: StatSummary = field(default_factory=StatSummary)
    freq_ungrouped: StatSummary = field(default_factory=StatSummary)


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
        n_groups = {
            name: len(r.groups)
            for name, r in video.grouping_results.items()
        }

        # Per-strategy group size and mean correlation stats
        group_stats: dict[str, dict[str, float]] = {}
        for name, r in video.grouping_results.items():
            sizes = [g.size for g in r.groups] if r.groups else []
            corrs = [
                g.group_mean_similarity(r.matrix)
                for g in r.groups
            ] if r.groups and r.matrix is not None else []
            # Filter out NaN correlations (groups with < 2 members)
            corrs = [c for c in corrs if np.isfinite(c)]
            # Mean total spikes per group
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

        # Leaf summaries from per-neuron summary_df:
        # - kinetics: unweighted + spike-weighted (number_of_spikes)
        # - frequency: unweighted (never spike-weighted)
        kin_unw, kin_wspk, freq_unw = combine_neuron_level_to_video(
            video.summary_df,
            spike_count_col="number_of_spikes",
            spike_freq_col="spike_frequency",
        )

        # --- grouped vs ungrouped neuron partition ---
        grouped_indices: set[int] = set()
        for result in video.grouping_results.values():
            for group in result.groups:
                grouped_indices.update(getattr(group, "neuron_indices", []))

        summary_df = video.summary_df
        if not summary_df.empty and grouped_indices:
            mask = summary_df.index.isin(grouped_indices)
            grouped_df = summary_df.loc[mask]
            ungrouped_df = summary_df.loc[~mask]
        else:
            grouped_df = pd.DataFrame()
            ungrouped_df = summary_df

        n_neurons_grouped = len(grouped_df)
        n_neurons_ungrouped = len(ungrouped_df)

        kin_grp, _, freq_grp = combine_neuron_level_to_video(
            grouped_df,
            spike_count_col="number_of_spikes",
            spike_freq_col="spike_frequency",
        )
        kin_ungrp, _, freq_ungrp = combine_neuron_level_to_video(
            ungrouped_df,
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
            n_neurons_grouped=n_neurons_grouped,
            n_neurons_ungrouped=n_neurons_ungrouped,
            n_groups=n_groups,
            group_stats=group_stats,
            kin_unweighted=kin_unw,
            kin_weighted_spikes=kin_wspk,
            freq_unweighted=freq_unw,
            kin_grouped=kin_grp,
            kin_ungrouped=kin_ungrp,
            freq_grouped=freq_grp,
            freq_ungrouped=freq_ungrp,
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
                    node.n_neurons_grouped = node.payload.n_neurons_grouped
                    node.n_neurons_ungrouped = node.payload.n_neurons_ungrouped
                    node.n_groups = node.payload.n_groups
                    node.group_stats = node.payload.group_stats
                    # For comparisons, define:
                    # - kin_unweighted: unweighted across neurons inside the video
                    # - kin_weighted: spike-weighted across neurons inside the video
                    node.kin_unweighted = node.payload.kin_unweighted
                    node.kin_weighted = node.payload.kin_weighted_spikes

                    # Frequency: only unweighted at neuron level
                    node.freq_unweighted = node.payload.freq_unweighted
                    node.freq_weighted = node.payload.freq_unweighted  # same at leaf

                    # Grouped vs ungrouped
                    node.kin_grouped = node.payload.kin_grouped
                    node.kin_ungrouped = node.payload.kin_ungrouped
                    node.freq_grouped = node.payload.freq_grouped
                    node.freq_ungrouped = node.payload.freq_ungrouped
                return

            # internal node: counts
            node.n_videos = sum(ch.n_videos for ch in node.children.values())
            node.n_neurons = sum(ch.n_neurons for ch in node.children.values())
            node.n_neurons_grouped = sum(ch.n_neurons_grouped for ch in node.children.values())
            node.n_neurons_ungrouped = sum(ch.n_neurons_ungrouped for ch in node.children.values())

            # merge per-strategy group counts
            merged_groups: dict[str, int] = {}
            for ch in node.children.values():
                for method, count in ch.n_groups.items():
                    merged_groups[method] = merged_groups.get(method, 0) + count
            node.n_groups = merged_groups

            # aggregate per-strategy group stats (weighted average by n_groups)
            kids = list(node.children.values())
            all_methods = {m for ch in kids for m in ch.group_stats}
            merged_gstats: dict[str, dict[str, float]] = {}
            for method in all_methods:
                accum: dict[str, list[tuple[float, float]]] = {}  # stat -> [(value, weight)]
                for ch in kids:
                    gs = ch.group_stats.get(method)
                    w = float(ch.n_groups.get(method, 0))
                    if gs is None or w == 0:
                        continue
                    for stat_name, stat_val in gs.items():
                        accum.setdefault(stat_name, []).append((stat_val, w))
                merged_gstats[method] = {}
                for stat_name, pairs in accum.items():
                    total_w = sum(w for _, w in pairs)
                    if total_w > 0:
                        merged_gstats[method][stat_name] = sum(v * w for v, w in pairs) / total_w
                    else:
                        merged_gstats[method][stat_name] = 0.0
            node.group_stats = merged_gstats

            # Unweighted: each immediate child counts equally

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

            # Grouped vs ungrouped: same weighting logic
            if children_are_videos:
                wg = [(ch.kin_grouped, float(ch.n_neurons)) for ch in kids]
                wug = [(ch.kin_ungrouped, float(ch.n_neurons)) for ch in kids]
                wfg = [(ch.freq_grouped, float(ch.n_neurons)) for ch in kids]
                wfug = [(ch.freq_ungrouped, float(ch.n_neurons)) for ch in kids]
            else:
                wg = [(ch.kin_grouped, float(ch.n_videos)) for ch in kids]
                wug = [(ch.kin_ungrouped, float(ch.n_videos)) for ch in kids]
                wfg = [(ch.freq_grouped, float(ch.n_videos)) for ch in kids]
                wfug = [(ch.freq_ungrouped, float(ch.n_videos)) for ch in kids]

            node.kin_grouped = aggregate_children(wg)
            node.kin_ungrouped = aggregate_children(wug)
            node.freq_grouped = aggregate_children(wfg)
            node.freq_ungrouped = aggregate_children(wfug)

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
            }
            # One column per grouping strategy
            for method in sorted(child.n_groups):
                row[f"n_groups_{method}"] = child.n_groups[method]

            # Per-strategy group size and correlation stats
            for method in sorted(child.group_stats):
                gs = child.group_stats[method]
                row[f"mean_group_size_{method}"] = gs.get("mean_group_size", 0.0)
                row[f"median_group_size_{method}"] = gs.get("median_group_size", 0.0)
                row[f"mean_group_corr_{method}"] = gs.get("mean_group_corr", 0.0)
                row[f"mean_spikes_per_group_{method}"] = gs.get("mean_spikes_per_group", 0.0)

            # Fraction of neurons grouped vs ungrouped
            total = child.n_neurons_grouped + child.n_neurons_ungrouped
            row["frac_grouped"] = child.n_neurons_grouped / total if total > 0 else 0.0
            row["frac_ungrouped"] = child.n_neurons_ungrouped / total if total > 0 else 0.0

            # Flatten kinetics (unweighted + weighted + grouped + ungrouped)
            for summary, scheme in [
                (child.kin_unweighted, "unweighted"),
                (child.kin_weighted, "weighted"),
                (child.kin_grouped, "grouped"),
                (child.kin_ungrouped, "ungrouped"),
            ]:
                for stat in sorted(summary.means):
                    row[f"{stat}_mean_{scheme}"] = summary.means[stat]
                    row[f"{stat}_var_{scheme}"] = summary.vars_total.get(stat, 0.0)
                    row[f"{stat}_within_{scheme}"] = summary.vars_within.get(stat, 0.0)
                    row[f"{stat}_between_{scheme}"] = summary.vars_between.get(stat, 0.0)

            # Flatten frequency (unweighted + weighted + grouped + ungrouped)
            for summary, scheme in [
                (child.freq_unweighted, "unweighted"),
                (child.freq_weighted, "weighted"),
                (child.freq_grouped, "grouped"),
                (child.freq_ungrouped, "ungrouped"),
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

        df = pd.DataFrame(rows).sort_values("child")

        # Reorder columns: core identifiers first, then all means, then
        # variance columns (var/within/between) — so the most-compared
        # values are immediately visible.
        core = [c for c in df.columns if c in ("child", "n_videos", "n_neurons")
                or c.startswith("n_groups_") or c.startswith("frac_")
                or c.startswith("mean_group_size_") or c.startswith("median_group_size_")
                or c.startswith("mean_group_corr_") or c.startswith("mean_spikes_per_group_")]
        rest = [c for c in df.columns if c not in core]
        mean_cols = [c for c in rest if "_mean_" in c]
        var_cols = [c for c in rest if c not in mean_cols]
        df = df[core + mean_cols + var_cols]

        return df
