"""Save sibling comparison tables to Excel with an auto-generated legend."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from gcamp_analysis.experiments.tree import TreeNode
from utils.visualization import plot_delta_corr_vs_dispersion, plot_neuron_centroid_distances


# Column suffixes that the legend explains automatically.
_KIND_DEFS = {
    "mean": "Mean value.",
    "var": "Total variance (within + between).",
    "within": "Within-child variance component.",
    "between": "Between-child variance component.",
}
_SCHEME_DEFS = {
    "unweighted": "Each immediate child weighted equally.",
    "weighted": "Children weighted by n_neurons (video level) or n_videos (higher levels).",
    "grouped": "Neurons belonging to at least one neuron group.",
    "ungrouped": "Neurons not belonging to any neuron group.",
}
_CORE_COLS = {
    "child": "Name of the child node (one row per sibling).",
    "n_videos": "Number of videos under this child.",
    "n_neurons": "Total neurons under this child.",
}


def _build_legend(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-generate a legend from column naming convention.

    Expects columns like ``{stat}_{kind}_{scheme}`` where *kind* is one
    of mean/var/within/between and *scheme* is unweighted/weighted.
    """
    rows = []
    for col in df.columns:
        if col in _CORE_COLS:
            rows.append({"column": col, "description": _CORE_COLS[col]})
            continue

        # Per-strategy group count columns (n_groups_corr, etc.)
        if col.startswith("n_groups_"):
            method = col[len("n_groups_"):]
            rows.append({"column": col, "description": f"Number of neuron groups found by the '{method}' strategy."})
            continue

        # Per-strategy group size stats
        if col.startswith("mean_group_size_"):
            method = col[len("mean_group_size_"):]
            rows.append({"column": col, "description": f"Mean neuron-group size (number of neurons per group) for the '{method}' strategy."})
            continue
        if col.startswith("median_group_size_"):
            method = col[len("median_group_size_"):]
            rows.append({"column": col, "description": f"Median neuron-group size for the '{method}' strategy."})
            continue

        # Per-strategy mean within-group correlation
        if col.startswith("mean_group_corr_"):
            method = col[len("mean_group_corr_"):]
            rows.append({"column": col, "description": f"Mean within-group pairwise correlation (averaged across groups) for the '{method}' strategy."})
            continue

        # Fraction grouped/ungrouped
        if col == "frac_grouped":
            rows.append({"column": col, "description": "Fraction of neurons belonging to at least one neuron group."})
            continue
        if col == "frac_ungrouped":
            rows.append({"column": col, "description": "Fraction of neurons not belonging to any neuron group."})
            continue

        parts = col.rsplit("_", 2)
        if len(parts) == 3:
            stat, kind, scheme = parts
            kind_desc = _KIND_DEFS.get(kind)
            scheme_desc = _SCHEME_DEFS.get(scheme)
            if kind_desc and scheme_desc:
                rows.append({"column": col, "description": f"{kind_desc} {scheme_desc}"})
                continue

        rows.append({"column": col, "description": ""})

    return pd.DataFrame(rows)


def save_comparisons(
    *,
    root: TreeNode,
    sibling_tables: dict[Path, pd.DataFrame],
    output_subdir: str = "metrics",
    filename: str = "sibling_comparisons.xlsx",
) -> None:
    """Write one Excel workbook per internal node.

    Parameters
    ----------
    root : TreeNode
        Root of the experiment tree.
    sibling_tables : dict[Path, DataFrame]
        Output of ``ExperimentProcessor.compare_siblings``.
    output_subdir : str, optional
        Sub-directory under each node for output (default ``'metrics'``).
    filename : str, optional
        Workbook filename (default ``'sibling_comparisons.xlsx'``).
    """
    for node_path, df in sibling_tables.items():
        if df is None or df.empty:
            continue

        out_dir = Path(node_path) / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        legend_df = _build_legend(df)

        with pd.ExcelWriter(out_dir / filename, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="summary")
            legend_df.to_excel(writer, index=False, sheet_name="legend")


def save_treatment_comparisons(
    root: TreeNode,
    output_subdir: str = "metrics",
) -> None:
    """Save aggregate treatment comparison DataFrames at every tree node.

    At each node that has ``treatment_comparison_df``, writes one CSV
    per strategy containing the concatenated per-group metrics from all
    descendant videos.  Also saves delta-correlation vs. dispersion and
    centroid-distance figures when raw group metrics are available.
    """
    for node in root.iter_nodes():
        tc_dfs = getattr(node, "treatment_comparison_df", {})
        if not tc_dfs:
            continue
        out_dir = node.path / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for strat, df in tc_dfs.items():
            path = out_dir / f"{node.name}_{strat}_treatment_comparison.csv"
            df.to_csv(path, index=False)

        # Generate treatment comparison figures from raw group metrics
        tc_metrics = getattr(node, "treatment_comparison_metrics", {})
        for strat, metrics in tc_metrics.items():
            if not metrics:
                continue
            title_prefix = f"{node.name} \u2014 {strat}"

            fig1, ax1 = plt.subplots(figsize=(7, 5))
            plot_delta_corr_vs_dispersion(metrics, ax=ax1, title=title_prefix)
            fig1.tight_layout()
            fig1.savefig(out_dir / f"{node.name}_{strat}_delta_corr_vs_dispersion.png",
                         dpi=150, bbox_inches="tight")
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(7, 6))
            plot_neuron_centroid_distances(metrics, ax=ax2, title=title_prefix)
            fig2.tight_layout()
            fig2.savefig(out_dir / f"{node.name}_{strat}_centroid_distances.png",
                         dpi=150, bbox_inches="tight")
            plt.close(fig2)
