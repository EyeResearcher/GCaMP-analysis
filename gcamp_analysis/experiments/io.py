"""Save sibling comparison tables to Excel with an auto-generated legend."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gcamp_analysis.experiments.tree import TreeNode
from utils.visualization import plot_delta_corr_vs_dispersion, plot_neuron_centroid_distances


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
    rows = []
    for col in df.columns:
        if col in _CORE_COLS:
            rows.append({"column": col, "description": _CORE_COLS[col]})
            continue
        if col.startswith("n_groups_"):
            method = col[len("n_groups_"):]
            rows.append({"column": col, "description": f"Number of neuron groups found by the '{method}' strategy."})
            continue
        if col.startswith("mean_group_size_"):
            method = col[len("mean_group_size_"):]
            rows.append({"column": col, "description": f"Mean neuron-group size for the '{method}' strategy."})
            continue
        if col.startswith("median_group_size_"):
            method = col[len("median_group_size_"):]
            rows.append({"column": col, "description": f"Median neuron-group size for the '{method}' strategy."})
            continue
        if col.startswith("mean_group_corr_"):
            method = col[len("mean_group_corr_"):]
            rows.append({"column": col, "description": f"Mean within-group pairwise correlation for the '{method}' strategy."})
            continue
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
    for node_path, df in sibling_tables.items():
        if df is None or df.empty:
            continue

        out_dir = Path(node_path) / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        legend_df = _build_legend(df)
        with pd.ExcelWriter(out_dir / filename, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="summary")
            legend_df.to_excel(writer, index=False, sheet_name="legend")


def save_section_comparisons(
    root: TreeNode,
    output_subdir: str = "metrics",
) -> None:
    """Save aggregate section comparison DataFrames at every tree node."""
    for node in root.iter_nodes():
        comparison_dfs = getattr(node, "section_comparison_df", {})
        if not comparison_dfs:
            continue

        out_dir = node.path / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        for strategy_name, strategy_dfs in comparison_dfs.items():
            for section_key, df in strategy_dfs.items():
                path = out_dir / f"{node.name}_{strategy_name}_{section_key}_section_comparison.csv"
                df.to_csv(path, index=False)

        comparison_metrics = getattr(node, "section_comparison_metrics", {})
        for strategy_name, strategy_metrics in comparison_metrics.items():
            for section_key, metrics in strategy_metrics.items():
                if not metrics:
                    continue
                title_prefix = f"{node.name} - {strategy_name} - {section_key}"

                fig1, ax1 = plt.subplots(figsize=(7, 5))
                plot_delta_corr_vs_dispersion(metrics, ax=ax1, title=title_prefix)
                fig1.tight_layout()
                fig1.savefig(
                    out_dir / f"{node.name}_{strategy_name}_{section_key}_delta_corr_vs_dispersion.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig1)

                fig2, ax2 = plt.subplots(figsize=(7, 6))
                plot_neuron_centroid_distances(metrics, ax=ax2, title=title_prefix)
                fig2.tight_layout()
                fig2.savefig(
                    out_dir / f"{node.name}_{strategy_name}_{section_key}_centroid_distances.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig2)


save_treatment_comparisons = save_section_comparisons
