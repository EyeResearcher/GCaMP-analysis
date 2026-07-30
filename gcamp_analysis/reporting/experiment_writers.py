"""Writers for experiment-level comparison tables and section outputs.

This module owns filesystem and plotting side effects for processed experiment
trees. The ``experiments`` package computes ``NodeSummary`` objects and sibling
comparison DataFrames; reporting code writes those completed results to disk.

Adding experiment output
------------------------
Add scientific values and aggregation rules in ``gcamp_analysis.experiments``.
Once the completed value exists on a ``NodeSummary`` or comparison DataFrame,
add its serialization here. Legend descriptions belong here because they
describe exported columns rather than aggregation semantics.

The functions in ``gcamp_analysis.experiments.io`` are compatibility
re-exports. New code should import these writers from
``gcamp_analysis.reporting``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gcamp_analysis.experiments.tree import TreeNode
from utils.visualization import (
    plot_delta_corr_vs_dispersion,
    plot_neuron_centroid_distances,
)


_KIND_DEFS = {
    "mean": "Mean value.",
    "var": "Total variance (within + between).",
    "within": "Within-child variance component.",
    "between": "Between-child variance component.",
}
_SCHEME_DEFS = {
    "unweighted": "Each immediate child weighted equally.",
    "weighted": (
        "Children weighted by n_neurons (video level) or n_videos "
        "(higher levels)."
    ),
    "grouped": "Neurons belonging to at least one neuron group.",
    "ungrouped": "Neurons not belonging to any neuron group.",
}
_CORE_COLS = {
    "child": "Name of the child node (one row per sibling).",
    "n_videos": "Number of videos under this child.",
    "n_neurons": "Total neurons under this child.",
}


def build_comparison_legend(comparison: pd.DataFrame) -> pd.DataFrame:
    """Build descriptions for the columns in a sibling comparison table."""
    rows = []
    for column in comparison.columns:
        if column in _CORE_COLS:
            rows.append(
                {"column": column, "description": _CORE_COLS[column]}
            )
            continue
        if column.startswith("n_groups_"):
            strategy = column[len("n_groups_"):]
            rows.append(
                {
                    "column": column,
                    "description": (
                        "Number of neuron groups found by the "
                        f"'{strategy}' strategy."
                    ),
                }
            )
            continue
        if column.startswith("mean_group_size_"):
            strategy = column[len("mean_group_size_"):]
            rows.append(
                {
                    "column": column,
                    "description": (
                        f"Mean neuron-group size for the '{strategy}' "
                        "strategy."
                    ),
                }
            )
            continue
        if column.startswith("median_group_size_"):
            strategy = column[len("median_group_size_"):]
            rows.append(
                {
                    "column": column,
                    "description": (
                        f"Median neuron-group size for the '{strategy}' "
                        "strategy."
                    ),
                }
            )
            continue
        if column.startswith("mean_group_corr_"):
            strategy = column[len("mean_group_corr_"):]
            rows.append(
                {
                    "column": column,
                    "description": (
                        "Mean within-group pairwise correlation for the "
                        f"'{strategy}' strategy."
                    ),
                }
            )
            continue
        if column == "frac_grouped":
            rows.append(
                {
                    "column": column,
                    "description": (
                        "Fraction of neurons belonging to at least one "
                        "neuron group."
                    ),
                }
            )
            continue
        if column == "frac_ungrouped":
            rows.append(
                {
                    "column": column,
                    "description": (
                        "Fraction of neurons not belonging to any neuron "
                        "group."
                    ),
                }
            )
            continue

        parts = column.rsplit("_", 2)
        if len(parts) == 3:
            _, kind, scheme = parts
            kind_description = _KIND_DEFS.get(kind)
            scheme_description = _SCHEME_DEFS.get(scheme)
            if kind_description and scheme_description:
                rows.append(
                    {
                        "column": column,
                        "description": (
                            f"{kind_description} {scheme_description}"
                        ),
                    }
                )
                continue

        rows.append({"column": column, "description": ""})

    return pd.DataFrame(rows)


def save_comparisons(
    *,
    root: TreeNode,
    sibling_tables: dict[Path, pd.DataFrame],
    output_subdir: str = "metrics",
    filename: str = "sibling_comparisons.xlsx",
) -> None:
    """Write each sibling comparison table and its legend to Excel."""
    del root  # Retained for API compatibility with existing callers.

    for node_path, comparison in sibling_tables.items():
        if comparison is None or comparison.empty:
            continue

        out_dir = Path(node_path) / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        legend = build_comparison_legend(comparison)
        with pd.ExcelWriter(out_dir / filename, engine="openpyxl") as writer:
            comparison.to_excel(
                writer,
                index=False,
                sheet_name="summary",
            )
            legend.to_excel(writer, index=False, sheet_name="legend")

