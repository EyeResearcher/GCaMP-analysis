"""Save sibling comparison tables to Excel with an auto-generated legend."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gcamp_analysis.experiments.tree import TreeNode


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
