from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from experiments.tree import TreeNode


def _infer_weighting_basis_for_node(node: TreeNode) -> str:
    """
    Matches your aggregation rule in processor:
      - if node's immediate children are videos (leaf nodes) => weighted by n_neurons
      - else => weighted by n_videos
    """
    kids = list(node.children.values())
    if not kids:
        return "n/a"
    children_are_videos = all(ch.is_leaf() for ch in kids)
    return "n_neurons" if children_are_videos else "n_videos"


def _build_legend(df: pd.DataFrame, weighting_basis: str) -> pd.DataFrame:
    rows: List[dict] = []

    # --- Node-level note about what "weighted" means on THIS sheet
    if weighting_basis == "n_neurons":
        weighted_note = (
            "Weighted columns on this sheet: each child is a video; weights = child n_neurons. "
            "Interpretation: weighted values approximate an average over neurons across videos."
        )
    elif weighting_basis == "n_videos":
        weighted_note = (
            "Weighted columns on this sheet: each child is a group of videos (e.g., timepoint); "
            "weights = child n_videos. Interpretation: weighted values approximate an average over videos "
            "across the compared groups."
        )
    else:
        weighted_note = "Weighted columns on this sheet: not applicable."

    rows.append({"column_name": "(node)", "meaning": weighted_note})

    # --- Core columns
    core = {
        "parent": "Filesystem path of the parent node whose children are being compared.",
        "child": "Name of the child node (one row per sibling).",
        "n_videos": "Total number of videos under this child node.",
        "n_neurons": "Total number of neurons under this child node.",
    }
    for col, meaning in core.items():
        if col in df.columns:
            rows.append({"column_name": col, "meaning": meaning})

    # --- Shared definitions for variance decomposition
    # total = within + between
    base_defs = {
        "within": (
            "Within-child component. Computed as the (weighted) mean of child TOTAL variances. "
            "This captures variability internal to each child."
        ),
        "between": (
            "Between-child component. Computed as the (weighted) variance of child means. "
            "This captures heterogeneity across children."
        ),
        "var": (
            "Total variance. Defined as within + between (law of total variance)."
        ),
    }

    def _meaning_for_suffix(stat: str, kind: str, scheme: str) -> str:
        # kind in {"mean","var","within","between"}, scheme in {"unweighted","weighted"}
        if scheme == "unweighted":
            scheme_note = "Unweighted across immediate children (each child counts equally)."
        else:
            scheme_note = f"Weighted across immediate children using {weighting_basis} (see node note above)."

        if kind == "mean":
            return f"{scheme_note} Mean of '{stat}'."
        if kind == "var":
            return f"{scheme_note} {base_defs['var']}"
        if kind == "within":
            return f"{scheme_note} {base_defs['within']}"
        if kind == "between":
            return f"{scheme_note} {base_defs['between']}"
        return "Metric column."

    # --- Parse column naming produced by your compare.py
    # Examples:
    #   tau_decay_mean_unweighted
    #   tau_decay_var_weighted
    #   tau_decay_within_weighted
    #   tau_decay_between_unweighted
    #   spike_frequency_mean_unweighted  (special stat name with underscore)
    for col in df.columns:
        if col in core:
            continue
        if col in ("parent", "child"):
            continue

        # Handle spike_frequency explicitly (because stat name contains underscore)
        if col.startswith("spike_frequency_"):
            # expected: spike_frequency_{kind}_{scheme}
            parts = col.split("_")
            # ["spike", "frequency", kind, scheme] -> kind=parts[2], scheme=parts[3]
            if len(parts) >= 4:
                kind = parts[2]
                scheme = parts[3]
                rows.append({"column_name": col, "meaning": _meaning_for_suffix("spike_frequency", kind, scheme)})
            else:
                rows.append({"column_name": col, "meaning": "Spike frequency metric."})
            continue

        # General case: {stat}_{kind}_{scheme} where kind ∈ mean/var/within/between
        # Find the last two tokens (kind, scheme)
        parts = col.split("_")
        if len(parts) < 3:
            rows.append({"column_name": col, "meaning": "Metric column."})
            continue

        kind = parts[-2]
        scheme = parts[-1]
        if kind not in {"mean", "var", "within", "between"} or scheme not in {"unweighted", "weighted"}:
            rows.append({"column_name": col, "meaning": "Metric column (see naming suffix)."})
            continue

        stat = "_".join(parts[:-2])
        rows.append({"column_name": col, "meaning": _meaning_for_suffix(stat, kind, scheme)})

    return pd.DataFrame(rows)


def save_node_level_comparisons_with_legend(
    *,
    root: TreeNode,
    sibling_tables: Dict[Path, pd.DataFrame],
    output_subdir: str = "metrics",
    filename: str = "sibling_comparisons.xlsx",
) -> None:
    path_to_node: Dict[Path, TreeNode] = {n.path: n for n in root.iter_nodes()}

    for node_path, df in sibling_tables.items():
        if df is None or df.empty:
            continue

        node = path_to_node.get(Path(node_path))
        weighting_basis = _infer_weighting_basis_for_node(node) if node is not None else "n/a"

        out_dir = Path(node_path) / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

        legend_df = _build_legend(df, weighting_basis=weighting_basis)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="summary")
            legend_df.to_excel(writer, index=False, sheet_name="legend")
