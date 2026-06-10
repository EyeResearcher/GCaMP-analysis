"""Pure formatting utilities for sibling experiment comparisons.

This module converts named ``NodeSummary`` objects into the flat tables used
for sibling comparison exports. It does not traverse trees, run processing, or
mutate summaries. ``ExperimentProcessor.compare_siblings`` owns traversal and
delegates table construction here.

Adding a statistic to comparison output
---------------------------------------
Statistics stored as keys inside any existing ``StatSummary`` are exported
automatically by ``flatten_stat_summary``. Their columns follow:

``<stat>_<mean|var|within|between>_<scheme>``

where ``scheme`` is ``unweighted``, ``weighted``, ``grouped``, or
``ungrouped``.

New structural ``NodeSummary`` fields are not exported automatically. Add
their row representation to ``summary_to_comparison_row`` and, if necessary,
classify their columns in ``order_comparison_columns``. Keep aggregation
semantics in ``summary_utils.py``; this module only formats already-computed
values.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from gcamp_analysis.experiments.summary_utils import NodeSummary, StatSummary


def build_sibling_comparison(
    children: Iterable[tuple[str, NodeSummary]],
) -> pd.DataFrame | None:
    """Build a comparison table for named child summaries with data."""
    rows = [
        summary_to_comparison_row(name, summary)
        for name, summary in children
        if summary.n_videos > 0
    ]
    if len(rows) < 2:
        return None

    comparison = pd.DataFrame(rows).sort_values("child")
    return order_comparison_columns(comparison)


def summary_to_comparison_row(
    name: str,
    summary: NodeSummary,
) -> dict[str, object]:
    """Flatten one child summary into a comparison-table row."""
    row: dict[str, object] = {
        "child": name,
        "n_videos": summary.n_videos,
        "n_neurons": summary.n_neurons,
    }
    row.update(_flatten_group_counts(summary))
    row.update(_flatten_group_stats(summary))
    row.update(_grouping_fractions(summary))

    schemes = (
        (summary.kin_unweighted, "unweighted"),
        (summary.kin_weighted, "weighted"),
        (summary.kin_grouped, "grouped"),
        (summary.kin_ungrouped, "ungrouped"),
        (summary.freq_unweighted, "unweighted"),
        (summary.freq_weighted, "weighted"),
        (summary.freq_grouped, "grouped"),
        (summary.freq_ungrouped, "ungrouped"),
    )
    for stat_summary, scheme in schemes:
        row.update(flatten_stat_summary(stat_summary, scheme))

    return row


def flatten_stat_summary(
    summary: StatSummary,
    scheme: str,
) -> dict[str, float]:
    """Flatten all keyed statistics in one summary for an output scheme."""
    values: dict[str, float] = {}
    for stat in sorted(summary.means):
        values[f"{stat}_mean_{scheme}"] = summary.means[stat]
        values[f"{stat}_var_{scheme}"] = summary.vars_total.get(stat, 0.0)
        values[f"{stat}_within_{scheme}"] = summary.vars_within.get(stat, 0.0)
        values[f"{stat}_between_{scheme}"] = summary.vars_between.get(stat, 0.0)
    return values


def order_comparison_columns(comparison: pd.DataFrame) -> pd.DataFrame:
    """Place identifiers and primary means before variance components."""
    core = [
        column
        for column in comparison.columns
        if column in ("child", "n_videos", "n_neurons")
        or column.startswith("n_groups_")
        or column.startswith("frac_")
        or column.startswith("mean_group_size_")
        or column.startswith("median_group_size_")
        or column.startswith("mean_group_corr_")
        or column.startswith("mean_spikes_per_group_")
    ]
    remaining = [column for column in comparison.columns if column not in core]
    means = [column for column in remaining if "_mean_" in column]
    variances = [column for column in remaining if column not in means]
    return comparison[core + means + variances]


def _flatten_group_counts(summary: NodeSummary) -> dict[str, int]:
    return {
        f"n_groups_{strategy}": summary.n_groups[strategy]
        for strategy in sorted(summary.n_groups)
    }


def _flatten_group_stats(summary: NodeSummary) -> dict[str, float]:
    values: dict[str, float] = {}
    for strategy in sorted(summary.group_stats):
        stats = summary.group_stats[strategy]
        values[f"mean_group_size_{strategy}"] = stats.get(
            "mean_group_size", 0.0
        )
        values[f"median_group_size_{strategy}"] = stats.get(
            "median_group_size", 0.0
        )
        values[f"mean_group_corr_{strategy}"] = stats.get(
            "mean_group_corr", 0.0
        )
        values[f"mean_spikes_per_group_{strategy}"] = stats.get(
            "mean_spikes_per_group", 0.0
        )

        if strategy == "light-evoked":
            values["total_ON_cells"] = stats.get("total_ON_cells", 0)
            values["total_OFF_cells"] = stats.get("total_OFF_cells", 0)
            values.update(
                {
                    key: value
                    for key, value in stats.items()
                    if key.startswith("n_cells_")
                }
            )
    return values


def _grouping_fractions(summary: NodeSummary) -> dict[str, float]:
    total = summary.n_neurons_grouped + summary.n_neurons_ungrouped
    if total <= 0:
        return {"frac_grouped": 0.0, "frac_ungrouped": 0.0}
    return {
        "frac_grouped": summary.n_neurons_grouped / total,
        "frac_ungrouped": summary.n_neurons_ungrouped / total,
    }
