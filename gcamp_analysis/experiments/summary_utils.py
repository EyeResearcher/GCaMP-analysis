"""Summary data structures and pure hierarchical aggregation functions.

This module owns the statistical semantics of experiment-tree aggregation.
It has no dependency on ``TreeNode`` or ``ExperimentProcessor`` and should
remain usable with plain dataclass instances in unit tests.

Data flows through three levels:

``per-neuron DataFrame -> VideoRunRecord -> NodeSummary -> parent NodeSummary``

``StatSummary`` stores keyed means and a within/between variance decomposition.
``NodeSummary`` stores all values attached to one processed tree node.
``summary_from_video_record`` converts a video leaf into that common form.
``aggregate_node_summaries`` combines child summaries into a parent.
``comparison_utils.py`` separately owns flattening completed summaries into
export tables; output formatting should not be added to this module.

Adding an aggregated statistic
------------------------------
There are two common cases:

* A new kinetics-like scalar with per-neuron ``mean_<name>`` and optionally
  ``var_<name>`` columns usually needs no structural change. ``summarize_video``
  discovers the name, and ``aggregate_children`` propagates it through each
  ``StatSummary`` dictionary.
* A new count, table, category, or independently weighted summary must be added
  to ``NodeSummary``. It must then be initialized in
  ``summary_from_video_record`` and explicitly combined in
  ``aggregate_node_summaries``. Add a small private helper when the merge rule
  is more complex than a sum or a call to ``aggregate_children``.

Choose weights intentionally. Current video-child rules use total neuron count
for ordinary weighted summaries, grouped neuron count for grouped summaries,
and ungrouped neuron count for ungrouped summaries. Higher-level children use
video count. Changing these rules changes the meaning of exported statistics
and requires focused tests.

Every ``StatSummary`` statistic tracks its mean and total variance, with total
variance decomposed using the law of total variance into within-child and
between-child components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import pandas as pd

if TYPE_CHECKING:
    from gcamp_analysis.experiments.models import VideoRunRecord


@dataclass
class StatSummary:
    """Variance-decomposed summary for an arbitrary set of statistics.

    Attributes
    ----------
    means : dict[str, float]
        Weighted mean for each statistic.
    vars_total : dict[str, float]
        Total variance (within + between).
    vars_within : dict[str, float]
        Within-child component: :math:`E_w[\\text{child var}]`.
    vars_between : dict[str, float]
        Between-child component: :math:`\\text{Var}_w(\\text{child mean})`.
    """
    means: dict[str, float] = field(default_factory=dict)
    vars_total: dict[str, float] = field(default_factory=dict)
    vars_within: dict[str, float] = field(default_factory=dict)
    vars_between: dict[str, float] = field(default_factory=dict)

    # Backwards-compat shim: existing code expects ``.vars``
    @property
    def vars(self) -> dict[str, float]:
        """Alias for ``vars_total``."""
        return self.vars_total


@dataclass
class NodeSummary:
    """All counts, statistics, and detail tables summarized by a tree node."""

    n_videos: int = 0
    n_neurons: int = 0
    n_neurons_grouped: int = 0
    n_neurons_ungrouped: int = 0

    n_groups: dict[str, int] = field(default_factory=dict)
    group_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    kin_unweighted: StatSummary = field(default_factory=StatSummary)
    kin_weighted: StatSummary = field(default_factory=StatSummary)
    freq_unweighted: StatSummary = field(default_factory=StatSummary)
    freq_weighted: StatSummary = field(default_factory=StatSummary)

    kin_grouped: StatSummary = field(default_factory=StatSummary)
    kin_ungrouped: StatSummary = field(default_factory=StatSummary)
    freq_grouped: StatSummary = field(default_factory=StatSummary)
    freq_ungrouped: StatSummary = field(default_factory=StatSummary)

    light_evoked_details: dict[str, pd.DataFrame] = field(default_factory=dict)


def summary_from_video_record(
    record: VideoRunRecord,
    *,
    source: str,
) -> NodeSummary:
    """Convert one processed video record into a leaf-node summary."""
    return NodeSummary(
        n_videos=1,
        n_neurons=record.n_neurons,
        n_neurons_grouped=record.n_neurons_grouped,
        n_neurons_ungrouped=record.n_neurons_ungrouped,
        n_groups=dict(record.n_groups_per_strategy),
        group_stats=record.group_stats,
        kin_unweighted=record.kin_unweighted,
        kin_weighted=record.kin_weighted_spikes,
        freq_unweighted=record.freq_unweighted,
        # Frequency currently has only one per-video summary.
        freq_weighted=record.freq_unweighted,
        kin_grouped=record.kin_grouped,
        kin_ungrouped=record.kin_ungrouped,
        freq_grouped=record.freq_grouped,
        freq_ungrouped=record.freq_ungrouped,
        light_evoked_details={
            key: df.assign(source=source)
            for key, df in record.light_evoked_details.items()
        },
    )


def aggregate_node_summaries(
    children: Iterable[NodeSummary],
    *,
    children_are_videos: bool,
) -> NodeSummary:
    """Return a parent summary computed solely from child summaries."""
    children = list(children)
    if not children:
        return NodeSummary()

    standard_weights = [
        float(child.n_neurons if children_are_videos else child.n_videos)
        for child in children
    ]
    grouped_weights = [
        float(child.n_neurons_grouped if children_are_videos else child.n_videos)
        for child in children
    ]
    ungrouped_weights = [
        float(child.n_neurons_ungrouped if children_are_videos else child.n_videos)
        for child in children
    ]

    return NodeSummary(
        n_videos=sum(child.n_videos for child in children),
        n_neurons=sum(child.n_neurons for child in children),
        n_neurons_grouped=sum(child.n_neurons_grouped for child in children),
        n_neurons_ungrouped=sum(child.n_neurons_ungrouped for child in children),
        n_groups=_sum_group_counts(children),
        group_stats=_aggregate_group_stats(children),
        kin_unweighted=aggregate_children(
            (child.kin_weighted, 1.0) for child in children
        ),
        kin_weighted=aggregate_children(
            zip((child.kin_weighted for child in children), standard_weights)
        ),
        freq_unweighted=aggregate_children(
            (child.freq_weighted, 1.0) for child in children
        ),
        freq_weighted=aggregate_children(
            zip((child.freq_weighted for child in children), standard_weights)
        ),
        kin_grouped=aggregate_children(
            zip((child.kin_grouped for child in children), grouped_weights)
        ),
        kin_ungrouped=aggregate_children(
            zip((child.kin_ungrouped for child in children), ungrouped_weights)
        ),
        freq_grouped=aggregate_children(
            zip((child.freq_grouped for child in children), grouped_weights)
        ),
        freq_ungrouped=aggregate_children(
            zip((child.freq_ungrouped for child in children), ungrouped_weights)
        ),
        light_evoked_details=_merge_dataframes(
            child.light_evoked_details for child in children
        ),
    )


def _sum_group_counts(children: Iterable[NodeSummary]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for child in children:
        for strategy, count in child.n_groups.items():
            totals[strategy] = totals.get(strategy, 0) + count
    return totals


def _aggregate_group_stats(
    children: Iterable[NodeSummary],
) -> dict[str, dict[str, float]]:
    children = list(children)
    strategies = {
        strategy
        for child in children
        for strategy in child.group_stats
    }
    aggregated: dict[str, dict[str, float]] = {}

    for strategy in strategies:
        weighted_values: dict[str, list[tuple[float, float]]] = {}
        for child in children:
            stats = child.group_stats.get(strategy)
            weight = float(child.n_groups.get(strategy, 0))
            if stats is None or weight <= 0:
                continue
            for name, value in stats.items():
                weighted_values.setdefault(name, []).append((value, weight))

        aggregated[strategy] = {
            name: sum(value * weight for value, weight in values)
            / sum(weight for _, weight in values)
            for name, values in weighted_values.items()
        }

    return aggregated


def _merge_dataframes(
    mappings: Iterable[dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    grouped: dict[str, list[pd.DataFrame]] = {}
    for mapping in mappings:
        for key, df in mapping.items():
            if df is not None and not df.empty:
                grouped.setdefault(key, []).append(df)
    return {
        key: pd.concat(dataframes, ignore_index=True)
        for key, dataframes in grouped.items()
    }


def extract_stat_bases(summary_df: pd.DataFrame) -> list[str]:
    """Return sorted unique stat names found in *summary_df*.

    Looks for columns matching ``mean_{stat}`` and returns the
    ``{stat}`` portion.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Per-neuron summary table.

    Returns
    -------
    list[str]
    """
    if summary_df is None or summary_df.empty:
        return []
    bases = []
    for c in summary_df.columns:
        if isinstance(c, str) and c.startswith("mean_"):
            bases.append(c[len("mean_"):])
    return sorted(set(bases))


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def summarize_video(
    summary_df: pd.DataFrame,
    *,
    spike_count_col: str = "number_of_spikes",
    spike_freq_col: str = "spike_frequency",
) -> tuple[StatSummary, StatSummary, StatSummary]:
    """Aggregate per-neuron statistics to a single video.

    Parameters
    ----------
    summary_df : pd.DataFrame
        One row per neuron with ``mean_{stat}``, ``var_{stat}``, and
        spike count / frequency columns.
    spike_count_col : str, optional
        Column used as weight for the spike-weighted kinetics summary.
    spike_freq_col : str, optional
        Column holding spike frequency (summarised unweighted only).

    Returns
    -------
    kin_unw : StatSummary
        Unweighted kinetics summary.
    kin_wspk : StatSummary
        Spike-weighted kinetics summary.
    freq_unw : StatSummary
        Unweighted frequency summary.
    """
    kin_unw = StatSummary()
    kin_wspk = StatSummary()
    freq_unw = StatSummary()

    if summary_df is None or summary_df.empty:
        return kin_unw, kin_wspk, freq_unw

    bases = extract_stat_bases(summary_df)

    # spike weights per neuron (fallback to 1s if missing or all 0)
    if spike_count_col in summary_df.columns:
        w = _safe_numeric(summary_df[spike_count_col]).fillna(0.0)
    else:
        w = pd.Series(0.0, index=summary_df.index)

    if float(w.sum()) <= 0.0:
        w = pd.Series(1.0, index=summary_df.index)

    for base in bases:
        mean_col = f"mean_{base}"
        var_col = f"var_{base}"
        if mean_col not in summary_df.columns:
            continue

        m = _safe_numeric(summary_df[mean_col])
        v = _safe_numeric(summary_df[var_col]) if var_col in summary_df.columns else None

        mask = m.notna()
        if not mask.any():
            continue

        m2 = m[mask]
        w2 = w[mask]
        W = float(w2.sum())

        # ----- Unweighted across neurons
        mu_unw = float(m2.mean())
        # between neurons (unweighted) = Var(m2)
        between_unw = float(m2.var(ddof=0)) if len(m2) > 1 else 0.0

        # within neurons (unweighted): mean of var_{stat} if present, else 0
        if v is not None:
            v2 = v[mask].dropna()
            within_unw = float(v2.mean()) if not v2.empty else 0.0
        else:
            within_unw = 0.0

        kin_unw.means[base] = mu_unw
        kin_unw.vars_within[base] = within_unw
        kin_unw.vars_between[base] = between_unw
        kin_unw.vars_total[base] = within_unw + between_unw

        # ----- Spike-weighted across neurons
        if W > 0:
            mu_w = float((m2 * w2).sum() / W)

            # within = E_w[var_i]
            if v is not None:
                v2 = v[mask].dropna()
                wv = w.loc[v2.index]
                Wv = float(wv.sum())
                within_w = float((v2 * wv).sum() / Wv) if Wv > 0 else 0.0
            else:
                within_w = 0.0

            # between = Var_w(mean_i)
            between_w = float(((m2 - mu_w) ** 2 * w2).sum() / W) if W > 0 else 0.0

            kin_wspk.means[base] = mu_w
            kin_wspk.vars_within[base] = within_w
            kin_wspk.vars_between[base] = between_w
            kin_wspk.vars_total[base] = within_w + between_w

    # ----- Frequency: unweighted only
    if spike_freq_col in summary_df.columns:
        f = _safe_numeric(summary_df[spike_freq_col]).dropna()
        if not f.empty:
            mu = float(f.mean())
            between = float(f.var(ddof=0)) if len(f) > 1 else 0.0
            within = 0.0  # no within component for a single per-neuron scalar
            freq_unw.means["spike_frequency"] = mu
            freq_unw.vars_within["spike_frequency"] = within
            freq_unw.vars_between["spike_frequency"] = between
            freq_unw.vars_total["spike_frequency"] = within + between

    return kin_unw, kin_wspk, freq_unw


def aggregate_children(children: Iterable[tuple[StatSummary, float]]) -> StatSummary:
    """Combine child summaries into a parent using the law of total variance.

    Parameters
    ----------
    children : iterable of (StatSummary, float)
        Each pair is ``(child_summary, weight)``.

    Returns
    -------
    StatSummary
        Parent-level summary where:

        * ``mean = \u03a3 w\u00b7\u03bc / \u03a3 w``
        * ``within = \u03a3 w\u00b7var_total_child / \u03a3 w``
        * ``between = \u03a3 w\u00b7(\u03bc_child - \u03bc)\u00b2 / \u03a3 w``
        * ``total = within + between``

    Notes
    -----
    Each child's *total* variance is used as the within term at the
    parent level.
    """
    out = StatSummary()

    items = [(s, float(w)) for s, w in children if s is not None and w is not None and float(w) > 0]
    if not items:
        return out

    stats = set()
    for s, _ in items:
        stats.update(s.means.keys())

    for stat in sorted(stats):
        triples = []
        for s, w in items:
            if stat in s.means and stat in s.vars_total:
                triples.append((s.means[stat], s.vars_total[stat], w))

        if not triples:
            continue

        W = sum(w for _, _, w in triples)
        if W <= 0:
            continue

        mu = sum(w * m for m, _, w in triples) / W
        within = sum(w * v for _, v, w in triples) / W
        between = sum(w * (m - mu) ** 2 for m, _, w in triples) / W

        out.means[stat] = float(mu)
        out.vars_within[stat] = float(within)
        out.vars_between[stat] = float(between)
        out.vars_total[stat] = float(within + between)

    return out
