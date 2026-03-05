"""Variance-decomposition summaries for hierarchical experiment trees.

Every statistic is tracked with a mean and a within/between variance
decomposition via the law of total variance, enabling multi-level
comparisons (neuron → video → timepoint → treatment).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


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
