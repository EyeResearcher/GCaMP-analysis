"""JSON serialization for current recording-result bundles."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from recording_results.summaries import NodeSummary, StatSummary


def config_fingerprint(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stat_to_dict(summary: StatSummary) -> dict[str, dict[str, float]]:
    return {"means": dict(summary.means), "vars_total": dict(summary.vars_total),
            "vars_within": dict(summary.vars_within), "vars_between": dict(summary.vars_between)}


def stat_from_dict(value: Mapping[str, Any] | None) -> StatSummary:
    value = value or {}
    return StatSummary(means=numeric_mapping(value.get("means")),
                       vars_total=numeric_mapping(value.get("vars_total")),
                       vars_within=numeric_mapping(value.get("vars_within")),
                       vars_between=numeric_mapping(value.get("vars_between")))


def summary_to_dict(summary: NodeSummary) -> dict[str, Any]:
    return {"n_videos": summary.n_videos, "n_neurons": summary.n_neurons,
            "n_neurons_grouped": summary.n_neurons_grouped,
            "n_neurons_ungrouped": summary.n_neurons_ungrouped,
            "n_groups": dict(summary.n_groups), "group_stats": summary.group_stats,
            "kin_unweighted": stat_to_dict(summary.kin_unweighted),
            "kin_weighted": stat_to_dict(summary.kin_weighted),
            "freq_unweighted": stat_to_dict(summary.freq_unweighted),
            "freq_weighted": stat_to_dict(summary.freq_weighted),
            "kin_grouped": stat_to_dict(summary.kin_grouped),
            "kin_ungrouped": stat_to_dict(summary.kin_ungrouped),
            "freq_grouped": stat_to_dict(summary.freq_grouped),
            "freq_ungrouped": stat_to_dict(summary.freq_ungrouped)}


def summary_from_dict(value: Mapping[str, Any]) -> NodeSummary:
    return NodeSummary(n_videos=int(value.get("n_videos", 1)),
                       n_neurons=int(value.get("n_neurons", 0)),
                       n_neurons_grouped=int(value.get("n_neurons_grouped", 0)),
                       n_neurons_ungrouped=int(value.get("n_neurons_ungrouped", 0)),
                       n_groups={str(k): int(v) for k, v in (value.get("n_groups") or {}).items()},
                       group_stats={str(k): numeric_mapping(v) for k, v in (value.get("group_stats") or {}).items()},
                       kin_unweighted=stat_from_dict(value.get("kin_unweighted")),
                       kin_weighted=stat_from_dict(value.get("kin_weighted")),
                       freq_unweighted=stat_from_dict(value.get("freq_unweighted")),
                       freq_weighted=stat_from_dict(value.get("freq_weighted")),
                       kin_grouped=stat_from_dict(value.get("kin_grouped")),
                       kin_ungrouped=stat_from_dict(value.get("kin_ungrouped")),
                       freq_grouped=stat_from_dict(value.get("freq_grouped")),
                       freq_ungrouped=stat_from_dict(value.get("freq_ungrouped")))


def numeric_mapping(value: Mapping[str, Any] | None) -> dict[str, float]:
    return {str(key): float(item) for key, item in (value or {}).items() if item is not None}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return None if isinstance(value, float) and not math.isfinite(value) else value
