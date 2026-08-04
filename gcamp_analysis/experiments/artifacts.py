"""Versioned per-video artifacts used by downstream comparison workflows.

The analysis pipeline writes one JSON summary per video.  Comparison code
loads these summaries without rebuilding models or rerunning video analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from gcamp_analysis.experiments.models import VideoRunRecord
from gcamp_analysis.experiments.summary_utils import (
    NodeSummary,
    StatSummary,
    summary_from_video_record,
)


ARTIFACT_KIND = "gcamp-video-analysis-summary"
ARTIFACT_VERSION = 1
SUMMARY_SUFFIX = "_analysis_summary.json"


@dataclass(frozen=True)
class LoadedVideoSummary:
    """One validated video-summary artifact."""

    artifact_path: Path
    video_path: Path
    video_name: str
    summary: NodeSummary
    config_fingerprint: str | None
    analysis_metadata: dict[str, Any]


def config_fingerprint(value: Mapping[str, Any] | None) -> str | None:
    """Return a stable digest for JSON-compatible analysis configuration."""
    if value is None:
        return None
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summary_artifact_path(video_path: Path) -> Path:
    """Return the standard summary path for *video_path*."""
    video_path = Path(video_path)
    return video_path / "metrics" / f"{video_path.name}{SUMMARY_SUFFIX}"


def write_video_summary(
    record: VideoRunRecord,
    *,
    analysis_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist the comparison-ready portion of one processed video."""
    path = summary_artifact_path(record.video_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(analysis_metadata or {})
    summary = summary_from_video_record(record, source=record.video_dir.name)
    payload = {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": ARTIFACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "video_name": record.video_dir.name,
        "video_path": str(record.video_dir.resolve()),
        "metrics_path": str(record.metrics_dir.resolve()),
        "config_fingerprint": config_fingerprint(metadata),
        "analysis_metadata": metadata,
        "processing_counts": {
            "n_rois_total": record.n_rois_total,
            "n_rois_good": record.n_rois_good,
            "n_spikes_kept": record.n_spikes_kept,
        },
        "summary": _node_summary_to_dict(summary),
    }
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return path


def load_video_summary(path: Path) -> LoadedVideoSummary:
    """Load one artifact, rejecting incompatible kinds or schema versions."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError(f"Unsupported artifact kind in {path}.")
    if payload.get("schema_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported video-summary schema in {path}: "
            f"{payload.get('schema_version')!r}; expected {ARTIFACT_VERSION}."
        )
    return LoadedVideoSummary(
        artifact_path=path.resolve(),
        video_path=Path(payload["video_path"]),
        video_name=str(payload["video_name"]),
        summary=_node_summary_from_dict(payload["summary"]),
        config_fingerprint=payload.get("config_fingerprint"),
        analysis_metadata=dict(payload.get("analysis_metadata") or {}),
    )


def discover_summary_artifacts(root: Path) -> list[Path]:
    """Return all standard per-video summaries below *root*."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(root.rglob(f"*{SUMMARY_SUFFIX}"))


def _stat_to_dict(summary: StatSummary) -> dict[str, dict[str, float]]:
    return {
        "means": dict(summary.means),
        "vars_total": dict(summary.vars_total),
        "vars_within": dict(summary.vars_within),
        "vars_between": dict(summary.vars_between),
    }


def _stat_from_dict(value: Mapping[str, Any] | None) -> StatSummary:
    value = value or {}
    return StatSummary(
        means=_numeric_mapping(value.get("means")),
        vars_total=_numeric_mapping(value.get("vars_total")),
        vars_within=_numeric_mapping(value.get("vars_within")),
        vars_between=_numeric_mapping(value.get("vars_between")),
    )


def _node_summary_to_dict(summary: NodeSummary) -> dict[str, Any]:
    return {
        "n_videos": summary.n_videos,
        "n_neurons": summary.n_neurons,
        "n_neurons_grouped": summary.n_neurons_grouped,
        "n_neurons_ungrouped": summary.n_neurons_ungrouped,
        "n_groups": dict(summary.n_groups),
        "group_stats": summary.group_stats,
        "kin_unweighted": _stat_to_dict(summary.kin_unweighted),
        "kin_weighted": _stat_to_dict(summary.kin_weighted),
        "freq_unweighted": _stat_to_dict(summary.freq_unweighted),
        "freq_weighted": _stat_to_dict(summary.freq_weighted),
        "kin_grouped": _stat_to_dict(summary.kin_grouped),
        "kin_ungrouped": _stat_to_dict(summary.kin_ungrouped),
        "freq_grouped": _stat_to_dict(summary.freq_grouped),
        "freq_ungrouped": _stat_to_dict(summary.freq_ungrouped),
    }


def _node_summary_from_dict(value: Mapping[str, Any]) -> NodeSummary:
    return NodeSummary(
        n_videos=int(value.get("n_videos", 1)),
        n_neurons=int(value.get("n_neurons", 0)),
        n_neurons_grouped=int(value.get("n_neurons_grouped", 0)),
        n_neurons_ungrouped=int(value.get("n_neurons_ungrouped", 0)),
        n_groups={str(k): int(v) for k, v in (value.get("n_groups") or {}).items()},
        group_stats={
            str(strategy): _numeric_mapping(stats)
            for strategy, stats in (value.get("group_stats") or {}).items()
        },
        kin_unweighted=_stat_from_dict(value.get("kin_unweighted")),
        kin_weighted=_stat_from_dict(value.get("kin_weighted")),
        freq_unweighted=_stat_from_dict(value.get("freq_unweighted")),
        freq_weighted=_stat_from_dict(value.get("freq_weighted")),
        kin_grouped=_stat_from_dict(value.get("kin_grouped")),
        kin_ungrouped=_stat_from_dict(value.get("kin_ungrouped")),
        freq_grouped=_stat_from_dict(value.get("freq_grouped")),
        freq_ungrouped=_stat_from_dict(value.get("freq_ungrouped")),
    )


def _numeric_mapping(value: Mapping[str, Any] | None) -> dict[str, float]:
    return {
        str(key): float(item)
        for key, item in (value or {}).items()
        if item is not None
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
