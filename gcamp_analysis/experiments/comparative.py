"""Comparison-only workflows built from persisted per-video summaries.

Nothing in this module runs the video-analysis pipeline.  It validates and
loads completed artifacts, then performs descriptive longitudinal, treatment,
or generic hierarchical aggregation.  Optional longitudinal alignment calls
the existing ROI registration/tracking implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from gcamp_analysis.experiments.artifacts import (
    LoadedVideoSummary,
    discover_summary_artifacts,
    load_video_summary,
)
from gcamp_analysis.experiments.comparison_utils import (
    build_sibling_comparison,
    summary_to_comparison_row,
)
from gcamp_analysis.experiments.summary_utils import (
    NodeSummary,
    aggregate_node_summaries,
)
from gcamp_analysis.experiments.tree import ExperimentTreeBuilder, TreeNode, is_video_dir
from gcamp_analysis.longitudinal.models import RecordingRef
from gcamp_analysis.longitudinal.tracking import LongitudinalTracker
from gcamp_analysis.recording_discovery import parse_region_day


METADATA_COLUMNS = (
    "video_path",
    "group",
    "treatment",
    "animal",
    "subject",
    "region",
    "well",
    "day",
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    video_path: str = ""
    group: str = ""


@dataclass
class ValidationReport:
    """Validation messages emitted before any comparison is run."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        video_path: Path | str = "",
        group: str = "",
    ) -> None:
        self.issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                message=message,
                video_path=str(video_path),
                group=str(group),
            )
        )

    def to_frame(self) -> pd.DataFrame:
        columns = ["severity", "code", "group", "video_path", "message"]
        return pd.DataFrame(
            [issue.__dict__ for issue in self.issues], columns=columns
        )[columns]

    def raise_for_errors(self) -> None:
        if not self.has_errors:
            return
        errors = [issue.message for issue in self.issues if issue.severity == "error"]
        raise ValueError("Comparison input validation failed:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class ComparisonRecord:
    artifact: LoadedVideoSummary
    metadata: dict[str, Any]

    @property
    def video_path(self) -> Path:
        return self.artifact.video_path

    @property
    def summary(self) -> NodeSummary:
        return self.artifact.summary


@dataclass
class ComparisonDataset:
    records: list[ComparisonRecord]
    validation: ValidationReport

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.metadata for record in self.records])


@dataclass
class LongitudinalComparisonResult:
    validation: ValidationReport
    recordings: pd.DataFrame
    group_day_summary: pd.DataFrame
    alignment_manifests: dict[str, dict[str, Path]] = field(default_factory=dict)

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "validation": self.validation.to_frame(),
            "recordings": self.recordings,
            "group_day_summary": self.group_day_summary,
        }


@dataclass
class TreatmentComparisonResult:
    validation: ValidationReport
    recordings: pd.DataFrame
    replicate_summary: pd.DataFrame
    treatment_summary: pd.DataFrame
    treatment_day_summary: pd.DataFrame
    alignment_manifests: dict[str, dict[str, Path]] = field(default_factory=dict)

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "validation": self.validation.to_frame(),
            "recordings": self.recordings,
            "replicate_summary": self.replicate_summary,
            "treatment_summary": self.treatment_summary,
            "treatment_day_summary": self.treatment_day_summary,
        }


@dataclass
class HierarchicalComparisonResult:
    validation: ValidationReport
    root: TreeNode | None
    sibling_tables: dict[Path, pd.DataFrame]


def load_comparison_dataset(
    group_paths: Mapping[str, Path],
    *,
    metadata: (
        pd.DataFrame
        | Path
        | str
        | Sequence[Mapping[str, Any]]
        | Mapping[str, str]
        | None
    ) = None,
    required_fields: Iterable[str] = (),
) -> ComparisonDataset:
    """Load summaries below named folders and apply optional metadata overrides."""
    report = ValidationReport()
    folder_metadata = _folder_metadata_schema(metadata, report)
    explicit_metadata = None if isinstance(metadata, Mapping) else metadata
    metadata_frame = _load_metadata(explicit_metadata, report)
    metadata_lookup = _metadata_lookup(metadata_frame, report)
    records: list[ComparisonRecord] = []
    seen_paths: dict[Path, str] = {}

    for group, raw_root in group_paths.items():
        root = Path(raw_root).expanduser()
        if not root.is_dir():
            report.add(
                "error",
                "missing_group_folder",
                f"Group folder does not exist: {root}",
                group=group,
            )
            continue
        artifacts = discover_summary_artifacts(root)
        if not artifacts:
            report.add(
                "error",
                "no_video_summaries",
                f"No completed per-video summary artifacts found below {root}.",
                group=group,
            )
            continue
        for artifact_path in artifacts:
            try:
                artifact = load_video_summary(artifact_path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                report.add(
                    "error",
                    "invalid_video_summary",
                    f"Could not load {artifact_path}: {exc}",
                    video_path=artifact_path,
                    group=group,
                )
                continue
            canonical = artifact.video_path.resolve()
            if canonical in seen_paths:
                report.add(
                    "error",
                    "duplicate_video_assignment",
                    f"Video is assigned to both {seen_paths[canonical]!r} and {group!r}.",
                    video_path=canonical,
                    group=group,
                )
                continue
            seen_paths[canonical] = str(group)
            _validate_completed_artifact(artifact, report, str(group))
            inferred = _infer_metadata(
                artifact,
                str(group),
                root=root,
                folder_metadata=folder_metadata,
            )
            override = metadata_lookup.get(canonical, {})
            combined = {**inferred, **_non_missing_values(override)}
            combined["video_path"] = str(canonical)
            combined["group"] = combined.get("group") or str(group)
            records.append(ComparisonRecord(artifact=artifact, metadata=combined))

    fingerprints = {
        record.artifact.config_fingerprint
        for record in records
        if record.artifact.config_fingerprint
    }
    if len(fingerprints) > 1:
        report.add(
            "error",
            "incompatible_analysis_configs",
            "Loaded videos were analyzed with different configuration fingerprints.",
        )
    if records and any(
        record.artifact.config_fingerprint is None for record in records
    ):
        report.add(
            "warning",
            "missing_config_fingerprint",
            "At least one summary lacks a configuration fingerprint; full compatibility cannot be verified.",
        )
    for record in records:
        for field_name in required_fields:
            if _is_missing(record.metadata.get(field_name)):
                report.add(
                    "error",
                    "missing_metadata",
                    f"Required metadata field {field_name!r} is missing.",
                    video_path=record.video_path,
                    group=str(record.metadata.get("group", "")),
                )
    loaded_paths = {record.video_path.resolve() for record in records}
    for path in metadata_lookup:
        if path not in loaded_paths:
            report.add(
                "warning",
                "metadata_video_not_loaded",
                "Metadata row does not match a loaded video below the configured folders.",
                video_path=path,
            )
    return ComparisonDataset(records=records, validation=report)


def run_longitudinal_comparison(
    group_paths: Mapping[str, Path],
    *,
    metadata: pd.DataFrame | Path | str | Sequence[Mapping[str, Any]] | None = None,
    align: bool = False,
    output_dir: Path | None = None,
    strategy: str = "combined",
    anchor_day: int | None = None,
    top_fraction: float = 0.10,
    top_n: int | None = None,
) -> LongitudinalComparisonResult:
    """Compare whole-video statistics over time, optionally adding ROI tracking."""
    dataset = load_comparison_dataset(
        group_paths,
        metadata=metadata,
        required_fields=("group", "day"),
    )
    _validate_unique_keys(dataset, ("group", "day"), "longitudinal_group_day")
    _validate_minimum_group_size(dataset, ("group",), minimum=2)
    recordings = _recording_rows(dataset.records)
    group_day = _summaries_by_keys(dataset.records, ("group", "day"))
    manifests: dict[str, dict[str, Path]] = {}

    if align:
        if output_dir is None:
            dataset.validation.add(
                "error",
                "missing_alignment_output",
                "output_dir is required when align=True.",
            )
        _validate_alignment_inputs(dataset)
        if not dataset.validation.has_errors:
            for group, group_records in _partition(dataset.records, ("group",)).items():
                refs = _recording_refs(group_records, treatment="longitudinal", region=group[0])
                tracker = LongitudinalTracker(
                    experiment_root=group_records[0].video_path.parent,
                    strategy=strategy,
                )
                manifests[group[0]] = tracker.run(
                    treatment="longitudinal",
                    region=group[0],
                    output_dir=Path(output_dir),
                    anchor_day=anchor_day,
                    top_fraction=top_fraction,
                    top_n=top_n,
                    recordings=refs,
                )
    return LongitudinalComparisonResult(
        validation=dataset.validation,
        recordings=recordings,
        group_day_summary=group_day,
        alignment_manifests=manifests,
    )


def run_treatment_comparison(
    treatment_paths: Mapping[str, Path],
    *,
    replicate_unit: str,
    metadata: pd.DataFrame | Path | str | Sequence[Mapping[str, Any]] | None = None,
    longitudinal: bool = False,
    align: bool = False,
    output_dir: Path | None = None,
    strategy: str = "combined",
    anchor_day: int | None = None,
    top_fraction: float = 0.10,
    top_n: int | None = None,
) -> TreatmentComparisonResult:
    """Compare treatments using an experiment-defined independent replicate."""
    required = ["treatment", replicate_unit]
    if longitudinal:
        required.append("day")
    dataset = load_comparison_dataset(
        treatment_paths,
        metadata=metadata,
        required_fields=required,
    )
    for record in dataset.records:
        if _is_missing(record.metadata.get("treatment")):
            record.metadata["treatment"] = record.metadata["group"]
    treatments = {
        str(record.metadata.get("treatment")) for record in dataset.records
    }
    if len(treatments) < 2:
        dataset.validation.add(
            "error",
            "too_few_treatments",
            "Treatment comparison requires at least two treatment groups.",
        )
    _warn_low_replicate_counts(dataset, replicate_unit)

    recordings = _recording_rows(dataset.records)
    replicate_groups = _aggregate_groups(dataset.records, ("treatment", replicate_unit))
    replicate_summary = _rows_from_group_summaries(replicate_groups, ("treatment", replicate_unit))

    treatment_groups: dict[tuple[str, ...], NodeSummary] = {}
    for treatment_key, items in _partition_summaries(
        replicate_groups, key_indices=(0,)
    ).items():
        treatment_groups[treatment_key] = aggregate_node_summaries(
            items, children_are_videos=False
        )
    treatment_summary = _rows_from_group_summaries(treatment_groups, ("treatment",))

    treatment_day_summary = pd.DataFrame()
    manifests: dict[str, dict[str, Path]] = {}
    if longitudinal:
        _validate_unique_keys(
            dataset,
            ("treatment", replicate_unit, "day"),
            "replicate_day",
        )
        _validate_minimum_group_size(
            dataset, ("treatment", replicate_unit), minimum=2
        )
        replicate_day = _aggregate_groups(
            dataset.records, ("treatment", replicate_unit, "day")
        )
        treatment_day_groups: dict[tuple[str, ...], NodeSummary] = {}
        by_treatment_day: dict[tuple[str, str], list[NodeSummary]] = {}
        for key, summary in replicate_day.items():
            by_treatment_day.setdefault((key[0], key[2]), []).append(summary)
        for key, summaries in by_treatment_day.items():
            treatment_day_groups[key] = aggregate_node_summaries(
                summaries, children_are_videos=False
            )
        treatment_day_summary = _rows_from_group_summaries(
            treatment_day_groups, ("treatment", "day")
        )

        if align:
            if output_dir is None:
                dataset.validation.add(
                    "error",
                    "missing_alignment_output",
                    "output_dir is required when align=True.",
                )
            _validate_alignment_inputs(dataset)
            if not dataset.validation.has_errors:
                partitions = _partition(dataset.records, ("treatment", replicate_unit))
                for (treatment, replicate), group_records in partitions.items():
                    label = f"{treatment}/{replicate}"
                    refs = _recording_refs(
                        group_records,
                        treatment=treatment,
                        region=replicate,
                    )
                    tracker = LongitudinalTracker(
                        experiment_root=group_records[0].video_path.parent,
                        strategy=strategy,
                    )
                    manifests[label] = tracker.run(
                        treatment=treatment,
                        region=replicate,
                        output_dir=Path(output_dir),
                        anchor_day=anchor_day,
                        top_fraction=top_fraction,
                        top_n=top_n,
                        recordings=refs,
                    )

    return TreatmentComparisonResult(
        validation=dataset.validation,
        recordings=recordings,
        replicate_summary=replicate_summary,
        treatment_summary=treatment_summary,
        treatment_day_summary=treatment_day_summary,
        alignment_manifests=manifests,
    )


def run_hierarchical_comparison(root_path: Path) -> HierarchicalComparisonResult:
    """Recreate the existing filesystem sibling comparisons from artifacts."""
    root_path = Path(root_path)
    dataset = load_comparison_dataset({root_path.name: root_path})
    if dataset.validation.has_errors:
        return HierarchicalComparisonResult(dataset.validation, None, {})
    tree = ExperimentTreeBuilder(is_video_dir=is_video_dir).build(root_path)
    by_path = {record.video_path.resolve(): record.summary for record in dataset.records}

    def post(node: TreeNode) -> None:
        for child in node.children.values():
            post(child)
        if is_video_dir(node.path):
            summary = by_path.get(node.path.resolve())
            if summary is None:
                dataset.validation.add(
                    "error",
                    "missing_video_summary",
                    "Video directory has no persisted analysis summary.",
                    video_path=node.path,
                )
            else:
                node.summary = summary
            return
        node.summary = aggregate_node_summaries(
            (child.summary for child in node.children.values()),
            children_are_videos=bool(node.children)
            and all(is_video_dir(child.path) for child in node.children.values()),
        )

    post(tree)
    tables: dict[Path, pd.DataFrame] = {}
    for node in tree.iter_nodes():
        if len(node.children) < 2:
            continue
        table = build_sibling_comparison(
            (child.name, child.summary) for child in node.children.values()
        )
        if table is not None:
            tables[node.path] = table
    return HierarchicalComparisonResult(dataset.validation, tree, tables)


def save_comparison_tables(
    tables: Mapping[str, pd.DataFrame],
    path: Path,
) -> Path:
    """Write comparison and validation tables to one workbook."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tables.items():
            sheet = _unique_sheet_name(str(name), used)
            (table if table is not None else pd.DataFrame()).to_excel(
                writer, sheet_name=sheet, index=False
            )
    return path


def _load_metadata(value: Any, report: ValidationReport) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame(columns=METADATA_COLUMNS)
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            report.add("error", "missing_metadata_file", f"Metadata file does not exist: {path}")
            return pd.DataFrame(columns=METADATA_COLUMNS)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        report.add("error", "unsupported_metadata_file", f"Unsupported metadata format: {path}")
        return pd.DataFrame(columns=METADATA_COLUMNS)
    return pd.DataFrame(list(value))


def _metadata_lookup(frame: pd.DataFrame, report: ValidationReport) -> dict[Path, dict[str, Any]]:
    if frame.empty:
        return {}
    if "video_path" not in frame.columns:
        report.add("error", "missing_video_path_column", "Explicit metadata requires a video_path column.")
        return {}
    lookup: dict[Path, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        if _is_missing(row.get("video_path")):
            report.add("error", "empty_video_path", "Metadata contains an empty video_path.")
            continue
        path = Path(str(row["video_path"])).expanduser().resolve()
        if path in lookup:
            report.add("error", "duplicate_metadata_row", "Metadata contains duplicate video_path rows.", video_path=path)
            continue
        lookup[path] = row.to_dict()
    return lookup


def _folder_metadata_schema(
    metadata: Any,
    report: ValidationReport,
) -> dict[str, int]:
    """Translate folder locators into metadata-field/ancestor-depth pairs.

    Supported locators are ``video``, ``video_parent``,
    ``video_grandparent``, and ``video_ancestor_N`` where the video itself is
    ancestor zero. Values are metadata field names.
    """
    if not isinstance(metadata, Mapping):
        return {}

    depths = {"video": 0, "video_parent": 1, "video_grandparent": 2}
    schema: dict[str, int] = {}
    for locator, field_name in metadata.items():
        if not isinstance(locator, str) or not isinstance(field_name, str):
            report.add(
                "error",
                "invalid_folder_metadata",
                "Folder metadata must map string folder locators to string field names.",
            )
            continue
        depth = depths.get(locator)
        if depth is None:
            match = re.fullmatch(r"video_ancestor_(\d+)", locator)
            depth = int(match.group(1)) if match else None
        if depth is None:
            report.add(
                "error",
                "invalid_folder_metadata_locator",
                f"Unsupported folder metadata locator {locator!r}.",
            )
            continue
        if field_name in schema:
            report.add(
                "error",
                "duplicate_folder_metadata_field",
                f"Metadata field {field_name!r} is assigned by multiple folder levels.",
            )
            continue
        schema[field_name] = depth
    return schema


def _infer_metadata(
    artifact: LoadedVideoSummary,
    group: str,
    *,
    root: Path,
    folder_metadata: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    parsed = parse_region_day(artifact.video_name)
    region, day = parsed if parsed is not None else (None, None)
    try:
        relative_parts = artifact.video_path.resolve().relative_to(
            root.resolve()
        ).parts
    except ValueError:
        relative_parts = ()
    # Treatment datasets commonly use treatment/animal/video. Do not invent
    # an animal when videos are stored directly below the treatment root.
    animal = relative_parts[0] if len(relative_parts) >= 2 else None
    inferred = {
        "video_path": str(artifact.video_path.resolve()),
        "group": group,
        "treatment": group,
        "animal": animal,
        "subject": animal,
        "region": region,
        "well": region,
        "day": day,
    }
    if folder_metadata:
        video = artifact.video_path.resolve()
        root_resolved = root.resolve()
        levels: list[str] = []
        current = video
        while True:
            levels.append(current.name)
            if current == root_resolved:
                break
            if current.parent == current or root_resolved not in current.parents:
                levels = []
                break
            current = current.parent
        for field_name, depth in folder_metadata.items():
            inferred[field_name] = levels[depth] if depth < len(levels) else None
    return inferred


def _validate_completed_artifact(
    artifact: LoadedVideoSummary,
    report: ValidationReport,
    group: str,
) -> None:
    video = artifact.video_path
    metrics = video / "metrics" / f"{video.name}_metrics.xlsx"
    if not metrics.is_file():
        report.add(
            "error",
            "missing_video_metrics",
            f"Completed per-video metrics workbook is missing: {metrics}",
            video_path=video,
            group=group,
        )
    artifact_time = artifact.artifact_path.stat().st_mtime_ns
    inputs = [
        video / "suite2p" / "plane0" / "F.npy",
        video / "suite2p" / "plane0" / "stat.npy",
        video / "suite2p" / "plane0" / "ops.npy",
        metrics,
    ]
    newer = [path for path in inputs if path.is_file() and path.stat().st_mtime_ns > artifact_time]
    if newer:
        report.add(
            "error",
            "stale_video_summary",
            "Video summary is older than analysis input/output files: "
            + ", ".join(str(path) for path in newer),
            video_path=video,
            group=group,
        )


def _non_missing_values(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not _is_missing(item)}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _validate_unique_keys(
    dataset: ComparisonDataset,
    columns: tuple[str, ...],
    code_suffix: str,
) -> None:
    seen: dict[tuple[str, ...], Path] = {}
    for record in dataset.records:
        key = tuple(str(record.metadata.get(column)) for column in columns)
        if key in seen:
            dataset.validation.add(
                "error",
                f"duplicate_{code_suffix}",
                f"Multiple videos share {dict(zip(columns, key))}.",
                video_path=record.video_path,
                group=str(record.metadata.get("group", "")),
            )
        else:
            seen[key] = record.video_path


def _validate_alignment_inputs(dataset: ComparisonDataset) -> None:
    for record in dataset.records:
        video = record.video_path
        plane0 = video / "suite2p" / "plane0"
        for required in (plane0 / "ops.npy", plane0 / "stat.npy"):
            if not required.is_file():
                dataset.validation.add(
                    "error", "missing_alignment_input", f"Missing alignment input: {required}", video_path=video
                )
        snaps = list(video.glob("*_snap.tif"))
        if not snaps:
            dataset.validation.add(
                "error", "missing_snap", "Alignment requires a *_snap.tif image.", video_path=video
            )


def _validate_minimum_group_size(
    dataset: ComparisonDataset,
    keys: tuple[str, ...],
    *,
    minimum: int,
) -> None:
    for key, records in _partition(dataset.records, keys).items():
        if len(records) >= minimum:
            continue
        dataset.validation.add(
            "error",
            "too_few_longitudinal_timepoints",
            f"Longitudinal group {dict(zip(keys, key))} has {len(records)} "
            f"timepoint(s); at least {minimum} are required.",
            video_path=records[0].video_path if records else "",
            group="/".join(key),
        )


def _warn_low_replicate_counts(
    dataset: ComparisonDataset,
    replicate_unit: str,
) -> None:
    values: dict[str, set[str]] = {}
    for record in dataset.records:
        treatment = str(record.metadata.get("treatment"))
        replicate = str(record.metadata.get(replicate_unit))
        values.setdefault(treatment, set()).add(replicate)
    for treatment, replicates in values.items():
        if len(replicates) >= 2:
            continue
        dataset.validation.add(
            "warning",
            "low_replicate_count",
            f"Treatment {treatment!r} has only {len(replicates)} independent "
            f"{replicate_unit} replicate(s).",
            group=treatment,
        )


def _recording_rows(records: Sequence[ComparisonRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record.metadata)
        row.update(summary_to_comparison_row(record.artifact.video_name, record.summary))
        row["artifact_path"] = str(record.artifact.artifact_path)
        rows.append(row)
    return pd.DataFrame(rows)


def _partition(
    records: Sequence[ComparisonRecord],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], list[ComparisonRecord]]:
    groups: dict[tuple[str, ...], list[ComparisonRecord]] = {}
    for record in records:
        key = tuple(str(record.metadata.get(name)) for name in keys)
        groups.setdefault(key, []).append(record)
    return groups


def _aggregate_groups(
    records: Sequence[ComparisonRecord],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], NodeSummary]:
    return {
        key: aggregate_node_summaries(
            (record.summary for record in items), children_are_videos=True
        )
        for key, items in _partition(records, keys).items()
    }


def _summaries_by_keys(
    records: Sequence[ComparisonRecord],
    keys: tuple[str, ...],
) -> pd.DataFrame:
    return _rows_from_group_summaries(_aggregate_groups(records, keys), keys)


def _rows_from_group_summaries(
    summaries: Mapping[tuple[str, ...], NodeSummary],
    keys: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for key, summary in sorted(summaries.items()):
        row = dict(zip(keys, key))
        row.update(summary_to_comparison_row("/".join(key), summary))
        rows.append(row)
    return pd.DataFrame(rows)


def _partition_summaries(
    summaries: Mapping[tuple[str, ...], NodeSummary],
    *,
    key_indices: tuple[int, ...],
) -> dict[tuple[str, ...], list[NodeSummary]]:
    groups: dict[tuple[str, ...], list[NodeSummary]] = {}
    for key, summary in summaries.items():
        parent = tuple(key[index] for index in key_indices)
        groups.setdefault(parent, []).append(summary)
    return groups


def _recording_refs(
    records: Sequence[ComparisonRecord],
    *,
    treatment: str,
    region: str,
) -> list[RecordingRef]:
    refs = []
    for record in records:
        video = record.video_path
        refs.append(
            RecordingRef(
                treatment=treatment,
                region=region,
                day=int(record.metadata["day"]),
                recording_name=video.name,
                video_dir=video,
                plane0_dir=video / "suite2p" / "plane0",
                metrics_path=video / "metrics" / f"{video.name}_metrics.xlsx",
            )
        )
    return sorted(refs, key=lambda item: item.day)


def _unique_sheet_name(name: str, used: set[str]) -> str:
    base = "".join(character if character not in "[]:*?/\\" else "_" for character in name)[:31] or "sheet"
    candidate = base
    counter = 1
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{base[:31-len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate
