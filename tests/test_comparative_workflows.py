from __future__ import annotations

from pathlib import Path

import pandas as pd

from gcamp_analysis.experiments.artifacts import (
    load_video_summary,
    summary_artifact_path,
    write_video_summary,
)
from gcamp_analysis.experiments.comparative import (
    load_comparison_dataset,
    run_longitudinal_comparison,
    run_treatment_comparison,
)
from gcamp_analysis.experiments.models import VideoRunRecord
from gcamp_analysis.experiments.summary_utils import StatSummary


def _write_video(root: Path, name: str, mean: float = 1.0) -> Path:
    video = root / name
    video.mkdir(parents=True)
    metrics_dir = video / "metrics"
    metrics_dir.mkdir()
    with pd.ExcelWriter(metrics_dir / f"{name}_metrics.xlsx", engine="openpyxl") as writer:
        pd.DataFrame({"complete": [True]}).to_excel(writer, index=False)
    stat = StatSummary(
        means={"rise": mean},
        vars_total={"rise": 0.0},
        vars_within={"rise": 0.0},
        vars_between={"rise": 0.0},
    )
    record = VideoRunRecord(
        video_dir=video,
        metrics_dir=metrics_dir,
        n_rois_total=5,
        n_rois_good=4,
        n_neurons=3,
        n_spikes_kept=8,
        n_neurons_grouped=2,
        n_neurons_ungrouped=1,
        kin_unweighted=stat,
        kin_weighted_spikes=stat,
    )
    write_video_summary(record, analysis_metadata={"config": "same"})
    return video


def test_video_summary_round_trip(tmp_path: Path) -> None:
    video = _write_video(tmp_path, "1-1_Day2", mean=2.5)

    loaded = load_video_summary(summary_artifact_path(video))

    assert loaded.video_path == video.resolve()
    assert loaded.summary.n_neurons == 3
    assert loaded.summary.kin_weighted.means["rise"] == 2.5


def test_explicit_metadata_overrides_folder_name_inference(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "group", "unconventional_name")
    metadata = pd.DataFrame(
        {
            "video_path": [video],
            "group": ["explicit_group"],
            "region": ["region-A"],
            "day": [7],
        }
    )

    dataset = load_comparison_dataset(
        {"inferred_group": tmp_path / "group"},
        metadata=metadata,
        required_fields=("region", "day"),
    )

    assert not dataset.validation.has_errors
    assert dataset.records[0].metadata["group"] == "explicit_group"
    assert dataset.records[0].metadata["region"] == "region-A"
    assert dataset.records[0].metadata["day"] == 7


def test_longitudinal_without_alignment_returns_whole_video_stats(tmp_path: Path) -> None:
    group = tmp_path / "region"
    _write_video(group, "1-1", mean=1.0)
    _write_video(group, "1-1_Day2", mean=3.0)

    result = run_longitudinal_comparison({"region_1": group}, align=False)

    assert not result.validation.has_errors
    assert result.alignment_manifests == {}
    assert result.group_day_summary["day"].tolist() == ["1", "2"]
    assert result.group_day_summary["rise_mean_weighted"].tolist() == [1.0, 3.0]


def test_treatment_comparison_uses_configured_replicate_unit(tmp_path: Path) -> None:
    control_a = _write_video(tmp_path / "control", "control_a", mean=1.0)
    control_b = _write_video(tmp_path / "control", "control_b", mean=3.0)
    drug_a = _write_video(tmp_path / "drug", "drug_a", mean=5.0)
    drug_b = _write_video(tmp_path / "drug", "drug_b", mean=7.0)
    metadata = pd.DataFrame(
        {
            "video_path": [control_a, control_b, drug_a, drug_b],
            "treatment": ["control", "control", "drug", "drug"],
            "animal": ["a", "b", "a", "b"],
        }
    )

    result = run_treatment_comparison(
        {"control": tmp_path / "control", "drug": tmp_path / "drug"},
        replicate_unit="animal",
        metadata=metadata,
    )

    assert not result.validation.has_errors
    means = result.treatment_summary.set_index("treatment")["rise_mean_weighted"]
    assert means["control"] == 2.0
    assert means["drug"] == 6.0


def test_treatment_comparison_infers_animal_from_parent_folder(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    drug = tmp_path / "drug"
    _write_video(control / "animal-a", "animal-a_L-1", mean=1.0)
    _write_video(control / "animal-b", "animal-b_L-1", mean=3.0)
    _write_video(drug / "animal-c", "animal-c_L-1", mean=5.0)
    _write_video(drug / "animal-d", "animal-d_L-1", mean=7.0)

    result = run_treatment_comparison(
        {"control": control, "drug": drug},
        replicate_unit="animal",
    )

    assert not result.validation.has_errors
    assert set(result.recordings["animal"]) == {
        "animal-a",
        "animal-b",
        "animal-c",
        "animal-d",
    }
    assert not any(
        issue.code == "low_replicate_count" for issue in result.validation.issues
    )


def test_treatment_comparison_maps_folder_levels_to_metadata(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control-folder"
    drug = tmp_path / "drug-folder"
    _write_video(control / "animal-a", "region-1", mean=1.0)
    _write_video(control / "animal-b", "region-2", mean=3.0)
    _write_video(drug / "animal-c", "region-3", mean=5.0)
    _write_video(drug / "animal-d", "region-4", mean=7.0)

    result = run_treatment_comparison(
        {"control-label": control, "drug-label": drug},
        replicate_unit="animal",
        metadata={
            "video": "region",
            "video_parent": "animal",
            "video_grandparent": "treatment",
        },
    )

    assert not result.validation.has_errors
    rows = result.recordings.set_index("region")
    assert rows.loc["region-1", "animal"] == "animal-a"
    assert rows.loc["region-1", "treatment"] == "control-folder"
    assert rows.loc["region-4", "animal"] == "animal-d"
    assert rows.loc["region-4", "treatment"] == "drug-folder"
