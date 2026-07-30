from pathlib import Path

import pandas as pd
import pytest

from gcamp_analysis.experiments.comparison_utils import (
    build_sibling_comparison,
    flatten_stat_summary,
)
from gcamp_analysis.experiments.models import VideoRunRecord
from gcamp_analysis.experiments.summary_utils import (
    NodeSummary,
    StatSummary,
    aggregate_node_summaries,
    summary_from_video_record,
)
from gcamp_analysis.experiments.tree import TreeNode
from gcamp_analysis.reporting import (
    build_comparison_legend,
    save_comparisons,
)


def _stat(mean: float) -> StatSummary:
    return StatSummary(
        means={"value": mean},
        vars_total={"value": 0.0},
        vars_within={"value": 0.0},
        vars_between={"value": 0.0},
    )


def test_summary_from_video_record_tags_detail_tables() -> None:
    record = VideoRunRecord(
        video_dir=Path("video"),
        metrics_dir=Path("video/metrics"),
        n_rois_total=4,
        n_rois_good=3,
        n_neurons=2,
        n_spikes_kept=5,
        light_evoked_details={"responses": pd.DataFrame({"value": [1]})},
    )

    summary = summary_from_video_record(record, source="video")

    assert summary.n_videos == 1
    assert summary.n_neurons == 2
    assert summary.light_evoked_details["responses"]["source"].tolist() == [
        "video"
    ]


def test_aggregate_node_summaries_uses_partition_counts_for_video_weights() -> None:
    first = NodeSummary(
        n_videos=1,
        n_neurons=10,
        n_neurons_grouped=9,
        n_neurons_ungrouped=1,
        kin_weighted=_stat(1.0),
        kin_grouped=_stat(1.0),
        kin_ungrouped=_stat(10.0),
    )
    second = NodeSummary(
        n_videos=1,
        n_neurons=10,
        n_neurons_grouped=1,
        n_neurons_ungrouped=9,
        kin_weighted=_stat(3.0),
        kin_grouped=_stat(3.0),
        kin_ungrouped=_stat(20.0),
    )

    summary = aggregate_node_summaries(
        [first, second],
        children_are_videos=True,
    )

    assert summary.kin_weighted.means["value"] == pytest.approx(2.0)
    assert summary.kin_grouped.means["value"] == pytest.approx(1.2)
    assert summary.kin_ungrouped.means["value"] == pytest.approx(19.0)


def test_aggregate_node_summaries_merges_nested_outputs() -> None:
    first = NodeSummary(
        n_videos=1,
        light_evoked_details={"responses": pd.DataFrame({"value": [1]})},
    )
    second = NodeSummary(
        n_videos=1,
        light_evoked_details={"responses": pd.DataFrame({"value": [2]})},
    )

    summary = aggregate_node_summaries(
        [first, second],
        children_are_videos=False,
    )

    assert summary.light_evoked_details["responses"]["value"].tolist() == [1, 2]


def test_flatten_stat_summary_exports_all_variance_components() -> None:
    summary = StatSummary(
        means={"amplitude": 2.0},
        vars_total={"amplitude": 4.0},
        vars_within={"amplitude": 1.5},
        vars_between={"amplitude": 2.5},
    )

    assert flatten_stat_summary(summary, "weighted") == {
        "amplitude_mean_weighted": 2.0,
        "amplitude_var_weighted": 4.0,
        "amplitude_within_weighted": 1.5,
        "amplitude_between_weighted": 2.5,
    }


def test_build_sibling_comparison_requires_two_summaries_with_data() -> None:
    result = build_sibling_comparison(
        [
            ("empty", NodeSummary()),
            ("processed", NodeSummary(n_videos=1)),
        ]
    )

    assert result is None


def test_build_sibling_comparison_sorts_children_and_flattens_stats() -> None:
    result = build_sibling_comparison(
        [
            (
                "b",
                NodeSummary(
                    n_videos=1,
                    n_neurons=2,
                    kin_weighted=_stat(3.0),
                ),
            ),
            (
                "a",
                NodeSummary(
                    n_videos=1,
                    n_neurons=4,
                    kin_weighted=_stat(1.0),
                ),
            ),
        ]
    )

    assert result is not None
    assert result["child"].tolist() == ["a", "b"]
    assert result["value_mean_weighted"].tolist() == [1.0, 3.0]


def test_build_comparison_legend_describes_stat_columns() -> None:
    comparison = pd.DataFrame(
        columns=[
            "child",
            "n_groups_corr",
            "amplitude_mean_weighted",
            "amplitude_between_weighted",
        ]
    )

    legend = build_comparison_legend(comparison).set_index("column")

    assert "one row per sibling" in legend.loc["child", "description"]
    assert "corr" in legend.loc["n_groups_corr", "description"]
    assert "Mean value" in legend.loc[
        "amplitude_mean_weighted",
        "description",
    ]
    assert "Between-child" in legend.loc[
        "amplitude_between_weighted",
        "description",
    ]


def test_save_comparisons_writes_summary_and_legend(tmp_path: Path) -> None:
    root = TreeNode(name="root", path=tmp_path)
    comparison = pd.DataFrame(
        {
            "child": ["a", "b"],
            "n_videos": [1, 1],
            "n_neurons": [2, 3],
        }
    )

    save_comparisons(
        root=root,
        sibling_tables={tmp_path: comparison},
    )

    output = tmp_path / "metrics" / "sibling_comparisons.xlsx"
    assert output.exists()
    workbook = pd.ExcelFile(output)
    assert workbook.sheet_names == ["summary", "legend"]
