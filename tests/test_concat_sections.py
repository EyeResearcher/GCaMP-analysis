from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gcamp_analysis.data_classes.neuron import Neuron
from gcamp_analysis.data_classes.roi import ROI
from gcamp_analysis.data_classes.video import Video, VideoStatistics, VideoStatisticsWriter
from gcamp_analysis.grouping_processing.service import GroupingService
from gcamp_analysis.grouping_processing.treatment_comparison import SectionComparisonResult
from gcamp_analysis.roi_processing.filtering import ROIService
from gcamp_analysis.roi_processing.traces import TraceService
from gcamp_analysis.spike_processing.filtering import SpikeService


def _fake_suite2p(n_rois: int = 2, n_frames: int = 12) -> dict:
    base = np.array(
        [
            [0.0, 2.0, 0.2, 0.1, 0.0, 3.0, 0.1, 0.0, 0.2, 4.0, 0.1, 0.0],
            [0.0, 1.5, 0.1, 0.0, 0.0, 2.5, 0.2, 0.0, 0.1, 3.5, 0.2, 0.0],
        ],
        dtype=float,
    )
    if n_rois != 2 or n_frames != 12:
        base = np.tile(base[0], (n_rois, 1))[:, :n_frames]
    return {
        "F": base,
        "Fneu": np.zeros_like(base),
        "stat": np.asarray([{"med": [0.0, 0.0]} for _ in range(base.shape[0])], dtype=object),
        "fs": 15.0,
        "ops": {"fs": 15.0},
    }


def _write_concat_csv(path: Path, rows: list[tuple]) -> None:
    df = pd.DataFrame(
        rows,
        columns=["index", "source file name", "section type", "start frame", "end frame"],
    )
    df.to_csv(path / "videoA_concat_order.csv", index=False)


class _PredictAllTrue:
    def predict(self, X):
        return np.ones(len(X), dtype=bool)


@pytest.fixture
def fake_video_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    video_dir = tmp_path / "exp" / "drug" / "day1" / "videoA"
    suite2p_dir = video_dir / "suite2p" / "plane0"
    suite2p_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "gcamp_analysis.data_classes.video.load_suite2p_data",
        lambda _path: _fake_suite2p(),
    )
    return video_dir


def _make_concat_video(fake_video_dir: Path, rows: list[tuple]) -> Video:
    _write_concat_csv(fake_video_dir, rows)
    return Video(
        path=fake_video_dir,
        suite2p_path=fake_video_dir / "suite2p" / "plane0",
        is_concatenated=True,
    )


def _seed_neurons(video: Video) -> None:
    rois = [
        ROI(index=i, f_trace=video.suite2p_data["F"][i, :], stats=video.suite2p_data["stat"][i])
        for i in range(video.n_rois)
    ]
    for roi in rois:
        roi.is_good = True
        roi.active_segments = {section.section_key: True for section in video.concat_sections}
    video.neurons = [Neuron(roi=roi, filtered_index=i, fs=float(video.fs)) for i, roi in enumerate(rois)]


def test_video_loads_and_parses_concat_sections(fake_video_dir: Path):
    video = _make_concat_video(
        fake_video_dir,
        [
            (0, "src_a", "baseline", 0, 4),
            (1, "src_b", "treatment", 4, 8),
            (2, "src_c", "recovery", 8, 10),
            (3, "src_d", "treatment", 10, 12),
        ],
    )

    assert [section.section_key for section in video.concat_sections] == [
        "baseline",
        "treatment_1",
        "recovery_1",
        "treatment_2",
    ]
    assert [section.section_kind for section in video.concat_sections] == [
        "baseline",
        "treatment",
        "recovery",
        "treatment",
    ]
    assert video.baseline_section.frame_slice == slice(0, 4)
    assert [section.section_key for section in video.get_sections_by_kind("treatment")] == [
        "treatment_1",
        "treatment_2",
    ]
    assert [section.section_key for section in video.iter_nonbaseline_sections()] == [
        "treatment_1",
        "recovery_1",
        "treatment_2",
    ]


def test_trace_service_assigns_section_trace_mapping(fake_video_dir: Path):
    video = _make_concat_video(
        fake_video_dir,
        [
            (0, "src_a", "baseline", 0, 4),
            (1, "src_b", "treatment", 4, 8),
            (2, "src_c", "recovery", 8, 12),
        ],
    )

    TraceService().run(video)

    assert set(video.section_traces) == {"baseline", "treatment_1", "recovery_1"}
    assert video.section_traces["baseline"]["norm_f"].shape == (2, 4)
    assert video.section_traces["treatment_1"]["norm_f"].shape == (2, 4)
    assert video.section_traces["recovery_1"]["norm_f"].shape == (2, 4)
    np.testing.assert_allclose(video.section_traces["baseline"]["norm_f"], video.norm_f[:, 0:4])
    np.testing.assert_allclose(video.section_traces["treatment_1"]["norm_f"], video.norm_f[:, 4:8])


def test_concat_csv_validation_rejects_missing_columns(fake_video_dir: Path):
    bad = pd.DataFrame(
        [(0, "src_a", "baseline", 0)],
        columns=["index", "source file name", "section type", "start frame"],
    )
    bad.to_csv(fake_video_dir / "videoA_concat_order.csv", index=False)

    with pytest.raises(ValueError):
        Video(
            path=fake_video_dir,
            suite2p_path=fake_video_dir / "suite2p" / "plane0",
            is_concatenated=True,
        )


def test_concat_csv_validation_rejects_invalid_type_and_missing_baseline(fake_video_dir: Path):
    _write_concat_csv(fake_video_dir, [(0, "src_a", "drug", 0, 4)])
    with pytest.raises(ValueError):
        Video(
            path=fake_video_dir,
            suite2p_path=fake_video_dir / "suite2p" / "plane0",
            is_concatenated=True,
        )

    _write_concat_csv(fake_video_dir, [(0, "src_a", "treatment", 0, 4)])
    with pytest.raises(ValueError):
        Video(
            path=fake_video_dir,
            suite2p_path=fake_video_dir / "suite2p" / "plane0",
            is_concatenated=True,
        )


def test_roi_filter_records_all_section_activity_and_uses_union(fake_video_dir: Path, monkeypatch: pytest.MonkeyPatch):
    video = _make_concat_video(
        fake_video_dir,
        [
            (0, "src_a", "baseline", 0, 4),
            (1, "src_b", "treatment", 4, 8),
            (2, "src_c", "treatment", 8, 10),
            (3, "src_d", "recovery", 10, 12),
        ],
    )
    TraceService().run(video)

    responses = iter(
        [
            (np.array([True, False]), [{"picked": "baseline_0"}, {"picked": "baseline_1"}]),
            (np.array([False, False]), [{"picked": "t1_0"}, {"picked": "t1_1"}]),
            (np.array([False, True]), [{"picked": "t2_0"}, {"picked": "t2_1"}]),
            (np.array([False, False]), [{"picked": "r1_0"}, {"picked": "r1_1"}]),
        ]
    )

    def fake_get_preds(self, traces, rois, model, transform):
        return next(responses)

    monkeypatch.setattr(ROIService, "_get_preds", fake_get_preds)

    roi_service = ROIService()
    all_rois = roi_service.create_rois(video)
    good_rois, mask = roi_service.filter_rois(video, all_rois, _PredictAllTrue(), model_config={})

    assert mask.tolist() == [True, True]
    assert [roi.index for roi in good_rois] == [0, 1]
    assert all_rois[0].active_segments == {
        "baseline": True,
        "treatment_1": False,
        "treatment_2": False,
        "recovery_1": False,
    }
    assert all_rois[1].active_segments == {
        "baseline": False,
        "treatment_1": False,
        "treatment_2": True,
        "recovery_1": False,
    }
    assert all_rois[0].features["picked"] == "baseline_0"
    assert all_rois[1].features["picked"] == "t2_1"


def test_spike_service_shifts_peaks_and_summarizes_by_section(fake_video_dir: Path, monkeypatch: pytest.MonkeyPatch):
    video = _make_concat_video(
        fake_video_dir,
        [
            (0, "src_a", "baseline", 0, 4),
            (1, "src_b", "treatment", 4, 8),
            (2, "src_c", "recovery", 8, 12),
        ],
    )
    TraceService().run(video)
    _seed_neurons(video)

    peaks_by_length = {
        4: (
            [{"peak_feature": 1.0}],
            [1],
            np.asarray([1], dtype=int),
        ),
    }

    def fake_describe_spikes(trace, roi_index, fs):
        return peaks_by_length[len(trace)]

    monkeypatch.setattr("gcamp_analysis.spike_processing.filtering.describe_spikes", fake_describe_spikes)

    class FakeKinetics:
        def __init__(self, fs):
            self.fs = fs
            self.calls = 0

        def compute(self, window):
            self.calls += 1
            return {"width": float(self.calls)}

    monkeypatch.setattr("gcamp_analysis.spike_processing.filtering.SpikeKinetics", FakeKinetics)

    spike_service = SpikeService(n_jobs=1)
    spk_df = spike_service.extract_spike_features(video)

    first_neuron = video.neurons[0]
    assert first_neuron.peaks.tolist() == [1, 5, 9]
    assert [feature["_section_key"] for feature in first_neuron.spk_features] == [
        "baseline",
        "treatment_1",
        "recovery_1",
    ]

    video.neurons[0].roi.active_segments = {
        "baseline": True,
        "treatment_1": False,
        "recovery_1": True,
    }
    video.neurons[1].roi.active_segments = {
        "baseline": True,
        "treatment_1": True,
        "recovery_1": True,
    }

    for neuron in video.neurons:
        neuron.peaks_filtered = neuron.peaks.tolist()

    summary_df = spike_service.compute_spike_statistics(video)
    assert not spk_df.empty
    assert "baseline_spike_frequency" in summary_df.columns
    assert "treatment_1_spike_frequency" in summary_df.columns
    assert "recovery_1_spike_frequency" in summary_df.columns
    assert "baseline_active" in summary_df.columns
    assert "treatment_1_active" in summary_df.columns
    assert summary_df.loc[0, "baseline_number_of_spikes"] == 1
    assert summary_df.loc[0, "treatment_1_number_of_spikes"] == 0
    assert summary_df.loc[0, "recovery_1_number_of_spikes"] == 1
    assert summary_df.loc[0, "number_of_spikes"] == 2
    assert summary_df.loc[0, "spike_indices"] == [1, 9]
    assert summary_df.loc[0, "spike_values_raw"] == [2.0, 4.0]
    assert summary_df.loc[0, "spike_frequency"] == pytest.approx(3.75)
    assert summary_df.loc[0, "mean_width"] == pytest.approx(2.0)
    assert summary_df.loc[0, "treatment_1_active"] == False
    assert summary_df.loc[0, "treatment_1_spike_frequency"] == 0.0
    assert pd.isna(summary_df.loc[0, "treatment_1_mean_width"])
    assert pd.isna(summary_df.loc[0, "treatment_1_var_width"])
    assert summary_df.loc[0, "baseline_mean_width"] == pytest.approx(1.0)
    assert summary_df.loc[0, "recovery_1_mean_width"] == pytest.approx(3.0)


def test_grouping_service_creates_section_comparisons_per_nonbaseline_section(
    fake_video_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    video = _make_concat_video(
        fake_video_dir,
        [
            (0, "src_a", "baseline", 0, 4),
            (1, "src_b", "treatment", 4, 8),
            (2, "src_c", "recovery", 8, 12),
        ],
    )
    TraceService().run(video)
    _seed_neurons(video)

    captured_section_keys: list[str] = []

    def fake_combined_grouping(**kwargs):
        return {
            "groups": [{"group_id": "g1", "neuron_indices": [0, 1]}],
            "matrix": np.asarray([[1.0, 0.5], [0.5, 1.0]], dtype=float),
            "config_label": "combined",
            "metadata": {},
        }

    def fake_section_comparison(
        section_traces,
        section_spike_trains,
        section_t_stop,
        neuron_indices,
        baseline_groups,
        baseline_matrix,
        *,
        section_key,
        section_kind,
        **kwargs,
    ):
        captured_section_keys.append(section_key)
        return {
            "group_metrics": [{"group_id": "g1", "section_key": section_key, "n_section_subgroups": 1}],
            "section_matrix": np.asarray([[1.0, 0.25], [0.25, 1.0]], dtype=float),
            "subgroups": {},
        }

    monkeypatch.setitem(
        __import__("gcamp_analysis.grouping_processing.service", fromlist=["STRATEGY_REGISTRY"]).STRATEGY_REGISTRY,
        "combined",
        fake_combined_grouping,
    )
    monkeypatch.setattr(
        "gcamp_analysis.grouping_processing.service.run_section_comparison",
        fake_section_comparison,
    )

    grouping_service = GroupingService(strategies=["combined"])
    report = grouping_service.run(video, {"combined": {"corr": {}, "sttc": {}, "cluster": {}}})

    assert report is not None
    assert captured_section_keys == ["treatment_1", "recovery_1"]
    assert set(video.section_comparison_results["combined"]) == {"treatment_1", "recovery_1"}


def test_video_statistics_writer_emits_one_file_per_strategy_section_pair(tmp_path: Path):
    stats = VideoStatistics(
        video_name="videoA",
        per_neuron_spike_summaries=pd.DataFrame({"value": [1]}),
        grouping_stats=pd.DataFrame({"group": [1]}),
        bad_rois_features=pd.DataFrame(),
        section_comparison={
            "combined": {
                "treatment_1": SectionComparisonResult(
                    strategy_name="combined",
                    section_key="treatment_1",
                    section_kind="treatment",
                    group_metrics=[{"group_id": "g1"}],
                    section_matrix=np.asarray([[1.0]]),
                ),
                "recovery_1": SectionComparisonResult(
                    strategy_name="combined",
                    section_key="recovery_1",
                    section_kind="recovery",
                    group_metrics=[{"group_id": "g1"}],
                    section_matrix=np.asarray([[1.0]]),
                ),
            }
        },
    )

    manifest = VideoStatisticsWriter().write(stats, output_root=tmp_path)

    assert "combined_treatment_1_section_comparison" in manifest
    assert "combined_recovery_1_section_matrix_npy" in manifest
    assert Path(manifest["combined_treatment_1_section_comparison"]).exists()
    assert Path(manifest["combined_recovery_1_section_matrix_npy"]).exists()

    workbook = pd.ExcelFile(manifest["metrics_excel"])
    assert "baseline-treatment_1" in workbook.sheet_names
    assert "baseline-recovery_1" in workbook.sheet_names
