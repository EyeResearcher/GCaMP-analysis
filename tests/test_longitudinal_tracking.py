from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from experiment_analysis.longitudinal.models import CellMatch, RegistrationResult
from experiment_analysis.longitudinal.registration import (
    estimate_mask_translation,
    estimate_snap_translation,
    estimate_translation,
    match_rois_to_anchor,
    shift_image,
)
from experiment_analysis.longitudinal.tracking import (
    LongitudinalTracker,
    _find_snap,
    _load_groups,
    _mask_outline,
)
from experiment_analysis.longitudinal.tracking import discover_recordings
from experiment_analysis.longitudinal.models import RecordingRef
import tifffile


def _stat_entry(y0: int, x0: int, size: int = 4) -> dict:
    yy, xx = np.mgrid[y0 : y0 + size, x0 : x0 + size]
    return {"ypix": yy.ravel(), "xpix": xx.ravel()}


def test_discover_recordings_keeps_regions_separate(tmp_path: Path) -> None:
    treatment = tmp_path / "BP"
    expected = ["1-1", "1-1_Day2", "1-2", "1-2_Day7"]
    for name in expected:
        plane0 = treatment / name / "suite2p" / "plane0"
        plane0.mkdir(parents=True)
        np.save(plane0 / "ops.npy", {"meanImg": np.zeros((8, 8))})
        np.save(plane0 / "stat.npy", np.asarray([], dtype=object))
    (treatment / "notes").mkdir()

    found = discover_recordings(tmp_path)

    assert [(item.region, item.day) for item in found] == [
        ("1-1", 1),
        ("1-1", 2),
        ("1-2", 1),
        ("1-2", 7),
    ]


def test_estimate_translation_maps_moving_image_to_anchor() -> None:
    anchor = np.zeros((96, 96), dtype=float)
    anchor[20:28, 30:39] = 2.0
    anchor[62:74, 55:63] = 1.0
    moving = shift_image(anchor, 5, -7)

    result = estimate_translation(anchor, moving, max_shift=15)

    assert (result.shift_y, result.shift_x) == (-5, 7)
    assert result.correlation > 0.95


def test_estimate_snap_translation_maps_moving_snap_to_anchor() -> None:
    anchor = np.zeros((128, 128), dtype=float)
    anchor[18:29, 27:39] = 4.0
    anchor[70:84, 83:94] = 2.0
    anchor[92:99, 31:47] = 3.0
    moving = shift_image(anchor, -9, 6)

    result = estimate_snap_translation(anchor, moving, max_shift=20)

    assert (result.shift_y, result.shift_x) == (9, -6)
    assert result.method == "snap_phase_correlation_log_highpass"


def test_mask_matching_is_one_to_one_and_reports_ambiguity() -> None:
    anchor_stat = np.asarray(
        [_stat_entry(10, 10), _stat_entry(35, 40)],
        dtype=object,
    )
    # Moving masks are displaced by (+3, -2); registration reverses that.
    moving_stat = np.asarray(
        [_stat_entry(13, 8), _stat_entry(38, 38), _stat_entry(70, 70)],
        dtype=object,
    )
    registration = RegistrationResult(-3, 2, 1.0)

    matches, _ = match_rois_to_anchor(
        anchor_stat,
        moving_stat,
        (96, 96),
        registration,
    )

    assert [(match.anchor_roi, match.moving_roi) for match in matches] == [(0, 0), (1, 1)]
    assert all(match.iou == 1.0 for match in matches)
    assert all(not match.ambiguous for match in matches)


def test_estimate_mask_translation_recovers_displacement() -> None:
    anchor_stat = np.asarray(
        [_stat_entry(10, 10), _stat_entry(35, 40), _stat_entry(65, 20)],
        dtype=object,
    )
    moving_stat = np.asarray(
        [_stat_entry(14, 7), _stat_entry(39, 37), _stat_entry(69, 17)],
        dtype=object,
    )

    result = estimate_mask_translation(
        anchor_stat, moving_stat, (96, 96), max_shift=10
    )

    assert (result.shift_y, result.shift_x) == (-4, 3)
    assert result.method == "suite2p_mask_overlap"


def test_missing_group_sheet_means_no_groups(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.xlsx"
    with pd.ExcelWriter(metrics, engine="openpyxl") as writer:
        pd.DataFrame({"neuron_idx": [1]}).to_excel(
            writer, sheet_name="spike_summary", index=False
        )
    recording = RecordingRef(
        treatment="BP",
        region="1-1",
        day=2,
        recording_name="1-1_Day2",
        video_dir=tmp_path,
        plane0_dir=tmp_path,
        metrics_path=metrics,
    )

    assert _load_groups(recording, "combined") == {}


def test_mask_outline_marks_boundary_without_filling_interior() -> None:
    yy, xx = np.mgrid[3:8, 4:9]
    linear = np.ravel_multi_index((yy.ravel(), xx.ravel()), (12, 12))

    outline_y, outline_x = _mask_outline(linear, (12, 12))
    outlined = set(zip(outline_y.tolist(), outline_x.tolist()))

    assert (5, 6) not in outlined
    assert (3, 4) in outlined
    assert (2, 4) in outlined


def _recording(tmp_path: Path, day: int, stat: list[dict]) -> RecordingRef:
    name = "1-1" if day == 1 else f"1-1_Day{day}"
    video = tmp_path / "BP" / name
    plane0 = video / "suite2p" / "plane0"
    plane0.mkdir(parents=True)
    image = np.zeros((64, 64), dtype=np.float32)
    image[8:16, 8:16] = 3
    image[35:46, 40:53] = 2
    np.save(plane0 / "ops.npy", {"meanImg": image})
    np.save(plane0 / "stat.npy", np.asarray(stat, dtype=object))
    tifffile.imwrite(video / f"{name}_snap.tif", image)
    metrics = video / "metrics" / f"{name}_metrics.xlsx"
    metrics.parent.mkdir()
    with pd.ExcelWriter(metrics, engine="openpyxl") as writer:
        pd.DataFrame(
            {"group_id": ["A"], "neuron_indices": [str(list(range(len(stat))))],
             "method": ["combined"]}
        ).to_excel(writer, sheet_name="grouping_stats", index=False)
        pd.DataFrame({"neuron_idx": list(range(len(stat)))}).to_excel(
            writer, sheet_name="spike_summary", index=False
        )
    return RecordingRef("BP", "1-1", day, name, video, plane0, metrics)


def test_adjacent_morphology_tracks_where_direct_match_fails(tmp_path: Path) -> None:
    recordings = [
        _recording(tmp_path, 1, [_stat_entry(20, 10)]),
        _recording(tmp_path, 2, [_stat_entry(20, 12)]),
        _recording(tmp_path, 3, [_stat_entry(20, 14)]),
    ]
    direct, _ = match_rois_to_anchor(
        np.asarray([_stat_entry(20, 14)], dtype=object),
        np.asarray([_stat_entry(20, 10)], dtype=object),
        (64, 64), RegistrationResult(0, 0, 1),
    )
    assert direct == []

    tracker = LongitudinalTracker(tmp_path, max_registration_shift=10)
    paths = tracker.run(
        treatment="BP", region="1-1", output_dir=tmp_path / "output",
        recordings=recordings, top_n=1,
    )
    history = pd.read_csv(paths["cell_history"]).set_index("day")
    assert history.loc[1, "day_roi"] == 0
    assert history.loc[1, "track_edge_count"] == 2
    assert history.loc[3, "track_edge_count"] == 0
    assert history.detected.all()
    edges = pd.read_csv(paths["cell_matches"])
    assert list(zip(edges.earlier_day, edges.later_day)) == [(1, 2), (2, 3)]
    assert {"earlier_roi", "later_roi", "match_score", "mask_iou",
            "centroid_distance_px", "ambiguous"} <= set(edges.columns)
    assert {"track_min_score", "track_min_iou",
            "track_max_centroid_distance_px", "track_has_ambiguous_edge"} <= set(history.columns)
    assert "match_score" not in history.columns

    middle_paths = tracker.run(
        treatment="BP", region="1-1", output_dir=tmp_path / "middle_output",
        recordings=recordings, anchor_day=2, top_n=1,
    )
    middle_history = pd.read_csv(middle_paths["cell_history"]).set_index("day")
    assert middle_history.detected.all()
    assert middle_history.loc[1, "track_edge_count"] == 1
    assert middle_history.loc[3, "track_edge_count"] == 1


def test_trace_composes_both_directions_and_stops_at_missing_edge(tmp_path: Path) -> None:
    recordings = [
        RecordingRef("BP", "1-1", day, str(day), tmp_path, tmp_path, tmp_path)
        for day in (1, 2, 3, 4)
    ]
    # In CellMatch, anchor_roi belongs to the later day and moving_roi to earlier.
    edge_12 = CellMatch(12, 10, 0.8, 0.6, 2.0, False)
    edge_23 = CellMatch(15, 12, 0.7, 0.5, 3.0, True)
    edge_34 = CellMatch(17, 15, 0.9, 0.4, 1.0, False)
    backward = {(1, 2): {12: edge_12}, (2, 3): {15: edge_23},
                (3, 4): {17: edge_34}}
    forward = {(1, 2): {10: edge_12}, (2, 3): {12: edge_23},
               (3, 4): {15: edge_34}}
    tracker = LongitudinalTracker(tmp_path)
    traced = tracker._trace_matches_from_anchor(
        recordings=recordings, chosen_anchor_day=4, anchor_roi_count=18,
        later_to_earlier=backward, earlier_to_later=forward,
    )
    first = traced[1][17]
    assert (first.day_roi, first.edge_count) == (10, 3)
    assert (first.minimum_score, first.minimum_iou,
            first.maximum_centroid_distance, first.has_ambiguous_edge) == (
                0.7, 0.4, 3.0, True
            )
    assert 16 not in traced[3]

    broken = dict(backward)
    broken[(2, 3)] = {}
    traced = tracker._trace_matches_from_anchor(
        recordings=recordings, chosen_anchor_day=4, anchor_roi_count=18,
        later_to_earlier=broken, earlier_to_later=forward,
    )
    assert 17 in traced[3]
    assert 17 not in traced[2] and 17 not in traced[1]

    middle = tracker._trace_matches_from_anchor(
        recordings=recordings, chosen_anchor_day=2, anchor_roi_count=13,
        later_to_earlier=backward, earlier_to_later=forward,
    )
    assert middle[1][12].day_roi == 10
    assert middle[4][12].day_roi == 17


def test_broken_edge_omits_overlay_and_history_for_earlier_day(tmp_path: Path) -> None:
    recordings = [
        _recording(tmp_path, 1, [_stat_entry(20, 10), _stat_entry(42, 10)]),
        _recording(tmp_path, 2, [_stat_entry(20, 12), _stat_entry(42, 35)]),
        _recording(tmp_path, 3, [_stat_entry(20, 14), _stat_entry(42, 35)]),
    ]
    paths = LongitudinalTracker(tmp_path, max_registration_shift=10).run(
        treatment="BP", region="1-1", output_dir=tmp_path / "output",
        recordings=recordings, top_n=1,
    )
    history = pd.read_csv(paths["cell_history"])
    assert not history.query("day == 1 and anchor_roi == 1").iloc[0].detected
    assert history.query("day == 1 and anchor_roi == 0").iloc[0].detected
    frames = tifffile.imread(paths["overlay_tiff"])
    # The connected ROI is colored; the disconnected ROI remains grayscale.
    assert len(set(frames[0, 21, 11].tolist())) > 1
    assert len(set(frames[0, 43, 11].tolist())) == 1


def test_pairwise_snap_shift_is_scaled_from_each_adjacent_edge(tmp_path: Path) -> None:
    recordings = [
        _recording(tmp_path, 1, [_stat_entry(20, 10)]),
        _recording(tmp_path, 2, [_stat_entry(20, 12)]),
        _recording(tmp_path, 3, [_stat_entry(20, 14)]),
    ]
    edges = [RegistrationResult(4, -6, 0.9), RegistrationResult(-2, 8, 0.8)]
    with patch(
        "experiment_analysis.longitudinal.tracking.estimate_snap_translation",
        side_effect=edges,
    ):
        composed, pairwise, rows, _ = LongitudinalTracker(tmp_path)._sequential_snap_registrations(
            recordings, anchor_day=2, mask_shape=(32, 32)
        )
    assert (pairwise[(1, 2)].shift_y, pairwise[(1, 2)].shift_x) == (2, -3)
    assert (pairwise[(2, 3)].shift_y, pairwise[(2, 3)].shift_x) == (-1, 4)
    assert (composed[1].shift_y, composed[1].shift_x) == (2, -3)
    assert (composed[3].shift_y, composed[3].shift_x) == (1, -4)
    assert rows[0]["mask_shift_y_px"] == pairwise[(1, 2)].shift_y


def test_adjacent_assignment_cannot_reuse_an_earlier_roi() -> None:
    later = np.asarray([_stat_entry(10, 10), _stat_entry(10, 10)], dtype=object)
    earlier = np.asarray([_stat_entry(10, 10)], dtype=object)
    matches, _ = match_rois_to_anchor(
        later, earlier, (32, 32), RegistrationResult(0, 0, 1.0)
    )
    assert len(matches) == 1
    assert matches[0].moving_roi == 0


def test_empty_pairwise_edge_table_keeps_its_schema(tmp_path: Path) -> None:
    recordings = [
        _recording(tmp_path, 1, [_stat_entry(20, 10)]),
        _recording(tmp_path, 2, [_stat_entry(20, 45)]),
    ]
    paths = LongitudinalTracker(tmp_path, max_registration_shift=10).run(
        treatment="BP", region="1-1", output_dir=tmp_path / "output",
        recordings=recordings, top_n=1,
    )
    edges = pd.read_csv(paths["cell_matches"])
    assert edges.empty
    assert list(edges.columns) == [
        "treatment", "region", "anchor_day", "earlier_day", "earlier_roi",
        "later_day", "later_roi", "match_score", "mask_iou",
        "centroid_distance_px", "ambiguous",
    ]


def test_snap_discovery_accepts_tiff_extension(tmp_path: Path) -> None:
    recording = _recording(tmp_path, 1, [_stat_entry(20, 10)])
    snap = recording.video_dir / f"{recording.recording_name}_snap.tif"
    tiff_snap = snap.with_suffix(".tiff")
    snap.rename(tiff_snap)

    assert _find_snap(recording) == tiff_snap
