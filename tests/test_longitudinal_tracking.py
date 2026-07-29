from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gcamp_analysis.longitudinal.models import RegistrationResult
from gcamp_analysis.longitudinal.registration import (
    estimate_mask_translation,
    estimate_snap_translation,
    estimate_translation,
    match_rois_to_anchor,
    shift_image,
)
from gcamp_analysis.longitudinal.tracking import _load_groups, _mask_outline
from gcamp_analysis.longitudinal.tracking import discover_recordings
from gcamp_analysis.longitudinal.models import RecordingRef


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
