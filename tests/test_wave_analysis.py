from __future__ import annotations

import numpy as np

from gcamp_analysis.waves.analysis import (
    WaveAnalysisConfig,
    _benjamini_hochberg,
    fit_propagation,
)


def _config() -> WaveAnalysisConfig:
    return WaveAnalysisConfig(
        propagation_null_repeats=199,
        radial_grid_size=17,
        random_seed=13,
    )


def test_planar_wave_is_detected_against_permuted_coordinates():
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 1000, size=(100, 2))
    onsets = 50 + 0.025 * coords[:, 0] + 0.01 * coords[:, 1]
    onsets += rng.normal(0, 0.7, size=coords.shape[0])
    result = fit_propagation(
        coords,
        onsets,
        fs=15.0,
        pixel_size_um=1.25,
        config=_config(),
        rng=np.random.default_rng(11),
    )
    assert result["model"] == "planar"
    assert result["propagation_r2"] > 0.9
    assert result["propagation_p"] <= 0.01
    assert np.isfinite(result["speed_um_s"])


def test_radial_wave_origin_and_significance_are_recovered():
    rng = np.random.default_rng(8)
    coords = rng.uniform(0, 1000, size=(120, 2))
    origin = np.asarray([250.0, 700.0])
    onsets = 30 + 0.02 * np.linalg.norm(coords - origin, axis=1)
    onsets += rng.normal(0, 0.5, size=coords.shape[0])
    result = fit_propagation(
        coords,
        onsets,
        fs=15.0,
        pixel_size_um=1.25,
        config=_config(),
        rng=np.random.default_rng(12),
    )
    assert result["model"] == "radial"
    assert result["propagation_r2"] > 0.9
    assert result["propagation_p"] <= 0.01
    assert np.linalg.norm(
        np.asarray([result["source_x_px"], result["source_y_px"]]) - origin
    ) < 75


def test_bh_adjustment_is_monotonic_in_rank():
    values = np.asarray([0.04, 0.001, 0.03, 0.2])
    adjusted = _benjamini_hochberg(values)
    order = np.argsort(values)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= values)
