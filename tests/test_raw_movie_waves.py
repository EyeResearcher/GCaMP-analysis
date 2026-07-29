import numpy as np

from gcamp_analysis.waves.raw_movie import (
    RawMovieWaveConfig,
    analyze_block_movie,
    block_mean_frame,
)


def test_block_mean_frame():
    frame = np.arange(64, dtype=float).reshape(8, 8)
    result = block_mean_frame(frame, 4)
    assert result.shape == (2, 2)
    assert np.isclose(result[0, 0], frame[:4, :4].mean())


def test_movie_first_detector_finds_planar_front():
    rng = np.random.default_rng(4)
    n_frames, height, width = 80, 24, 24
    yy, xx = np.indices((height, width))
    arrival = 25 + 0.45 * xx
    time = np.arange(n_frames)[:, None, None]
    movie = 1000.0 + 30.0 / (1.0 + np.exp(-(time - arrival) / 0.8))
    movie += rng.normal(0.0, 1.0, size=movie.shape)
    config = RawMovieWaveConfig(
        block_size_pixels=1,
        baseline_sigma_frames=15,
        candidate_min_distance_frames=8,
        candidate_prominence_z=0.1,
        fit_half_window_frames=12,
        active_block_quantile=0.4,
        min_active_blocks=80,
        propagation_null_repeats=99,
    )
    episodes, _, _ = analyze_block_movie(
        movie.astype(np.float32),
        fs=15.0,
        pixel_size_um=1.0,
        config=config,
        ranges=[(20, 45)],
    )
    assert len(episodes) >= 1
    best = episodes.sort_values("movie_propagation_p").iloc[0]
    assert best["movie_model"] == "planar"
    assert best["movie_propagation_r2"] > 0.5
    assert best["movie_propagation_p"] <= 0.05
