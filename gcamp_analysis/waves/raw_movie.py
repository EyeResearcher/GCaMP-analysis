"""Raw-movie discovery of propagating calcium fronts.

This module deliberately does not use ROI traces to nominate events.  It
downsamples the fluorescence movie into spatial blocks, detects frames with
coherent spatiotemporal gradients, and then fits planar or radial arrival-time
fields to independently selected movie blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks

from .analysis import (
    WaveAnalysisConfig,
    _benjamini_hochberg,
    _linear_r2,
    _positive_radial_scores,
    _radial_distance_grid,
    fit_propagation,
)


@dataclass(frozen=True)
class RawMovieWaveConfig:
    """Settings for movie-first wave discovery."""

    block_size_pixels: int = 16
    baseline_sigma_frames: float = 35.0
    temporal_sigma_frames: float = 1.0
    spatial_sigma_blocks: float = 0.8
    candidate_min_distance_frames: int = 12
    candidate_prominence_z: float = 0.45
    fit_half_window_frames: int = 12
    active_block_quantile: float = 0.70
    min_active_blocks: int = 80
    null_patch_size_blocks: int = 8
    propagation_null_repeats: int = 499
    alpha: float = 0.05
    min_propagation_r2: float = 0.15
    random_seed: int = 7341


def block_mean_frame(frame: np.ndarray, block_size: int) -> np.ndarray:
    """Average non-overlapping square blocks in one image."""
    height = frame.shape[0] // block_size * block_size
    width = frame.shape[1] // block_size * block_size
    cropped = np.asarray(frame[:height, :width], dtype=np.float32)
    return cropped.reshape(
        height // block_size,
        block_size,
        width // block_size,
        block_size,
    ).mean(axis=(1, 3))


def load_block_movie(
    path: str | Path,
    *,
    block_size: int,
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray:
    """Load a TIFF movie as a compact block-mean array."""
    with tifffile.TiffFile(path) as tif:
        n_frames = len(tif.pages)
        stop = n_frames if stop is None else min(int(stop), n_frames)
        start = max(0, int(start))
        if stop <= start:
            raise ValueError("stop must be greater than start")
        first = block_mean_frame(tif.pages[start].asarray(), block_size)
        movie = np.empty((stop - start, *first.shape), dtype=np.float32)
        movie[0] = first
        for output_index, frame_index in enumerate(
            range(start + 1, stop), start=1
        ):
            movie[output_index] = block_mean_frame(
                tif.pages[frame_index].asarray(), block_size
            )
    return movie


def preprocess_block_movie(
    movie: np.ndarray, config: RawMovieWaveConfig
) -> np.ndarray:
    """Return spatially smoothed, global-signal-suppressed dF/F."""
    values = np.asarray(movie, dtype=np.float32)
    baseline = gaussian_filter1d(
        values,
        sigma=config.baseline_sigma_frames,
        axis=0,
        mode="nearest",
    )
    floor = np.maximum(np.median(baseline, axis=0, keepdims=True) * 0.05, 1.0)
    dff = (values - baseline) / np.maximum(baseline, floor)
    dff -= np.median(dff, axis=(1, 2), keepdims=True)
    return gaussian_filter(
        dff,
        sigma=(0.0, config.spatial_sigma_blocks, config.spatial_sigma_blocks),
        mode="nearest",
    )


def movie_gradient_metrics(
    processed: np.ndarray, config: RawMovieWaveConfig
) -> tuple[pd.DataFrame, np.ndarray]:
    """Calculate front energy, normal-flow coherence, and divergence by frame."""
    temporal = gaussian_filter1d(
        processed,
        sigma=config.temporal_sigma_frames,
        order=1,
        axis=0,
        mode="nearest",
    )
    grad_y = np.gradient(processed, axis=1)
    grad_x = np.gradient(processed, axis=2)
    positive = np.maximum(temporal, 0.0)
    gradient_magnitude = np.hypot(grad_x, grad_y)
    weights = positive * gradient_magnitude
    denominator = grad_x**2 + grad_y**2
    regularizer = np.quantile(denominator, 0.75, axis=(1, 2), keepdims=True)
    regularizer = np.maximum(regularizer * 0.1, 1e-12)
    flow_x = -temporal * grad_x / (denominator + regularizer)
    flow_y = -temporal * grad_y / (denominator + regularizer)
    flow_norm = np.hypot(flow_x, flow_y)
    unit_x = np.divide(
        flow_x, flow_norm, out=np.zeros_like(flow_x), where=flow_norm > 0
    )
    unit_y = np.divide(
        flow_y, flow_norm, out=np.zeros_like(flow_y), where=flow_norm > 0
    )
    weight_sum = weights.sum(axis=(1, 2))
    mean_x = np.divide(
        (weights * unit_x).sum(axis=(1, 2)),
        weight_sum,
        out=np.zeros_like(weight_sum),
        where=weight_sum > 0,
    )
    mean_y = np.divide(
        (weights * unit_y).sum(axis=(1, 2)),
        weight_sum,
        out=np.zeros_like(weight_sum),
        where=weight_sum > 0,
    )
    planar_coherence = np.hypot(mean_x, mean_y)
    divergence = np.gradient(flow_x, axis=2) + np.gradient(flow_y, axis=1)
    positive_divergence = np.quantile(np.maximum(divergence, 0.0), 0.95, axis=(1, 2))
    front_energy = np.quantile(positive, 0.90, axis=(1, 2))
    spatiality = temporal.std(axis=(1, 2)) / (
        np.mean(np.abs(temporal), axis=(1, 2)) + 1e-12
    )
    metrics = pd.DataFrame(
        {
            "front_energy": front_energy,
            "spatiality": spatiality,
            "planar_coherence": planar_coherence,
            "positive_divergence": positive_divergence,
        }
    )
    for column in metrics.columns:
        values = metrics[column].to_numpy()
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        metrics[f"{column}_z"] = (values - median) / max(1.4826 * mad, 1e-12)
    metrics["discovery_score"] = (
        metrics["front_energy_z"]
        + 0.5 * metrics["spatiality_z"]
        + 0.5
        * np.maximum(
            metrics["planar_coherence_z"],
            metrics["positive_divergence_z"],
        )
    )
    return metrics, temporal


def candidate_frames(
    metrics: pd.DataFrame,
    config: RawMovieWaveConfig,
    *,
    ranges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Find separated local maxima of the gradient discovery score."""
    score = metrics["discovery_score"].to_numpy()
    peaks, _ = find_peaks(
        score,
        distance=config.candidate_min_distance_frames,
        prominence=config.candidate_prominence_z,
    )
    if ranges:
        keep = np.zeros(peaks.size, dtype=bool)
        for start, stop in ranges:
            keep |= (peaks >= start) & (peaks < stop)
        peaks = peaks[keep]
    return peaks


def fit_movie_candidate(
    temporal_gradient: np.ndarray,
    center_frame: int,
    *,
    fs: float,
    pixel_size_um: float,
    config: RawMovieWaveConfig,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one candidate using blockwise times of maximum positive gradient."""
    start = max(0, center_frame - config.fit_half_window_frames)
    stop = min(
        temporal_gradient.shape[0],
        center_frame + config.fit_half_window_frames + 1,
    )
    window = np.maximum(temporal_gradient[start:stop], 0.0)
    amplitude = window.max(axis=0)
    relative_arrival = np.argmax(window, axis=0)
    threshold = float(np.quantile(amplitude, config.active_block_quantile))
    active = amplitude >= threshold
    active &= relative_arrival > 0
    active &= relative_arrival < window.shape[0] - 1
    rows, columns = np.indices(amplitude.shape)
    coords = np.column_stack(
        [
            (columns[active] + 0.5) * config.block_size_pixels,
            (rows[active] + 0.5) * config.block_size_pixels,
        ]
    )
    arrivals = relative_arrival[active].astype(float) + start
    if coords.shape[0] < config.min_active_blocks:
        return (
            {
                "movie_status": "insufficient_active_blocks",
                "movie_n_active_blocks": int(coords.shape[0]),
            },
            coords,
            arrivals,
            active,
        )
    wave_config = WaveAnalysisConfig(
        propagation_null_repeats=1
    )
    fitted = fit_propagation(
        coords,
        arrivals,
        fs=fs,
        pixel_size_um=pixel_size_um,
        config=wave_config,
        rng=rng,
    )
    observed_score = float(fitted["selection_score_r2"])
    if observed_score < config.min_propagation_r2:
        fitted["iid_propagation_p"] = fitted["propagation_p"]
        fitted["propagation_p"] = 1.0
        fitted["null_score_95"] = np.nan
        result = {
            f"movie_{key}": value for key, value in fitted.items()
        }
        result.update(
            {
                "movie_status": "ok",
                "movie_n_active_blocks": int(coords.shape[0]),
                "movie_arrival_span_frames": float(np.ptp(arrivals)),
            }
        )
        return result, coords, arrivals, active
    height, width = amplitude.shape
    patch = min(config.null_patch_size_blocks, height, width)
    while patch > 1 and (height % patch or width % patch):
        patch -= 1
    patch_rows = height // patch
    patch_columns = width // patch
    arrival_tiles = relative_arrival.reshape(
        patch_rows, patch, patch_columns, patch
    ).transpose(0, 2, 1, 3).reshape(-1, patch, patch)
    amplitude_tiles = amplitude.reshape(
        patch_rows, patch, patch_columns, patch
    ).transpose(0, 2, 1, 3).reshape(-1, patch, patch)
    grid_rows, grid_columns = np.indices((height, width))
    null_scores = np.zeros(config.propagation_null_repeats, dtype=float)
    for repeat in range(config.propagation_null_repeats):
        permutation = rng.permutation(arrival_tiles.shape[0])
        permuted_arrival = arrival_tiles[permutation].reshape(
            patch_rows, patch_columns, patch, patch
        ).transpose(0, 2, 1, 3).reshape(height, width)
        permuted_amplitude = amplitude_tiles[permutation].reshape(
            patch_rows, patch_columns, patch, patch
        ).transpose(0, 2, 1, 3).reshape(height, width)
        permuted_active = permuted_amplitude >= threshold
        permuted_active &= permuted_arrival > 0
        permuted_active &= permuted_arrival < window.shape[0] - 1
        permuted_coords = np.column_stack(
            [
                (grid_columns[permuted_active] + 0.5)
                * config.block_size_pixels,
                (grid_rows[permuted_active] + 0.5)
                * config.block_size_pixels,
            ]
        )
        permuted_values = permuted_arrival[permuted_active].astype(float)
        centered = permuted_coords - permuted_coords.mean(axis=0, keepdims=True)
        planar_score, _ = _linear_r2(
            np.column_stack(
                [centered, np.ones(permuted_coords.shape[0])]
            ),
            permuted_values,
        )
        _, distances = _radial_distance_grid(
            permuted_coords,
            wave_config.radial_grid_size,
            wave_config.radial_grid_margin_fraction,
        )
        radial_score = float(
            _positive_radial_scores(distances, permuted_values[:, None])[0]
        )
        null_scores[repeat] = max(planar_score, radial_score)
    patch_p = (
        1.0 + float(np.sum(null_scores >= observed_score))
    ) / (config.propagation_null_repeats + 1.0)
    fitted["iid_propagation_p"] = fitted["propagation_p"]
    fitted["propagation_p"] = patch_p
    fitted["null_score_95"] = float(np.quantile(null_scores, 0.95))
    result = {
        f"movie_{key}": value for key, value in fitted.items()
    }
    result.update(
        {
            "movie_status": "ok",
            "movie_n_active_blocks": int(coords.shape[0]),
            "movie_arrival_span_frames": float(np.ptp(arrivals)),
        }
    )
    return result, coords, arrivals, active


def analyze_block_movie(
    movie: np.ndarray,
    *,
    fs: float,
    pixel_size_um: float,
    config: RawMovieWaveConfig = RawMovieWaveConfig(),
    ranges: list[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Discover and statistically test movie-first propagation candidates."""
    processed = preprocess_block_movie(movie, config)
    metrics, temporal = movie_gradient_metrics(processed, config)
    peaks = candidate_frames(metrics, config, ranges=ranges)
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict] = []
    for center in peaks:
        fitted, _, _, _ = fit_movie_candidate(
            temporal,
            int(center),
            fs=fs,
            pixel_size_um=pixel_size_um,
            config=config,
            rng=rng,
        )
        rows.append(
            {
                "center_frame": int(center),
                "center_seconds": float(center / fs),
                **metrics.iloc[int(center)].to_dict(),
                **fitted,
            }
        )
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        return episodes, metrics, temporal
    episodes["movie_propagation_q"] = np.nan
    valid = episodes["movie_status"].eq("ok")
    episodes.loc[valid, "movie_propagation_q"] = _benjamini_hochberg(
        episodes.loc[valid, "movie_propagation_p"]
    )
    episodes["movie_significant_wave"] = (
        valid
        & (episodes["movie_propagation_q"] <= config.alpha)
        & (episodes["movie_propagation_r2"] >= config.min_propagation_r2)
    )
    return episodes, metrics, temporal
