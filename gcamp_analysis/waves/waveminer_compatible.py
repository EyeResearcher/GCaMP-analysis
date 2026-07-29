"""WaveMiner-compatible segmentation for calcium-imaging movies.

The official WaveMiner code cited by Yeager et al. (2025) is not publicly
available as of July 2026.  This module implements the algorithm documented in
that paper: isolate phasic activity, label activity connected in x/y/t with a
3-D flood fill, and measure the resulting space-time components.  Calcium-
specific thresholding choices are explicit and configurable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_erosion,
    convolve,
    distance_transform_edt,
    generate_binary_structure,
    label,
)

from .analysis import WaveAnalysisConfig, fit_propagation
from .raw_movie import RawMovieWaveConfig, preprocess_block_movie


@dataclass(frozen=True)
class WaveMinerCompatibleConfig:
    """Settings for documented WaveMiner-style x/y/t flood filling."""

    block_size_pixels: int = 16
    pixel_size_um: float = 1.242961138804478
    fs: float = 15.0
    phasic_mode: str = "level"
    phasic_z_threshold: float = 3.0
    minimum_active_neighbor_voxels: int = 1
    minimum_component_voxels: int = 12
    minimum_spatial_pixels: int = 8
    minimum_duration_frames: int = 3
    minimum_propagation_extent_um: float = 160.0
    minimum_arrival_span_frames: int = 3
    minimum_single_front_fraction: float = 0.50
    speed_frame_separation: int = 15
    propagation_null_repeats: int = 99
    random_seed: int = 8128

    @property
    def block_size_um(self) -> float:
        return self.block_size_pixels * self.pixel_size_um


def phasic_activity_movie(
    block_movie: np.ndarray,
    config: WaveMinerCompatibleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return processed dF/F, robust phasic z scores, and active voxels.

    WaveMiner dynamically removes tonic activity before flood filling.  For a
    calcium movie we operationalize phasic activity as a positive temporal
    derivative exceeding a block-specific robust baseline.
    """
    raw_config = RawMovieWaveConfig(
        block_size_pixels=config.block_size_pixels
    )
    processed = preprocess_block_movie(block_movie, raw_config)
    if config.phasic_mode == "level":
        phasic_signal = processed
    elif config.phasic_mode == "derivative":
        phasic_signal = np.gradient(processed, axis=0)
    else:
        raise ValueError("phasic_mode must be 'level' or 'derivative'")
    center = np.median(phasic_signal, axis=0, keepdims=True)
    mad = np.median(np.abs(phasic_signal - center), axis=0, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-7)
    phasic_z = (phasic_signal - center) / scale
    active = phasic_z >= config.phasic_z_threshold

    # The published flood fill discards active pixels with no active spatial
    # neighbor.  Apply that rule before 3-D labeling.
    spatial_kernel = np.ones((1, 3, 3), dtype=np.int8)
    spatial_kernel[0, 1, 1] = 0
    neighbor_count = convolve(
        active.astype(np.int8), spatial_kernel, mode="constant", cval=0
    )
    active &= neighbor_count >= config.minimum_active_neighbor_voxels
    return processed, phasic_z, active


def _front_boundary(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3)))


def _leading_edge_speed(
    component: np.ndarray,
    *,
    block_size_um: float,
    fs: float,
    frame_separation: int,
) -> tuple[float, int]:
    distances: list[float] = []
    for first in range(0, component.shape[0] - frame_separation):
        second = first + frame_separation
        first_front = _front_boundary(component[first])
        second_front = _front_boundary(component[second])
        if not np.any(first_front) or not np.any(second_front):
            continue
        distance = distance_transform_edt(~first_front)
        distances.append(
            float(np.median(distance[second_front]) * block_size_um)
        )
    if not distances:
        return np.nan, 0
    elapsed_seconds = frame_separation / fs
    return float(np.median(distances) / elapsed_seconds), len(distances)


def _single_front_fraction(component: np.ndarray) -> float:
    spatial_structure = generate_binary_structure(2, 2)
    counts = []
    for frame in component:
        if not np.any(frame):
            continue
        _, count = label(frame, structure=spatial_structure)
        counts.append(count)
    if not counts:
        return 0.0
    return float(np.mean(np.asarray(counts) == 1))


def _component_row(
    labels: np.ndarray,
    component_id: int,
    config: WaveMinerCompatibleConfig,
    rng: np.random.Generator,
) -> dict:
    positions = np.argwhere(labels == component_id)
    first_frame, first_y, first_x = positions.min(axis=0)
    last_frame, last_y, last_x = positions.max(axis=0)
    local = (
        labels[
            first_frame : last_frame + 1,
            first_y : last_y + 1,
            first_x : last_x + 1,
        ]
        == component_id
    )
    spatial = np.any(local, axis=0)
    spatial_y, spatial_x = np.nonzero(spatial)
    arrival = np.argmax(local, axis=0)
    arrival_values = arrival[spatial].astype(float) + first_frame
    coords = np.column_stack(
        [
            (spatial_x + first_x + 0.5) * config.block_size_pixels,
            (spatial_y + first_y + 0.5) * config.block_size_pixels,
        ]
    )
    propagation = {}
    if coords.shape[0] >= 20 and np.ptp(arrival_values) > 0:
        propagation = fit_propagation(
            coords,
            arrival_values,
            fs=config.fs,
            pixel_size_um=config.pixel_size_um,
            config=WaveAnalysisConfig(
                propagation_null_repeats=config.propagation_null_repeats
            ),
            rng=rng,
        )
    speed, speed_pairs = _leading_edge_speed(
        local,
        block_size_um=config.block_size_um,
        fs=config.fs,
        frame_separation=config.speed_frame_separation,
    )
    extent_x_um = (last_x - first_x + 1) * config.block_size_um
    extent_y_um = (last_y - first_y + 1) * config.block_size_um
    duration_frames = last_frame - first_frame + 1
    single_front_fraction = _single_front_fraction(local)
    arrival_span = int(np.ptp(arrival_values))
    waveminer_propagating_component = (
        max(extent_x_um, extent_y_um)
        >= config.minimum_propagation_extent_um
        and arrival_span >= config.minimum_arrival_span_frames
    )
    strict_front_candidate = (
        waveminer_propagating_component
        and single_front_fraction >= config.minimum_single_front_fraction
    )
    return {
        "component_id": component_id,
        "start_frame": int(first_frame),
        "end_frame": int(last_frame),
        "center_frame": int(round(np.median(positions[:, 0]))),
        "start_seconds": float(first_frame / config.fs),
        "end_seconds": float(last_frame / config.fs),
        "duration_frames": int(duration_frames),
        "duration_seconds": float(duration_frames / config.fs),
        "voxel_count": int(positions.shape[0]),
        "spatial_block_count": int(spatial.sum()),
        "area_um2": float(spatial.sum() * config.block_size_um**2),
        "extent_x_um": float(extent_x_um),
        "extent_y_um": float(extent_y_um),
        "arrival_span_frames": arrival_span,
        "arrival_span_seconds": float(arrival_span / config.fs),
        "single_front_fraction": single_front_fraction,
        "leading_edge_speed_um_s": speed,
        "leading_edge_speed_pairs": int(speed_pairs),
        "waveminer_component": True,
        "waveminer_propagating_component": bool(
            waveminer_propagating_component
        ),
        "strict_front_candidate": bool(strict_front_candidate),
        # Backward-compatible name used by the first sensitivity run.
        "propagating_candidate": bool(strict_front_candidate),
        **{f"arrival_{key}": value for key, value in propagation.items()},
    }


def segment_phasic_components(
    active: np.ndarray,
    config: WaveMinerCompatibleConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Label activity connected in x/y/t and summarize each component."""
    structure = generate_binary_structure(3, 3)
    labels, count = label(active, structure=structure)
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    keep_ids = np.flatnonzero(
        sizes
        >= max(
            config.minimum_component_voxels,
            config.minimum_spatial_pixels,
        )
    )
    keep_ids = keep_ids[keep_ids != 0]
    if keep_ids.size == 0:
        return pd.DataFrame(), np.zeros_like(labels, dtype=np.int32)
    keep_lookup = np.zeros(count + 1, dtype=np.int32)
    keep_lookup[keep_ids] = np.arange(1, keep_ids.size + 1)
    filtered = keep_lookup[labels]
    rng = np.random.default_rng(config.random_seed)
    rows = [
        _component_row(filtered, int(component_id), config, rng)
        for component_id in range(1, keep_ids.size + 1)
    ]
    return pd.DataFrame(rows), filtered


def analyze_waveminer_compatible(
    block_movie: np.ndarray,
    config: WaveMinerCompatibleConfig = WaveMinerCompatibleConfig(),
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Run calcium phasic filtering and documented 3-D flood filling."""
    processed, phasic_z, active = phasic_activity_movie(block_movie, config)
    components, labels = segment_phasic_components(active, config)
    if not components.empty:
        components.insert(0, "phasic_z_threshold", config.phasic_z_threshold)
    return components, labels, processed, phasic_z
