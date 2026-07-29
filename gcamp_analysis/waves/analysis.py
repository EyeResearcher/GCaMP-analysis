"""ROI-based detection of propagating retinal calcium activity.

The analysis deliberately separates two claims:

1. A candidate episode contains more near-coincident accepted calcium events
   than expected after independently circularly shifting every neuron's event
   train.
2. Activation times within that episode are more spatially ordered than
   expected after permuting activation times over the observed ROI positions.

The second test compares a planar traveling front with a radial expanding
front and uses the better fit while accounting for that model selection in
every permutation.
"""
from __future__ import annotations

import ast
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from gcamp_analysis.recording_discovery import parse_day


@dataclass(frozen=True)
class WaveAnalysisConfig:
    """Global, recording-independent analysis settings."""

    population_sigma_frames: float = 4.0
    derivative_event_quantile: float = 0.99
    derivative_event_min_separation_frames: int = 12
    candidate_half_window_frames: int = 18
    candidate_min_separation_frames: int = 24
    onset_smoothing_sigma_frames: float = 2.0
    min_participants: int = 20
    min_participant_fraction: float = 0.05
    min_spatial_coverage: float = 0.10
    min_onset_span_frames: int = 3
    min_propagation_r2: float = 0.15
    population_null_repeats: int = 200
    propagation_null_repeats: int = 499
    alpha: float = 0.05
    radial_grid_size: int = 21
    radial_grid_margin_fraction: float = 0.25
    recurrence_source_radius_um: float = 250.0
    recurrence_angle_degrees: float = 45.0
    movie_block_size_pixels: int = 16
    movie_activity_quantile: float = 0.50
    movie_propagation_null_repeats: int = 199
    random_seed: int = 1729


def discover_recordings(dataset_root: Path) -> pd.DataFrame:
    """Return recordings containing Suite2p fluorescence, one row per folder."""
    root = Path(dataset_root)
    rows: list[dict] = []
    for f_path in sorted(root.glob("*/*/suite2p/plane0/F.npy")):
        recording = f_path.parents[2]
        day = parse_day(recording.name)
        rows.append(
            {
                "recording_path": str(recording),
                "recording": recording.name,
                "treatment": recording.parent.name,
                "day": day,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["day", "treatment", "recording"], ascending=[False, True, True]
    ).reset_index(drop=True)


def _metrics_path(recording: Path) -> Path:
    expected = recording / "metrics" / f"{recording.name}_metrics.xlsx"
    if expected.exists():
        return expected
    candidates = sorted((recording / "metrics").glob("*_metrics.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"No metrics workbook found under {recording}")
    return candidates[0]


def _parse_event_list(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.asarray([], dtype=int)
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=int)


def _read_xml_metadata(recording: Path) -> tuple[float | None, float | None]:
    """Return (pixel_size_um, measured_fs), if OME metadata is present."""
    xml_paths = sorted(recording.glob("*.ome.xml")) or sorted(recording.glob("*.xml"))
    if not xml_paths:
        return None, None
    root = ET.parse(xml_paths[0]).getroot()
    pixels = next((node for node in root.iter() if node.tag.endswith("Pixels")), None)
    if pixels is None:
        return None, None
    pixel_size = pixels.attrib.get("PhysicalSizeX")
    delta_t = []
    for node in root.iter():
        if node.tag.endswith("Plane") and "DeltaT" in node.attrib:
            delta_t.append(float(node.attrib["DeltaT"]))
    measured_fs = None
    if len(delta_t) >= 2:
        differences = np.diff(np.asarray(delta_t, dtype=float))
        unit = next(
            (
                node.attrib.get("DeltaTUnit", "ms")
                for node in root.iter()
                if node.tag.endswith("Plane") and "DeltaT" in node.attrib
            ),
            "ms",
        )
        interval = float(np.median(differences))
        if interval > 0:
            measured_fs = (1000.0 / interval) if unit.lower() == "ms" else (1.0 / interval)
    return (float(pixel_size) if pixel_size is not None else None), measured_fs


def _load_recording(recording: Path) -> dict:
    suite2p = recording / "suite2p" / "plane0"
    fluorescence = np.load(suite2p / "F.npy", mmap_mode="r")
    stat = np.load(suite2p / "stat.npy", allow_pickle=True)
    ops = np.load(suite2p / "ops.npy", allow_pickle=True).item()
    summary = pd.read_excel(_metrics_path(recording), sheet_name="spike_summary")
    roi_indices = summary["neuron_idx"].astype(int).to_numpy()
    valid = (roi_indices >= 0) & (roi_indices < fluorescence.shape[0])
    summary = summary.loc[valid].reset_index(drop=True)
    roi_indices = summary["neuron_idx"].astype(int).to_numpy()
    coords = np.asarray(
        [[float(stat[index]["med"][1]), float(stat[index]["med"][0])] for index in roi_indices],
        dtype=float,
    )
    accepted_events = [_parse_event_list(value) for value in summary["spike_indices"]]
    pixel_size, measured_fs = _read_xml_metadata(recording)
    fs = float(ops.get("fs", measured_fs or 15.0))
    return {
        "F": fluorescence,
        "coords": coords,
        "accepted_events": accepted_events,
        "roi_indices": roi_indices,
        "n_frames": int(fluorescence.shape[1]),
        "fs": fs,
        "measured_fs": measured_fs,
        "pixel_size_um": pixel_size,
        "Ly": int(ops.get("Ly", 0)),
        "Lx": int(ops.get("Lx", 0)),
    }


def _event_density(
    events: list[np.ndarray], n_frames: int, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    raster = np.zeros((len(events), n_frames), dtype=np.uint8)
    for row, event_frames in enumerate(events):
        frames = event_frames[(event_frames >= 0) & (event_frames < n_frames)]
        raster[row, frames] = 1
    density = gaussian_filter1d(
        raster.sum(axis=0).astype(float), sigma=sigma, mode="constant"
    )
    return raster, density


def _population_null_maxima(
    events: list[np.ndarray],
    n_frames: int,
    sigma: float,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    maxima = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        counts = np.zeros(n_frames, dtype=float)
        for event_frames in events:
            if event_frames.size == 0:
                continue
            shifted = (event_frames + rng.integers(0, n_frames)) % n_frames
            np.add.at(counts, shifted, 1.0)
        maxima[repeat] = float(
            gaussian_filter1d(counts, sigma=sigma, mode="wrap").max()
        )
    return maxima


def _candidate_episodes(
    density: np.ndarray,
    null_maxima: np.ndarray,
    config: WaveAnalysisConfig,
) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(null_maxima, 1.0 - config.alpha))
    peaks, properties = find_peaks(
        density,
        height=threshold,
        distance=config.candidate_min_separation_frames,
        prominence=max(0.25, threshold * 0.05),
    )
    return peaks.astype(int), threshold


def _derivative_onset_events(
    fluorescence: np.ndarray,
    roi_indices: np.ndarray,
    config: WaveAnalysisConfig,
) -> list[np.ndarray]:
    """Detect strong rising edges using the same per-ROI z-score as ``z_f``.

    The package's accepted calcium-event peaks select analysis neurons, while
    these minimally smoothed derivative maxima provide the timing observable.
    """
    traces = np.asarray(fluorescence[roi_indices], dtype=float)
    means = traces.mean(axis=1, keepdims=True)
    standard_deviations = traces.std(axis=1, keepdims=True)
    zscore = np.divide(
        traces - means,
        standard_deviations,
        out=np.zeros_like(traces),
        where=standard_deviations > 0,
    )
    derivative = gaussian_filter1d(
        zscore,
        sigma=config.onset_smoothing_sigma_frames,
        order=1,
        axis=1,
        mode="nearest",
    )
    onset_events: list[np.ndarray] = []
    for row in derivative:
        threshold = float(np.quantile(row, config.derivative_event_quantile))
        peaks, _ = find_peaks(
            row,
            height=threshold,
            distance=config.derivative_event_min_separation_frames,
        )
        onset_events.append(peaks.astype(int))
    return onset_events


def _episode_arrivals(
    center: int,
    onset_events: list[np.ndarray],
    config: WaveAnalysisConfig,
) -> tuple[np.ndarray, np.ndarray]:
    participant_rows: list[int] = []
    onsets: list[int] = []
    low = center - config.candidate_half_window_frames
    high = center + config.candidate_half_window_frames
    for row, event_frames in enumerate(onset_events):
        within = event_frames[(event_frames >= low) & (event_frames <= high)]
        if within.size == 0:
            continue
        onset = int(within[np.argmin(np.abs(within - center))])
        participant_rows.append(row)
        onsets.append(onset)
    return np.asarray(participant_rows, dtype=int), np.asarray(onsets, dtype=float)


def _spatial_coverage(coords: np.ndarray, all_coords: np.ndarray) -> float:
    if coords.shape[0] < 3 or all_coords.shape[0] < 3:
        return 0.0
    try:
        total_area = float(ConvexHull(all_coords).volume)
        episode_area = float(ConvexHull(coords).volume)
    except Exception:
        return 0.0
    return episode_area / total_area if total_area > 0 else 0.0


def _linear_r2(design: np.ndarray, values: np.ndarray) -> tuple[float, np.ndarray]:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    prediction = design @ coefficients
    total = float(np.sum((values - values.mean()) ** 2))
    residual = float(np.sum((values - prediction) ** 2))
    score = max(0.0, 1.0 - residual / total) if total > 0 else 0.0
    return score, coefficients


def _radial_distance_grid(
    coords: np.ndarray,
    grid_size: int,
    margin_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    low = coords.min(axis=0)
    high = coords.max(axis=0)
    span = np.maximum(high - low, 1.0)
    xs = np.linspace(
        low[0] - margin_fraction * span[0],
        high[0] + margin_fraction * span[0],
        grid_size,
    )
    ys = np.linspace(
        low[1] - margin_fraction * span[1],
        high[1] + margin_fraction * span[1],
        grid_size,
    )
    origins = np.asarray([(x, y) for y in ys for x in xs], dtype=float)
    distances = np.sqrt(
        np.sum((coords[:, None, :] - origins[None, :, :]) ** 2, axis=2)
    )
    return origins, distances


def _positive_radial_scores(
    distances: np.ndarray, value_matrix: np.ndarray
) -> np.ndarray:
    """Maximum positive distance-time correlation squared for each value column."""
    d_centered = distances - distances.mean(axis=0, keepdims=True)
    d_norm = np.sqrt(np.sum(d_centered**2, axis=0, keepdims=True))
    d_standard = np.divide(
        d_centered,
        d_norm,
        out=np.zeros_like(d_centered),
        where=d_norm > 0,
    )
    values = value_matrix - value_matrix.mean(axis=0, keepdims=True)
    v_norm = np.sqrt(np.sum(values**2, axis=0, keepdims=True))
    values = np.divide(values, v_norm, out=np.zeros_like(values), where=v_norm > 0)
    correlations = d_standard.T @ values
    return np.maximum(correlations, 0.0).max(axis=0) ** 2


def _refine_radial_origin(
    coords: np.ndarray, values: np.ndarray, initial_origin: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    low = coords.min(axis=0)
    high = coords.max(axis=0)
    span = np.maximum(high - low, 1.0)
    bounds = [
        (low[0] - 0.5 * span[0], high[0] + 0.5 * span[0]),
        (low[1] - 0.5 * span[1], high[1] + 0.5 * span[1]),
    ]

    def objective(origin: np.ndarray) -> float:
        distances = np.linalg.norm(coords - origin[None, :], axis=1)
        design = np.column_stack([distances, np.ones(distances.size)])
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        if coefficients[0] < 0:
            return float(np.sum((values - values.mean()) ** 2))
        return float(np.sum((values - design @ coefficients) ** 2))

    result = minimize(
        objective,
        np.asarray(initial_origin, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
    )
    origin = np.asarray(result.x, dtype=float)
    distances = np.linalg.norm(coords - origin[None, :], axis=1)
    score, coefficients = _linear_r2(
        np.column_stack([distances, np.ones(distances.size)]), values
    )
    if coefficients[0] < 0:
        score = 0.0
    return origin, float(coefficients[0]), float(coefficients[1]), float(score)


def fit_propagation(
    coords: np.ndarray,
    onset_frames: np.ndarray,
    *,
    fs: float,
    pixel_size_um: float | None,
    config: WaveAnalysisConfig,
    rng: np.random.Generator,
) -> dict:
    """Fit planar/radial fronts and obtain a model-selection-aware permutation p."""
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(onset_frames, dtype=float)
    centered = coords - coords.mean(axis=0, keepdims=True)
    planar_design = np.column_stack([centered, np.ones(coords.shape[0])])
    planar_r2, planar_coef = _linear_r2(planar_design, values)

    origins, distances = _radial_distance_grid(
        coords, config.radial_grid_size, config.radial_grid_margin_fraction
    )
    radial_scores = _positive_radial_scores(distances, values[:, None])
    radial_grid_r2 = float(radial_scores[0])
    centered_values = values - values.mean()
    d_centered = distances - distances.mean(axis=0, keepdims=True)
    denom = np.sqrt(
        np.sum(d_centered**2, axis=0) * np.sum(centered_values**2)
    )
    correlations = np.divide(
        d_centered.T @ centered_values,
        denom,
        out=np.zeros(distances.shape[1], dtype=float),
        where=denom > 0,
    )
    best_grid = int(np.argmax(np.maximum(correlations, 0.0)))
    radial_origin, radial_beta, radial_intercept, radial_refined_r2 = (
        _refine_radial_origin(coords, values, origins[best_grid])
    )

    observed_score = max(planar_r2, radial_grid_r2)
    null_count = config.propagation_null_repeats
    permuted = np.column_stack(
        [rng.permutation(values) for _ in range(null_count)]
    )
    radial_null = _positive_radial_scores(distances, permuted)
    planar_null = np.empty(null_count, dtype=float)
    for index in range(null_count):
        planar_null[index], _ = _linear_r2(planar_design, permuted[:, index])
    null_scores = np.maximum(planar_null, radial_null)
    p_value = (1.0 + float(np.sum(null_scores >= observed_score))) / (
        null_count + 1.0
    )

    if radial_grid_r2 > planar_r2:
        model = "radial"
        score = radial_refined_r2
        source_x, source_y = radial_origin
        slope_frames_per_pixel = radial_beta
        direction_degrees = math.nan
    else:
        model = "planar"
        score = planar_r2
        gradient = planar_coef[:2]
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > 0:
            projection = coords @ (gradient / gradient_norm)
            source = coords[int(np.argmin(projection))]
            direction_degrees = float(
                np.degrees(np.arctan2(gradient[1], gradient[0])) % 360.0
            )
            slope_frames_per_pixel = gradient_norm
        else:
            source = coords.mean(axis=0)
            direction_degrees = math.nan
            slope_frames_per_pixel = math.nan
        source_x, source_y = source

    speed_um_s = math.nan
    if (
        pixel_size_um is not None
        and np.isfinite(slope_frames_per_pixel)
        and slope_frames_per_pixel > 0
    ):
        speed_um_s = float(pixel_size_um * fs / slope_frames_per_pixel)

    return {
        "model": model,
        "propagation_r2": float(score),
        "selection_score_r2": float(observed_score),
        "propagation_p": float(p_value),
        "source_x_px": float(source_x),
        "source_y_px": float(source_y),
        "direction_degrees": direction_degrees,
        "slope_frames_per_pixel": float(slope_frames_per_pixel),
        "speed_um_s": speed_um_s,
        "planar_r2": float(planar_r2),
        "radial_grid_r2": float(radial_grid_r2),
        "radial_refined_r2": float(radial_refined_r2),
        "null_score_95": float(np.quantile(null_scores, 0.95)),
    }


def _benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * p.size / np.arange(1, p.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _circular_angle_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _assign_recurrence_clusters(
    episodes: pd.DataFrame, config: WaveAnalysisConfig, pixel_size_um: float
) -> pd.DataFrame:
    episodes = episodes.copy()
    episodes["recurrence_cluster"] = -1
    significant = episodes.index[episodes["significant_wave"]].to_numpy()
    if significant.size < 2:
        return episodes
    rows = episodes.loc[significant]
    n = len(rows)
    distance = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            source_distance_um = pixel_size_um * math.hypot(
                rows.iloc[i]["source_x_px"] - rows.iloc[j]["source_x_px"],
                rows.iloc[i]["source_y_px"] - rows.iloc[j]["source_y_px"],
            )
            spatial_term = source_distance_um / config.recurrence_source_radius_um
            a = rows.iloc[i]["direction_degrees"]
            b = rows.iloc[j]["direction_degrees"]
            if np.isfinite(a) and np.isfinite(b):
                angle_term = _circular_angle_difference(a, b) / config.recurrence_angle_degrees
            elif rows.iloc[i]["model"] == rows.iloc[j]["model"] == "radial":
                angle_term = 0.0
            else:
                angle_term = 1.0
            value = math.hypot(spatial_term, angle_term)
            distance[i, j] = distance[j, i] = value
    labels = DBSCAN(eps=1.0, min_samples=2, metric="precomputed").fit_predict(distance)
    episodes.loc[significant, "recurrence_cluster"] = labels
    return episodes


def _plot_recording(
    recording_name: str,
    density: np.ndarray,
    threshold: float,
    fs: float,
    episodes: pd.DataFrame,
    coords: np.ndarray,
    output_path: Path,
) -> None:
    significant = episodes[episodes["significant_wave"]]
    n_maps = min(6, len(significant))
    figure = plt.figure(figsize=(14, 4 + 3 * math.ceil(max(1, n_maps) / 3)))
    grid = figure.add_gridspec(1 + math.ceil(max(1, n_maps) / 3), 3)
    axis = figure.add_subplot(grid[0, :])
    time = np.arange(density.size) / fs
    axis.plot(time, density, color="black", linewidth=1)
    axis.axhline(threshold, color="tab:red", linestyle="--", label="95% null maximum")
    for _, row in episodes.iterrows():
        color = "tab:green" if row["significant_wave"] else "tab:orange"
        axis.axvline(row["center_frame"] / fs, color=color, alpha=0.7)
    axis.set(title=f"{recording_name}: rising-edge population density", xlabel="Time (s)")
    axis.set_ylabel("Smoothed events/frame")
    axis.legend(loc="upper right")

    for plot_index, (_, row) in enumerate(significant.head(6).iterrows()):
        map_axis = figure.add_subplot(grid[1 + plot_index // 3, plot_index % 3])
        participating = np.asarray(json.loads(row["participant_rows"]), dtype=int)
        onsets = np.asarray(json.loads(row["onset_frames"]), dtype=float)
        scatter = map_axis.scatter(
            coords[participating, 0],
            coords[participating, 1],
            c=(onsets - onsets.min()) / fs,
            s=18,
            cmap="viridis",
        )
        map_axis.scatter(
            row["source_x_px"],
            row["source_y_px"],
            marker="*",
            s=160,
            c="red",
            edgecolors="white",
        )
        map_axis.invert_yaxis()
        map_axis.set_aspect("equal")
        map_axis.set_title(
            f"t={row['center_frame']/fs:.1f}s, {row['model']}, "
            f"R²={row['propagation_r2']:.2f}, q={row['propagation_q']:.3g}"
        )
        figure.colorbar(scatter, ax=map_axis, label="Onset delay (s)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _movie_path(recording: Path) -> Path | None:
    candidates = [
        path
        for path in sorted(recording.glob("*.tif*"))
        if "snap" not in path.stem.lower()
    ]
    return candidates[0] if candidates else None


def _block_mean_movie(movie: np.ndarray, block_size: int) -> np.ndarray:
    frames, height, width = movie.shape
    height = (height // block_size) * block_size
    width = (width // block_size) * block_size
    cropped = np.asarray(movie[:, :height, :width], dtype=np.float32)
    return cropped.reshape(
        frames,
        height // block_size,
        block_size,
        width // block_size,
        block_size,
    ).mean(axis=(2, 4))


def _validate_movie_episode(
    tif: tifffile.TiffFile,
    center_frame: int,
    *,
    fs: float,
    pixel_size_um: float,
    config: WaveAnalysisConfig,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Fit propagation independently to block-averaged movie fluorescence."""
    n_frames = len(tif.pages)
    start = max(0, center_frame - config.candidate_half_window_frames)
    stop = min(n_frames, center_frame + config.candidate_half_window_frames + 1)
    movie = tif.asarray(key=slice(start, stop))
    blocks = _block_mean_movie(movie, config.movie_block_size_pixels)
    traces = blocks.reshape(blocks.shape[0], -1).astype(float)
    means = traces.mean(axis=0, keepdims=True)
    deviations = traces.std(axis=0, keepdims=True)
    zscore = np.divide(
        traces - means,
        deviations,
        out=np.zeros_like(traces),
        where=deviations > 0,
    )
    # Remove frame-global illumination fluctuations while retaining a moving front.
    zscore -= np.median(zscore, axis=1, keepdims=True)
    derivative = gaussian_filter1d(
        zscore,
        sigma=config.onset_smoothing_sigma_frames,
        order=1,
        axis=0,
        mode="nearest",
    )
    amplitude = derivative.max(axis=0)
    threshold = float(np.quantile(amplitude, config.movie_activity_quantile))
    active = amplitude >= threshold
    rows, columns = np.indices(blocks.shape[1:])
    border = max(1, int(round(0.04 * min(blocks.shape[1:]))))
    interior = (
        (rows >= border)
        & (rows < blocks.shape[1] - border)
        & (columns >= border)
        & (columns < blocks.shape[2] - border)
    ).reshape(-1)
    active &= interior
    onset_local = np.argmax(derivative[:, active], axis=0)
    x = (columns.reshape(-1)[active] + 0.5) * config.movie_block_size_pixels
    y = (rows.reshape(-1)[active] + 0.5) * config.movie_block_size_pixels
    coords = np.column_stack([x, y]).astype(float)
    onsets = (start + onset_local).astype(float)
    movie_config = replace(
        config, propagation_null_repeats=config.movie_propagation_null_repeats
    )
    fit = fit_propagation(
        coords,
        onsets,
        fs=fs,
        pixel_size_um=pixel_size_um,
        config=movie_config,
        rng=rng,
    )
    result = {
        f"movie_{key}": value
        for key, value in fit.items()
        if key
        in {
            "model",
            "propagation_r2",
            "propagation_p",
            "source_x_px",
            "source_y_px",
            "direction_degrees",
            "speed_um_s",
        }
    }
    result["movie_n_active_blocks"] = int(active.sum())
    return result, coords, onsets


def validate_significant_movies(
    episodes: pd.DataFrame,
    output_dir: Path,
    config: WaveAnalysisConfig = WaveAnalysisConfig(),
) -> pd.DataFrame:
    """Independently fit movie fronts for every ROI-tested population episode."""
    if episodes.empty or "significant_wave" not in episodes:
        return episodes
    validated = episodes.copy()
    movie_columns = [
        "movie_model",
        "movie_propagation_r2",
        "movie_propagation_p",
        "movie_source_x_px",
        "movie_source_y_px",
        "movie_direction_degrees",
        "movie_speed_um_s",
        "movie_n_active_blocks",
    ]
    for column in movie_columns:
        validated[column] = pd.Series(
            [None] * len(validated), index=validated.index, dtype="object"
        ) if column == "movie_model" else np.nan
    for (treatment, recording_name), group in validated.groupby(
        ["treatment", "recording"]
    ):
        inventory_match = validated.loc[group.index, "recording_path"] if "recording_path" in validated else None
        if inventory_match is not None:
            recording = Path(inventory_match.iloc[0])
        else:
            # Per-episode tables omit the path; infer it from the inventory beside outputs.
            inventory = pd.read_csv(Path(output_dir) / "recording_inventory.csv")
            match = inventory[
                (inventory["treatment"] == treatment)
                & (inventory["recording"] == recording_name)
            ]
            if match.empty:
                continue
            recording = Path(match.iloc[0]["recording_path"])
        path = _movie_path(recording)
        if path is None:
            continue
        ops = np.load(recording / "suite2p" / "plane0" / "ops.npy", allow_pickle=True).item()
        fs = float(ops.get("fs", 15.0))
        pixel_size, _ = _read_xml_metadata(recording)
        if pixel_size is None:
            pixel_size = 1.0
        rng = np.random.default_rng(
            config.random_seed + sum(recording_name.encode("utf-8")) + 100_000
        )
        with tifffile.TiffFile(path) as tif:
            for index, row in group.iterrows():
                result, _, _ = _validate_movie_episode(
                    tif,
                    int(row["center_frame"]),
                    fs=fs,
                    pixel_size_um=pixel_size,
                    config=config,
                    rng=rng,
                )
                for column, value in result.items():
                    validated.loc[index, column] = value
    validated["movie_same_model"] = validated["model"] == validated["movie_model"]
    validated["movie_angle_difference_degrees"] = _circular_angle_difference(
        validated["direction_degrees"], validated["movie_direction_degrees"]
    )
    pixel_sizes = validated.get(
        "pixel_size_um", pd.Series(1.0, index=validated.index)
    )
    if np.isscalar(pixel_sizes):
        pixel_sizes = pd.Series(float(pixel_sizes), index=validated.index)
    pixel_sizes = pd.to_numeric(pixel_sizes, errors="coerce").fillna(1.0)
    validated["movie_source_difference_um"] = pixel_sizes * np.hypot(
        validated["source_x_px"] - validated["movie_source_x_px"],
        validated["source_y_px"] - validated["movie_source_y_px"],
    )
    planar_agreement = (
        (validated["model"] == "planar")
        & (validated["movie_angle_difference_degrees"] <= config.recurrence_angle_degrees)
    )
    radial_agreement = (
        (validated["model"] == "radial")
        & (validated["movie_source_difference_um"] <= config.recurrence_source_radius_um)
    )
    validated["movie_validated"] = (
        validated["significant_wave"].astype(bool)
        & validated["movie_same_model"].astype(bool)
        & (validated["movie_propagation_p"] <= config.alpha)
        & (planar_agreement | radial_agreement)
    )
    return validated


def analyze_recording(
    recording: Path,
    output_dir: Path,
    config: WaveAnalysisConfig = WaveAnalysisConfig(),
) -> tuple[pd.DataFrame, dict]:
    """Analyze one recording and write its episode table and diagnostic figure."""
    recording = Path(recording)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_recording(recording)
    seed_offset = sum(recording.name.encode("utf-8")) + sum(recording.parent.name.encode("utf-8"))
    rng = np.random.default_rng(config.random_seed + seed_offset)
    onset_events = _derivative_onset_events(data["F"], data["roi_indices"], config)
    _, density = _event_density(
        onset_events, data["n_frames"], config.population_sigma_frames
    )
    null_maxima = _population_null_maxima(
        onset_events,
        data["n_frames"],
        config.population_sigma_frames,
        config.population_null_repeats,
        rng,
    )
    centers, threshold = _candidate_episodes(density, null_maxima, config)
    rows: list[dict] = []
    minimum_participants = max(
        config.min_participants,
        int(math.ceil(config.min_participant_fraction * len(onset_events))),
    )
    for episode_id, center in enumerate(centers):
        participant_rows, onsets = _episode_arrivals(
            int(center),
            onset_events,
            config,
        )
        if participant_rows.size < minimum_participants:
            continue
        episode_coords = data["coords"][participant_rows]
        coverage = _spatial_coverage(episode_coords, data["coords"])
        onset_span = float(np.ptp(onsets))
        if coverage < config.min_spatial_coverage or onset_span < config.min_onset_span_frames:
            continue
        fit = fit_propagation(
            episode_coords,
            onsets,
            fs=data["fs"],
            pixel_size_um=data["pixel_size_um"],
            config=config,
            rng=rng,
        )
        population_p = (1.0 + float(np.sum(null_maxima >= density[center]))) / (
            null_maxima.size + 1.0
        )
        rows.append(
            {
                "episode_id": episode_id,
                "center_frame": int(center),
                "center_seconds": float(center / data["fs"]),
                "population_density": float(density[center]),
                "population_p_fwer": population_p,
                "n_participants": int(participant_rows.size),
                "participant_fraction": float(participant_rows.size / len(onset_events)),
                "spatial_coverage": float(coverage),
                "onset_span_frames": onset_span,
                "onset_span_seconds": float(onset_span / data["fs"]),
                "participant_rows": json.dumps(participant_rows.tolist()),
                "roi_indices": json.dumps(data["roi_indices"][participant_rows].tolist()),
                "onset_frames": json.dumps(onsets.astype(int).tolist()),
                **fit,
            }
        )
    episodes = pd.DataFrame(rows)
    if not episodes.empty:
        episodes["propagation_q"] = _benjamini_hochberg(episodes["propagation_p"])
        episodes["significant_wave"] = (
            (episodes["population_p_fwer"] <= config.alpha)
            & (episodes["propagation_q"] <= config.alpha)
            & (episodes["propagation_r2"] >= config.min_propagation_r2)
        )
        episodes = _assign_recurrence_clusters(
            episodes, config, data["pixel_size_um"] or 1.0
        )
    else:
        episodes = pd.DataFrame(
            columns=[
                "episode_id",
                "center_frame",
                "propagation_q",
                "significant_wave",
                "recurrence_cluster",
            ]
        )
    episodes.insert(0, "recording", recording.name)
    episodes.insert(0, "treatment", recording.parent.name)
    day = parse_day(recording.name)
    episodes.insert(0, "day", day)
    episodes.to_csv(output_dir / f"{recording.parent.name}_{recording.name}_episodes.csv", index=False)
    _plot_recording(
        recording.name,
        density,
        threshold,
        data["fs"],
        episodes,
        data["coords"],
        output_dir / f"{recording.parent.name}_{recording.name}_diagnostic.png",
    )
    recurrent = episodes.loc[
        episodes.get("recurrence_cluster", pd.Series(dtype=int)).ge(0)
    ]
    summary = {
        "day": day,
        "treatment": recording.parent.name,
        "recording": recording.name,
        "recording_path": str(recording),
        "n_frames": data["n_frames"],
        "duration_seconds": data["n_frames"] / data["fs"],
        "fs_ops": data["fs"],
        "fs_xml": data["measured_fs"],
        "pixel_size_um": data["pixel_size_um"],
        "n_analysis_neurons": len(onset_events),
        "mean_derivative_events_per_neuron": float(
            np.mean([len(values) for values in onset_events])
        ),
        "population_threshold": threshold,
        "n_population_candidates": int(len(centers)),
        "n_tested_episodes": int(len(episodes)),
        "n_significant_waves": int(episodes["significant_wave"].sum()) if len(episodes) else 0,
        "n_recurrent_wave_clusters": int(recurrent["recurrence_cluster"].nunique()) if len(recurrent) else 0,
        "n_waves_in_recurrent_clusters": int(len(recurrent)),
    }
    return episodes, summary


def analyze_dataset(
    dataset_root: Path,
    output_dir: Path,
    days: Iterable[int],
    config: WaveAnalysisConfig = WaveAnalysisConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze selected days and write combined episode/recording tables."""
    inventory = discover_recordings(dataset_root)
    selected = inventory[inventory["day"].isin([int(day) for day in days])]
    all_episodes: list[pd.DataFrame] = []
    summaries: list[dict] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in selected.itertuples(index=False):
        episodes, summary = analyze_recording(
            Path(row.recording_path), output_dir, config
        )
        all_episodes.append(episodes)
        summaries.append(summary)
    combined = pd.concat(all_episodes, ignore_index=True) if all_episodes else pd.DataFrame()
    summary_frame = pd.DataFrame(summaries)
    inventory.to_csv(output_dir / "recording_inventory.csv", index=False)
    if not combined.empty:
        path_map = inventory.set_index(["treatment", "recording"])["recording_path"]
        combined["recording_path"] = [
            path_map.get((row.treatment, row.recording), "")
            for row in combined.itertuples(index=False)
        ]
        # XML-derived pixel sizes are already in recording summaries.
        summary_pixel_map = summary_frame.set_index(["treatment", "recording"])[
            "pixel_size_um"
        ]
        combined["pixel_size_um"] = [
            summary_pixel_map.get((row.treatment, row.recording), np.nan)
            for row in combined.itertuples(index=False)
        ]
        combined = validate_significant_movies(combined, output_dir, config)
        movie_counts = (
            combined.groupby(["treatment", "recording"])["movie_validated"]
            .sum()
            .rename("n_movie_validated_waves")
            .reset_index()
        )
        summary_frame = summary_frame.merge(
            movie_counts, on=["treatment", "recording"], how="left"
        )
        summary_frame["n_movie_validated_waves"] = (
            summary_frame["n_movie_validated_waves"].fillna(0).astype(int)
        )
    combined.to_csv(output_dir / "all_wave_episodes.csv", index=False)
    summary_frame.to_csv(output_dir / "recording_wave_summary.csv", index=False)
    (output_dir / "analysis_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    return combined, summary_frame
