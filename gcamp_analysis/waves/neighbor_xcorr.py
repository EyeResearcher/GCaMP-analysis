"""Neighbor-graph cross-correlation analysis of candidate calcium waves."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import detrend
from scipy.sparse import coo_matrix, csc_matrix, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr, splu
from scipy.spatial import cKDTree

from .analysis import (
    WaveAnalysisConfig,
    _benjamini_hochberg,
    _linear_r2,
    _positive_radial_scores,
    _radial_distance_grid,
    _refine_radial_origin,
)


@dataclass(frozen=True)
class NeighborXcorrConfig:
    """Settings for local lag estimation and graph reconstruction."""

    half_window_frames: int = 18
    smoothing_sigma_frames: float = 1.0
    max_lag_frames: int = 8
    neighbors_per_roi: int = 6
    max_neighbor_distance_um: float = 250.0
    min_peak_correlation: float = 0.20
    min_peak_sharpness: float = 0.03
    min_component_nodes: int = 20
    irls_iterations: int = 4
    propagation_null_repeats: int = 499
    null_distance_bins: int = 5
    null_reliability_bins: int = 3
    min_spatial_r2: float = 0.15
    alpha: float = 0.05
    random_seed: int = 2718


def build_neighbor_graph(
    coords_px: np.ndarray,
    *,
    pixel_size_um: float,
    neighbors_per_roi: int,
    max_distance_um: float,
) -> np.ndarray:
    """Return unique undirected k-nearest-neighbor edges."""
    coords = np.asarray(coords_px, dtype=float)
    if coords.shape[0] < 2:
        return np.empty((0, 2), dtype=int)
    k = min(neighbors_per_roi + 1, coords.shape[0])
    distances, neighbors = cKDTree(coords).query(coords, k=k)
    edges: set[tuple[int, int]] = set()
    max_distance_px = max_distance_um / pixel_size_um
    for i in range(coords.shape[0]):
        neighbor_values = np.atleast_1d(neighbors[i])
        distance_values = np.atleast_1d(distances[i])
        for j, distance in zip(neighbor_values[1:], distance_values[1:]):
            j = int(j)
            if j == i or float(distance) > max_distance_px:
                continue
            edges.add((min(i, j), max(i, j)))
    return np.asarray(sorted(edges), dtype=int)


def edge_cross_correlations(
    traces: np.ndarray,
    edges: np.ndarray,
    *,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate signed lag, peak correlation, and peak sharpness per edge.

    A positive lag means the second edge node follows the first.
    """
    traces = np.asarray(traces, dtype=float)
    edges = np.asarray(edges, dtype=int)
    lag_values = np.arange(-max_lag, max_lag + 1, dtype=int)
    correlations = np.empty((edges.shape[0], lag_values.size), dtype=float)
    for column, lag in enumerate(lag_values):
        if lag < 0:
            first = traces[edges[:, 0], -lag:]
            second = traces[edges[:, 1], : traces.shape[1] + lag]
        elif lag > 0:
            first = traces[edges[:, 0], : traces.shape[1] - lag]
            second = traces[edges[:, 1], lag:]
        else:
            first = traces[edges[:, 0]]
            second = traces[edges[:, 1]]
        first = first - first.mean(axis=1, keepdims=True)
        second = second - second.mean(axis=1, keepdims=True)
        denominator = np.sqrt(
            np.sum(first**2, axis=1) * np.sum(second**2, axis=1)
        )
        correlations[:, column] = np.divide(
            np.sum(first * second, axis=1),
            denominator,
            out=np.zeros(edges.shape[0], dtype=float),
            where=denominator > 0,
        )
    best_columns = np.argmax(correlations, axis=1)
    best_lags = lag_values[best_columns].astype(float)
    best_correlations = correlations[np.arange(edges.shape[0]), best_columns]
    sharpness = best_correlations - np.median(correlations, axis=1)
    return best_lags, best_correlations, sharpness


def _largest_reliable_component(
    n_nodes: int,
    edges: np.ndarray,
    lags: np.ndarray,
    correlations: np.ndarray,
    sharpness: np.ndarray,
    config: NeighborXcorrConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reliable = (
        (correlations >= config.min_peak_correlation)
        & (sharpness >= config.min_peak_sharpness)
    )
    edges = edges[reliable]
    lags = lags[reliable]
    correlations = correlations[reliable]
    sharpness = sharpness[reliable]
    if edges.size == 0:
        return (
            np.asarray([], dtype=int),
            np.empty((0, 2), dtype=int),
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
        )
    adjacency = coo_matrix(
        (
            np.ones(edges.shape[0] * 2),
            (
                np.r_[edges[:, 0], edges[:, 1]],
                np.r_[edges[:, 1], edges[:, 0]],
            ),
        ),
        shape=(n_nodes, n_nodes),
    )
    component_count, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels, minlength=component_count)
    largest_label = int(np.argmax(sizes))
    nodes = np.flatnonzero(labels == largest_label)
    in_component = np.isin(edges[:, 0], nodes) & np.isin(edges[:, 1], nodes)
    selected_edges = edges[in_component]
    node_map = np.full(n_nodes, -1, dtype=int)
    node_map[nodes] = np.arange(nodes.size)
    selected_edges = node_map[selected_edges]
    return (
        nodes,
        selected_edges,
        lags[in_component],
        correlations[in_component],
        sharpness[in_component],
    )


def _incidence_matrix(n_nodes: int, edges: np.ndarray) -> csc_matrix:
    rows = np.repeat(np.arange(edges.shape[0]), 2)
    columns = np.column_stack([edges[:, 0], edges[:, 1]]).reshape(-1)
    values = np.tile(np.asarray([-1.0, 1.0]), edges.shape[0])
    full = coo_matrix(
        (values, (rows, columns)), shape=(edges.shape[0], n_nodes)
    ).tocsc()
    return full[:, 1:]


def solve_graph_arrivals(
    n_nodes: int,
    edges: np.ndarray,
    lags: np.ndarray,
    base_weights: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robustly solve pairwise lag constraints into node arrival times."""
    incidence = _incidence_matrix(n_nodes, edges)
    weights = np.asarray(base_weights, dtype=float).copy()
    solution = np.zeros(n_nodes - 1, dtype=float)
    residuals = np.zeros(edges.shape[0], dtype=float)
    for _ in range(iterations):
        square_root = np.sqrt(np.maximum(weights, 1e-8))
        weighted_incidence = incidence.multiply(square_root[:, None])
        solution = lsqr(
            weighted_incidence,
            lags * square_root,
            atol=1e-9,
            btol=1e-9,
        )[0]
        residuals = incidence @ solution - lags
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        scale = max(0.5, 1.4826 * mad)
        huber = np.minimum(1.0, (1.5 * scale) / np.maximum(np.abs(residuals), 1e-8))
        weights = base_weights * huber
    arrivals = np.r_[0.0, solution]
    arrivals -= arrivals.mean()
    return arrivals, weights, residuals


def _spatial_scores(
    coords: np.ndarray,
    arrival_matrix: np.ndarray,
    wave_config: WaveAnalysisConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected, planar, and radial scores for arrival-time columns."""
    centered_coords = coords - coords.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(centered_coords, mode="reduced")
    values = arrival_matrix - arrival_matrix.mean(axis=0, keepdims=True)
    totals = np.sum(values**2, axis=0)
    explained = np.sum((q.T @ values) ** 2, axis=0)
    planar = np.divide(
        explained, totals, out=np.zeros_like(explained), where=totals > 0
    )
    _, distances = _radial_distance_grid(
        coords,
        wave_config.radial_grid_size,
        wave_config.radial_grid_margin_fraction,
    )
    radial = _positive_radial_scores(distances, arrival_matrix)
    return np.maximum(planar, radial), planar, radial


def graph_sign_flip_null(
    n_nodes: int,
    edges: np.ndarray,
    lags: np.ndarray,
    weights: np.ndarray,
    coords: np.ndarray,
    *,
    repeats: int,
    wave_config: WaveAnalysisConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate graph-aware spatial nulls by independently reversing edge lags."""
    incidence = _incidence_matrix(n_nodes, edges)
    weighted = incidence.multiply(weights[:, None])
    laplacian = incidence.T @ weighted
    laplacian = csc_matrix(laplacian + eye(laplacian.shape[0]) * 1e-8)
    factor = splu(laplacian)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(edges.shape[0], repeats))
    null_lags = lags[:, None] * signs
    rhs = incidence.T @ (weights[:, None] * null_lags)
    reduced = factor.solve(np.asarray(rhs))
    arrivals = np.vstack([np.zeros((1, repeats)), reduced])
    arrivals -= arrivals.mean(axis=0, keepdims=True)
    scores, _, _ = _spatial_scores(coords, arrivals, wave_config)
    return scores, arrivals


def _rank_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign approximately equal-count bins without duplicate-edge failures."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    bins = np.empty(values.size, dtype=int)
    bins[order] = np.minimum(
        n_bins - 1,
        np.arange(values.size) * n_bins // max(1, values.size),
    )
    return bins


def graph_stratified_lag_permutation_null(
    n_nodes: int,
    edges: np.ndarray,
    lags: np.ndarray,
    weights: np.ndarray,
    coords: np.ndarray,
    *,
    repeats: int,
    wave_config: WaveAnalysisConfig,
    rng: np.random.Generator,
    distance_bins: int = 5,
    reliability_bins: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute signed edge lags within distance/reliability strata.

    The spatial graph and edge weights remain fixed.  Permuting empirical
    signed lags breaks their large-scale spatial arrangement while retaining
    the lag distribution conditional on approximate edge length and
    reliability.
    """
    edges = np.asarray(edges, dtype=int)
    lags = np.asarray(lags, dtype=float)
    weights = np.asarray(weights, dtype=float)
    edge_distances = np.linalg.norm(
        coords[edges[:, 1]] - coords[edges[:, 0]], axis=1
    )
    distance_group = _rank_bins(edge_distances, distance_bins)
    reliability_group = _rank_bins(weights, reliability_bins)
    strata = distance_group * reliability_bins + reliability_group
    null_lags = np.empty((edges.shape[0], repeats), dtype=float)
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        for repeat in range(repeats):
            null_lags[indices, repeat] = lags[rng.permutation(indices)]

    incidence = _incidence_matrix(n_nodes, edges)
    weighted = incidence.multiply(weights[:, None])
    laplacian = incidence.T @ weighted
    laplacian = csc_matrix(laplacian + eye(laplacian.shape[0]) * 1e-8)
    factor = splu(laplacian)
    rhs = incidence.T @ (weights[:, None] * null_lags)
    reduced = factor.solve(np.asarray(rhs))
    arrivals = np.vstack([np.zeros((1, repeats)), reduced])
    arrivals -= arrivals.mean(axis=0, keepdims=True)
    scores, _, _ = _spatial_scores(coords, arrivals, wave_config)
    return scores, arrivals


def _fit_arrival_front(
    coords: np.ndarray,
    arrivals: np.ndarray,
    *,
    fs: float,
    pixel_size_um: float,
    wave_config: WaveAnalysisConfig,
) -> dict:
    centered = coords - coords.mean(axis=0, keepdims=True)
    planar_r2, planar_coefficients = _linear_r2(
        np.column_stack([centered, np.ones(coords.shape[0])]), arrivals
    )
    origins, distances = _radial_distance_grid(
        coords,
        wave_config.radial_grid_size,
        wave_config.radial_grid_margin_fraction,
    )
    radial_scores = _positive_radial_scores(distances, arrivals[:, None])
    radial_grid_r2 = float(radial_scores[0])
    centered_arrivals = arrivals - arrivals.mean()
    distance_centered = distances - distances.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.sum(distance_centered**2, axis=0) * np.sum(centered_arrivals**2)
    )
    correlations = np.divide(
        distance_centered.T @ centered_arrivals,
        denominator,
        out=np.zeros(distances.shape[1]),
        where=denominator > 0,
    )
    best_grid = int(np.argmax(np.maximum(correlations, 0.0)))
    radial_origin, radial_beta, _, radial_refined_r2 = _refine_radial_origin(
        coords, arrivals, origins[best_grid]
    )
    if radial_grid_r2 > planar_r2:
        model = "radial"
        score = radial_refined_r2
        source = radial_origin
        direction = math.nan
        slope = radial_beta
    else:
        model = "planar"
        score = planar_r2
        gradient = planar_coefficients[:2]
        slope = float(np.linalg.norm(gradient))
        if slope > 0:
            unit = gradient / slope
            source = coords[int(np.argmin(coords @ unit))]
            direction = float(np.degrees(np.arctan2(unit[1], unit[0])) % 360.0)
        else:
            source = coords.mean(axis=0)
            direction = math.nan
    speed = (
        float(pixel_size_um * fs / slope)
        if np.isfinite(slope) and slope > 0
        else math.nan
    )
    return {
        "xcorr_model": model,
        "xcorr_propagation_r2": float(score),
        "xcorr_selection_r2": float(max(planar_r2, radial_grid_r2)),
        "xcorr_planar_r2": float(planar_r2),
        "xcorr_radial_grid_r2": radial_grid_r2,
        "xcorr_source_x_px": float(source[0]),
        "xcorr_source_y_px": float(source[1]),
        "xcorr_direction_degrees": direction,
        "xcorr_speed_um_s": speed,
    }


def analyze_episode_neighbor_xcorr(
    row: pd.Series,
    *,
    config: NeighborXcorrConfig = NeighborXcorrConfig(),
) -> dict:
    """Analyze one previously detected population episode."""
    recording = Path(row["recording_path"])
    suite2p = recording / "suite2p" / "plane0"
    fluorescence = np.load(suite2p / "F.npy", mmap_mode="r")
    stat = np.load(suite2p / "stat.npy", allow_pickle=True)
    ops = np.load(suite2p / "ops.npy", allow_pickle=True).item()
    fs = float(ops.get("fs", 15.0))
    pixel_size = float(row.get("pixel_size_um", 1.0))
    roi_indices = np.asarray(json.loads(row["roi_indices"]), dtype=int)
    coords = np.asarray(
        [
            [float(stat[index]["med"][1]), float(stat[index]["med"][0])]
            for index in roi_indices
        ],
        dtype=float,
    )
    center = int(row["center_frame"])
    start = max(0, center - config.half_window_frames)
    stop = min(fluorescence.shape[1], center + config.half_window_frames + 1)
    traces = np.asarray(fluorescence[roi_indices, start:stop], dtype=float)
    traces = detrend(traces, axis=1, type="linear")
    traces = gaussian_filter1d(
        traces, sigma=config.smoothing_sigma_frames, axis=1
    )
    standard_deviations = traces.std(axis=1, keepdims=True)
    traces = np.divide(
        traces - traces.mean(axis=1, keepdims=True),
        standard_deviations,
        out=np.zeros_like(traces),
        where=standard_deviations > 0,
    )
    edges = build_neighbor_graph(
        coords,
        pixel_size_um=pixel_size,
        neighbors_per_roi=config.neighbors_per_roi,
        max_distance_um=config.max_neighbor_distance_um,
    )
    lags, correlations, sharpness = edge_cross_correlations(
        traces, edges, max_lag=config.max_lag_frames
    )
    nodes, component_edges, component_lags, component_correlations, component_sharpness = (
        _largest_reliable_component(
            len(roi_indices), edges, lags, correlations, sharpness, config
        )
    )
    base = {
        "xcorr_n_input_nodes": int(len(roi_indices)),
        "xcorr_n_graph_edges": int(len(edges)),
        "xcorr_reliable_edge_fraction": float(len(component_edges) / max(1, len(edges))),
        "xcorr_n_component_nodes": int(len(nodes)),
        "xcorr_n_component_edges": int(len(component_edges)),
    }
    if len(nodes) < config.min_component_nodes or len(component_edges) < len(nodes) - 1:
        return {**base, "xcorr_status": "insufficient_connected_graph"}
    weights = (
        np.maximum(component_correlations, 0.0) ** 2
        * np.maximum(component_sharpness, 0.01)
    )
    arrivals, robust_weights, residuals = solve_graph_arrivals(
        len(nodes),
        component_edges,
        component_lags,
        weights,
        iterations=config.irls_iterations,
    )
    component_coords = coords[nodes]
    predicted_lags = arrivals[component_edges[:, 1]] - arrivals[component_edges[:, 0]]
    total = float(
        np.sum(
            robust_weights
            * (component_lags - np.average(component_lags, weights=robust_weights)) ** 2
        )
    )
    residual_sum = float(
        np.sum(robust_weights * (component_lags - predicted_lags) ** 2)
    )
    edge_r2 = max(0.0, 1.0 - residual_sum / total) if total > 0 else 0.0
    wave_config = WaveAnalysisConfig(
        propagation_null_repeats=config.propagation_null_repeats
    )
    fit = _fit_arrival_front(
        component_coords,
        arrivals,
        fs=fs,
        pixel_size_um=pixel_size,
        wave_config=wave_config,
    )
    seed = (
        config.random_seed
        + sum(str(row["recording"]).encode("utf-8"))
        + int(row["center_frame"])
    )
    rng = np.random.default_rng(seed)
    sign_flip_scores, _ = graph_sign_flip_null(
        len(nodes),
        component_edges,
        component_lags,
        robust_weights,
        component_coords,
        repeats=config.propagation_null_repeats,
        wave_config=wave_config,
        rng=rng,
    )
    lag_permutation_scores, _ = graph_stratified_lag_permutation_null(
        len(nodes),
        component_edges,
        component_lags,
        robust_weights,
        component_coords,
        repeats=config.propagation_null_repeats,
        wave_config=wave_config,
        rng=rng,
        distance_bins=config.null_distance_bins,
        reliability_bins=config.null_reliability_bins,
    )
    observed = fit["xcorr_selection_r2"]
    sign_flip_p = (
        1.0 + float(np.sum(sign_flip_scores >= observed))
    ) / (sign_flip_scores.size + 1.0)
    lag_permutation_p = (
        1.0 + float(np.sum(lag_permutation_scores >= observed))
    ) / (
        lag_permutation_scores.size + 1.0
    )
    return {
        **base,
        **fit,
        "xcorr_status": "ok",
        "xcorr_peak_correlation_median": float(np.median(component_correlations)),
        "xcorr_peak_sharpness_median": float(np.median(component_sharpness)),
        "xcorr_edge_consistency_r2": edge_r2,
        "xcorr_edge_rmse_frames": float(np.sqrt(np.average(residuals**2, weights=robust_weights))),
        "xcorr_sign_flip_p": sign_flip_p,
        "xcorr_sign_flip_null_r2_95": float(
            np.quantile(sign_flip_scores, 0.95)
        ),
        "xcorr_lag_permutation_p": lag_permutation_p,
        "xcorr_lag_permutation_null_r2_95": float(
            np.quantile(lag_permutation_scores, 0.95)
        ),
        "xcorr_propagation_p": lag_permutation_p,
        "xcorr_null_r2_95": float(
            np.quantile(lag_permutation_scores, 0.95)
        ),
        "xcorr_component_roi_indices": json.dumps(roi_indices[nodes].tolist()),
        "xcorr_arrival_frames": json.dumps(arrivals.tolist()),
    }


def analyze_episode_table(
    episodes: pd.DataFrame,
    *,
    config: NeighborXcorrConfig = NeighborXcorrConfig(),
) -> pd.DataFrame:
    """Run graph cross-correlation over all supplied candidate episodes."""
    results = []
    for _, row in episodes.iterrows():
        results.append(analyze_episode_neighbor_xcorr(row, config=config))
    output = pd.concat(
        [episodes.reset_index(drop=True), pd.DataFrame(results)], axis=1
    )
    output["xcorr_propagation_q"] = np.nan
    ok = output["xcorr_status"] == "ok"
    for _, group in output[ok].groupby(["treatment", "recording"]):
        output.loc[group.index, "xcorr_propagation_q"] = _benjamini_hochberg(
            group["xcorr_propagation_p"]
        )
    output["xcorr_significant_wave"] = (
        ok
        & (output["xcorr_propagation_q"] <= config.alpha)
        & (output["xcorr_propagation_r2"] >= config.min_spatial_r2)
    )
    return output
