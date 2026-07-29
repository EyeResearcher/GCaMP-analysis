from __future__ import annotations

import numpy as np

from gcamp_analysis.waves.neighbor_xcorr import (
    NeighborXcorrConfig,
    build_neighbor_graph,
    edge_cross_correlations,
    graph_sign_flip_null,
    graph_stratified_lag_permutation_null,
    solve_graph_arrivals,
)
from gcamp_analysis.waves.analysis import WaveAnalysisConfig


def test_edge_cross_correlation_recovers_signed_delay():
    time = np.arange(80)
    first = np.exp(-0.5 * ((time - 30) / 3) ** 2)
    second = np.exp(-0.5 * ((time - 35) / 3) ** 2)
    lags, correlations, _ = edge_cross_correlations(
        np.vstack([first, second]),
        np.asarray([[0, 1]]),
        max_lag=10,
    )
    assert lags[0] == 5
    assert correlations[0] > 0.99


def test_graph_solution_recovers_node_arrivals():
    arrivals = np.asarray([0.0, 2.0, 4.0, 6.0])
    edges = np.asarray([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]])
    lags = arrivals[edges[:, 1]] - arrivals[edges[:, 0]]
    solved, _, residuals = solve_graph_arrivals(
        4, edges, lags, np.ones(len(edges)), iterations=3
    )
    np.testing.assert_allclose(solved - solved[0], arrivals, atol=1e-6)
    assert np.max(np.abs(residuals)) < 1e-6


def test_sign_flip_null_is_below_coherent_planar_field():
    rng = np.random.default_rng(4)
    coords = rng.uniform(0, 500, size=(45, 2))
    edges = build_neighbor_graph(
        coords,
        pixel_size_um=1.0,
        neighbors_per_roi=6,
        max_distance_um=180,
    )
    true_arrivals = 0.025 * coords[:, 0]
    lags = true_arrivals[edges[:, 1]] - true_arrivals[edges[:, 0]]
    observed, _, _ = solve_graph_arrivals(
        len(coords), edges, lags, np.ones(len(edges)), iterations=2
    )
    centered = coords - coords.mean(axis=0, keepdims=True)
    design = np.column_stack([centered, np.ones(len(coords))])
    coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
    prediction = design @ coefficients
    observed_r2 = 1 - np.sum((observed - prediction) ** 2) / np.sum(
        (observed - observed.mean()) ** 2
    )
    null_scores, _ = graph_sign_flip_null(
        len(coords),
        edges,
        lags,
        np.ones(len(edges)),
        coords,
        repeats=99,
        wave_config=WaveAnalysisConfig(radial_grid_size=13),
        rng=np.random.default_rng(5),
    )
    assert observed_r2 > 0.99
    assert observed_r2 > np.quantile(null_scores, 0.99)


def test_stratified_lag_permutation_breaks_coherent_planar_field():
    rng = np.random.default_rng(14)
    coords = rng.uniform(0, 500, size=(60, 2))
    edges = build_neighbor_graph(
        coords,
        pixel_size_um=1.0,
        neighbors_per_roi=6,
        max_distance_um=180,
    )
    true_arrivals = 0.02 * coords[:, 0] - 0.01 * coords[:, 1]
    lags = true_arrivals[edges[:, 1]] - true_arrivals[edges[:, 0]]
    scores, _ = graph_stratified_lag_permutation_null(
        len(coords),
        edges,
        lags,
        np.ones(len(edges)),
        coords,
        repeats=99,
        wave_config=WaveAnalysisConfig(radial_grid_size=13),
        rng=np.random.default_rng(15),
    )
    assert np.quantile(scores, 0.99) < 0.9
