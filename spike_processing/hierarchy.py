"""Peak clustering and hierarchy feature computation."""
from __future__ import annotations

import numpy as np


def build_peak_clusters(
    peaks: np.ndarray,
    widths: np.ndarray,
    width_factor: float = 1.5,
) -> list[np.ndarray]:
    """Group peaks into local clusters based on time proximity."""
    if peaks.size == 0:
        return []

    order = np.argsort(peaks)
    peaks_sorted = peaks[order]

    typical_width = float(np.median(widths)) if widths.size > 0 else 1.0
    if not np.isfinite(typical_width) or typical_width <= 0:
        typical_width = 1.0

    radius = width_factor * typical_width
    clusters: list[np.ndarray] = []
    current_cluster = [order[0]]

    for prev_idx, cur_idx in zip(order[:-1], order[1:]):
        if (peaks[cur_idx] - peaks[prev_idx]) <= radius:
            current_cluster.append(cur_idx)
        else:
            clusters.append(np.array(current_cluster, dtype=int))
            current_cluster = [cur_idx]

    clusters.append(np.array(current_cluster, dtype=int))
    return clusters


def compute_peak_hierarchy_features(
    peaks: np.ndarray,
    prominences: np.ndarray,
    widths: np.ndarray,
    width_factor: float = 1.5,
) -> dict:
    """Compute local hierarchy features for each peak."""
    n = peaks.size
    empty = {
        "dominance_score": np.array([], dtype=float),
        "local_rank": np.array([], dtype=int),
        "local_rank_norm": np.array([], dtype=float),
        "cluster_size": np.array([], dtype=int),
        "prom_gap": np.array([], dtype=float),
        "time_to_parent": np.array([], dtype=float),
    }
    if n == 0:
        return empty

    clusters = build_peak_clusters(peaks, widths, width_factor=width_factor)

    dominance_score = np.zeros(n, dtype=float)
    local_rank = np.zeros(n, dtype=int)
    local_rank_norm = np.zeros(n, dtype=float)
    cluster_size = np.zeros(n, dtype=int)
    prom_gap = np.zeros(n, dtype=float)
    time_to_parent = np.zeros(n, dtype=float)

    eps = 1e-9

    for cl in clusters:
        cl_prom = prominences[cl]
        cl_peaks = peaks[cl]

        parent_idx_in_cl = int(np.argmax(cl_prom))
        parent_prom = float(cl_prom[parent_idx_in_cl])
        parent_pos = int(cl_peaks[parent_idx_in_cl])

        rank_order = np.argsort(-cl_prom)
        rank_of = np.empty_like(rank_order)
        rank_of[rank_order] = np.arange(len(cl))

        for j, global_idx in enumerate(cl):
            r = int(rank_of[j])
            p = float(prominences[global_idx])

            cluster_size[global_idx] = len(cl)
            local_rank[global_idx] = r
            local_rank_norm[global_idx] = r / float(len(cl) - 1) if len(cl) > 1 else 0.0
            dominance_score[global_idx] = p / (parent_prom + eps)
            prom_gap[global_idx] = (parent_prom - p) / (parent_prom + eps)
            time_to_parent[global_idx] = abs(int(peaks[global_idx]) - parent_pos)

    return {
        "dominance_score": dominance_score,
        "local_rank": local_rank,
        "local_rank_norm": local_rank_norm,
        "cluster_size": cluster_size,
        "prom_gap": prom_gap,
        "time_to_parent": time_to_parent,
    }