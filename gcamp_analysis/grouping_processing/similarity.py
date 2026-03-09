"""Similarity / distance matrix functions.

Pure numerical functions — no strategy or clustering logic.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.neuron import Neuron

logger = logging.getLogger(__name__)

CorrMethod = Literal["pearson", "spearman"]


# ── helpers ──────────────────────────────────────────────────────────


def _zscore_row(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, float)
    m, s = np.nanmean(x), np.nanstd(x)
    return np.zeros_like(x) if (not np.isfinite(s) or s < eps) else (x - m) / s


def _rankdata_1d(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _corr_pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a, b = a - np.mean(a), b - np.mean(b)
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return 0.0 if denom < eps else float(np.dot(a, b) / denom)


def _corr_spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _corr_pearson(_rankdata_1d(a), _rankdata_1d(b))


# ── public functions ─────────────────────────────────────────────────


def compute_correlation_matrix(
    traces: np.ndarray,
    *,
    method: CorrMethod = "pearson",
    remove_global: bool = True,
    use_diff: bool = True,
    diff_order: int = 1,
    zscore_each: bool = True,
    clip_negatives: bool = True,
) -> np.ndarray:
    """Pairwise correlation similarity matrix from fluorescence traces.

    Returns (N, N) in [0, 1] (if *clip_negatives*) or [-1, 1].
    """
    X = np.asarray(traces, float)
    if remove_global and X.size:
        X = X - np.nanmean(X, axis=0, keepdims=True)
    if use_diff:
        for _ in range(max(1, int(diff_order))):
            X = np.diff(X, axis=1)
    if zscore_each:
        X = np.vstack([_zscore_row(x) for x in X])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n = X.shape[0]
    S = np.eye(n, dtype=float)
    corr_fn = _corr_spearman if method == "spearman" else _corr_pearson
    for i in range(n):
        for j in range(i + 1, n):
            c = corr_fn(X[i], X[j])
            if clip_negatives and c < 0:
                c = 0.0
            S[i, j] = S[j, i] = c
    return S


def compute_sttc_matrix(
    neurons: List["Neuron"],
    n_frames: int,
    time_window: float = 0.033,
    fs: float = 15.0,
    max_frame: int | None = None,
) -> np.ndarray:
    """Spike Time Tiling Coefficient (Cutts & Eglen 2014) — fully vectorized.

    Parameters
    ----------
    max_frame : int, optional
        If provided, only spikes with ``sm_f_idx < max_frame`` are used.
        This enables baseline-only STTC in concatenated mode.

    Returns symmetric matrix in [-1, 1].
    """
    n = len(neurons)
    if n == 0:
        return np.array([[]])

    dt_frames = int(time_window * fs)

    spike_matrix = np.zeros((n, n_frames), dtype=np.float32)
    for i, neuron in enumerate(neurons):
        if hasattr(neuron, "spikes") and neuron.spikes:
            valid = [
                s.sm_f_idx for s in neuron.spikes
                if 0 <= s.sm_f_idx < n_frames
                and (max_frame is None or s.sm_f_idx < max_frame)
            ]
            if valid:
                spike_matrix[i, valid] = 1.0

    kernel = np.ones(2 * dt_frames + 1, dtype=np.float32)
    tiled_matrix = np.zeros((n, n_frames), dtype=np.float32)
    for i in range(n):
        if np.any(spike_matrix[i]):
            tiled_matrix[i] = (np.convolve(spike_matrix[i], kernel, mode="same") > 0).astype(np.float32)

    T = tiled_matrix.sum(axis=1) / n_frames
    n_spikes = spike_matrix.sum(axis=1)
    overlap = spike_matrix @ tiled_matrix.T

    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.nan_to_num(overlap / n_spikes[:, None], nan=0.0, posinf=0.0, neginf=0.0)

    T_row, T_col = T[None, :], T[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        term_A = np.nan_to_num((P - T_row) / (1.0 - P * T_row), nan=0.0, posinf=1.0, neginf=-1.0)
        term_B = np.nan_to_num((P.T - T_col) / (1.0 - P.T * T_col), nan=0.0, posinf=1.0, neginf=-1.0)

    sttc = np.clip(0.5 * (term_A + term_B), -1.0, 1.0)
    np.fill_diagonal(sttc, 1.0)
    no_spikes = n_spikes == 0
    sttc[no_spikes, :] = 0.0
    sttc[:, no_spikes] = 0.0
    np.fill_diagonal(sttc, 1.0)
    return sttc.astype(np.float32)


def compute_dtw_matrix(
    neurons: List["Neuron"],
    downsample_factor: int = 3,
    use_gpu: bool = True,
    max_frame: int | None = None,
) -> Optional[np.ndarray]:
    """GPU-accelerated SoftDTW distance matrix.  Returns *None* if GPU unavailable.

    Parameters
    ----------
    max_frame : int, optional
        If provided, only trace samples up to this frame index are used
        (baseline-only in concatenated mode).
    """
    try:
        import torch
    except (ImportError, OSError) as e:
        logger.warning("PyTorch not available (%s) — skipping DTW", e.__class__.__name__)
        return None

    if use_gpu and not torch.cuda.is_available():
        logger.warning("GPU not available — skipping DTW to avoid hangups")
        return None

    traces = []
    for neuron in neurons:
        raw = neuron.f_trace
        if max_frame is not None:
            raw = raw[:max_frame]
        t = raw[::downsample_factor] if downsample_factor > 1 else raw
        traces.append((t - np.mean(t)) / (np.std(t) + 1e-8))

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    max_len = max(len(t) for t in traces)
    padded = np.zeros((len(neurons), max_len), dtype=np.float32)
    for i, t in enumerate(traces):
        padded[i, : len(t)] = t

    return _soft_dtw_pairwise(torch.from_numpy(padded).to(device), device, gamma=1.0)


def _soft_dtw_pairwise(traces, device, gamma: float) -> np.ndarray:
    import torch

    n, T = traces.shape
    dist = np.zeros((n, n), dtype=np.float32)
    bs = 32
    for i in range(n):
        for j0 in range(i, n, bs):
            j1 = min(j0 + bs, n)
            B = j1 - j0
            cost = (traces[i : i + 1].unsqueeze(2) - traces[j0:j1].unsqueeze(1)) ** 2
            R = torch.full((B, T + 1, T + 1), float("inf"), device=device)
            R[:, 0, 0] = 0
            for ii in range(1, T + 1):
                for jj in range(1, T + 1):
                    r_prev = torch.stack([R[:, ii - 1, jj], R[:, ii, jj - 1], R[:, ii - 1, jj - 1]], dim=1)
                    R[:, ii, jj] = cost[:, ii - 1, jj - 1] - gamma * torch.logsumexp(-r_prev / gamma, dim=1)
            d = R[:, T, T].cpu().numpy()
            dist[i, j0:j1] = d
            dist[j0:j1, i] = d
    return dist
