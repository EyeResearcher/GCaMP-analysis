"""Similarity / distance matrix functions.

Pure numerical functions — no strategy or clustering logic.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, List, Literal, Optional

import numpy as np
from scipy.signal import find_peaks
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

def _corr_weighted_pearson(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Weighted Pearson correlation that emphasizes active frames.

    Drop-in replacement for _corr_pearson(x, y).

    Weights are based on the pairwise activity magnitude so frames where either
    trace is strongly active contribute more than quiet baseline frames.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return 0.0

    x = x[valid]
    y = y[valid]

    if x.size < 2:
        return 0.0
    
    w = np.maximum(np.abs(x), np.abs(y)) ** 2

    w_sum = np.sum(w)
    if w_sum <= eps:
        return 0.0

    mx = np.sum(w * x) / w_sum
    my = np.sum(w * y) / w_sum

    xc = x - mx
    yc = y - my

    num = np.sum(w * xc * yc)
    den = np.sqrt(np.sum(w * xc * xc) * np.sum(w * yc * yc))

    if den <= eps:
        return 0.0

    c = num / den

    # Numerical safety
    if not np.isfinite(c):
        return 0.0
    return float(np.clip(c, -1.0, 1.0))

CORR_REGISTRY = {"pearson": _corr_pearson, "spearman": _corr_spearman, "weighted_pearson": _corr_weighted_pearson}

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
    corr_fn = CORR_REGISTRY.get(method)
    if corr_fn is None:
        raise ValueError(f"Unknown correlation method {method!r}. Available: {list(CORR_REGISTRY.keys())}")
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
) -> np.ndarray:
    """GPU-accelerated SoftDTW distance matrix.

    Returns a zero matrix when PyTorch / GPU is unavailable.

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
        return np.zeros((len(neurons), len(neurons)), dtype=np.float32)

    if use_gpu and not torch.cuda.is_available():
        logger.warning("GPU not available — skipping DTW to avoid hangups")
        return np.zeros((len(neurons), len(neurons)), dtype=np.float32)

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

def _enforce_single_direction(activated: np.ndarray) -> None:
    """Zero out minority-direction peaks so each neuron is purely ON or OFF.

    After per-bin deconfliction a neuron may still have a mix of +1 and -1
    entries across different bins.  This pass keeps only the dominant
    direction and removes any residual opposite-sign peaks.  Operates
    **in-place** on *activated*.
    """
    for i in range(activated.shape[0]):
        total_on = np.sum(activated[i] > 0)
        total_off = np.sum(activated[i] < 0)
        if total_on > total_off:
            activated[i][activated[i] < 0] = 0.0
        elif total_off > total_on:
            activated[i][activated[i] > 0] = 0.0


def _deconflict_bins(
    activated: np.ndarray,
    diff_trace: np.ndarray,
    bin_ranges: list[tuple[int, int]],
) -> None:
    """Resolve simultaneous ON/OFF peaks in every pulse bin.

    For each ``(lo, hi)`` bin and each neuron, if both positive and
    negative diff-peaks are present, keep only the direction with the
    larger magnitude in ``diff_trace`` and zero out the other.
    Operates **in-place** on *activated*.
    """
    n_neurons = activated.shape[0]
    for lo, hi in bin_ranges:
        for i in range(n_neurons):
            window = activated[i, lo:hi]
            on_positions = np.where(window > 0)[0] + lo
            off_positions = np.where(window < 0)[0] + lo
            if len(on_positions) == 0 or len(off_positions) == 0:
                continue
            on_pos_clipped = np.clip(on_positions, 0, diff_trace.shape[1] - 1)
            off_pos_clipped = np.clip(off_positions, 0, diff_trace.shape[1] - 1)
            max_on_mag = np.max(diff_trace[i, on_pos_clipped])
            max_off_mag = np.max(-diff_trace[i, off_pos_clipped])
            if max_on_mag >= max_off_mag:
                window[window < 0] = 0.0
            else:
                window[window > 0] = 0.0


def align_light_evoked(
    sm_norm_f: np.ndarray,
    bin_size: int,
    schedule: list[int],
    n_frames: int,
    *,
    prominence: float | None = None,
) -> np.ndarray:
    """Detect ON/OFF light-evoked responses aligned to a pulse schedule.

    For each pulse bin, if both an ON peak (positive diff) and an OFF peak
    (negative diff) are detected for the same neuron, only the one with the
    larger magnitude in the derivative is kept — the stronger response is
    the true stimulus response, and the weaker one is a recovery artifact
    or noise.

    Parameters
    ----------
    sm_norm_f : ndarray, shape (n_neurons, n_frames)
        Smoothed, normalised fluorescence traces.
    bin_size : int
        Width (in frames) of the window starting at each pulse frame.
    schedule : list[int]
        Pulse onset frame indices.
    n_frames : int
        Total number of frames.
    prominence : float or None
        Minimum prominence for ``find_peaks`` on the derivative trace.
        Raising this value filters out low-amplitude noise peaks.  Use
        ``None`` (default) for no prominence filtering.

    Returns
    -------
    ndarray, shape (n_neurons, n_frames)
        +1 (ON), -1 (OFF), or 0.
    """
    pulses = np.zeros(n_frames)
    bin_ranges = []
    for pulse in schedule:
        lo = max(0, pulse)
        hi = min(n_frames, pulse + bin_size)
        bin_ranges.append((lo, hi))
        for f in range(lo, hi):
            pulses[f] = 1.0

    diff_trace = np.diff(sm_norm_f, axis=1, prepend=sm_norm_f[:, :1])
    peak_kw: dict = {} if prominence is None else {"prominence": prominence}
    on_peaks = [find_peaks(diff_trace[i], **peak_kw)[0] for i in range(diff_trace.shape[0])]
    off_peaks = [find_peaks(-diff_trace[i], **peak_kw)[0] for i in range(diff_trace.shape[0])]

    train_trace = np.zeros_like(sm_norm_f)
    for i, peaks in enumerate(on_peaks):
        train_trace[i, peaks] = 1.0
    for i, peaks in enumerate(off_peaks):
        train_trace[i, peaks] = -1.0

    activated = pulses[np.newaxis, :] * train_trace

    _deconflict_bins(activated, diff_trace, bin_ranges)
    _enforce_single_direction(activated)

    return activated

