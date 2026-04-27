"""Similarity / distance matrix functions.

Pure numerical functions — no strategy or clustering logic.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, List, Literal, Optional

import numpy as np
from numba import njit
from numba.typed import List as NumbaList
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

# ---- Similarity Matrix Combination ----


def compute_combined_similarities(*matrices: np.ndarray) -> np.ndarray:
    """Element-wise multiply similarity matrices and symmetrise the result.

    Each matrix is clipped to [0, 1] and NaN-filled before multiplication.
    The final product is symmetrised and has 1s on the diagonal.
    """
    if not matrices:
        raise ValueError("At least one matrix is required")
    out = None
    for mat in matrices:
        m = np.nan_to_num(np.asarray(mat, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        m = np.clip(m, 0.0, 1.0)
        out = m if out is None else out * m
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 1.0)
    return out


# ---- Max Cross-Correlation Similarity Matrix ----

def max_crosscorr_similarity(
    traces: np.ndarray,
    max_lag: int = 5,
    clip_negative: bool = True,
) -> np.ndarray:
    """Similarity matrix based on maximum cross-correlation within a lag window.

    Parameters
    ----------
    traces : array-like, shape (n_traces, n_timepoints)
        Fluorescence traces to compare.
    max_lag : int
        Maximum lag (in frames) to consider for cross-correlation.  The similarity
        between two traces is the maximum Pearson correlation obtained by shifting one trace relative to the other within this lag window.
    clip_negative : bool, default True
        If True, negative correlations are clipped to 0 (output range [0, 1]).
        If False, the full Pearson range is preserved (output range [-1, 1]).

    Returns
    -------
    sim : ndarray, shape (n_traces, n_traces)
        Similarity matrix based on maximum cross-correlation.
    """
    traces = np.asarray(traces, dtype=float)
    n_traces, n_timepoints = traces.shape

    x = traces - traces.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    valid = std.squeeze() > 0
    x = np.divide(x, std, out=np.zeros_like(x), where=std > 0)

    # Track the signed correlation that achieves the largest absolute value
    # across lags, so the "max" cross-correlation can be negative when
    # ``clip_negative`` is False.
    sim = np.zeros((n_traces, n_traces), dtype=float)
    best_abs = np.zeros((n_traces, n_traces), dtype=float)

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = x[:, -lag:]
            b = x[:, :n_timepoints + lag]
        elif lag > 0:
            a = x[:, :n_timepoints - lag]
            b = x[:, lag:]
        else:
            a = x
            b = x

        cc = (a @ b.T) / a.shape[1]
        if clip_negative:
            sim = np.maximum(sim, cc)
        else:
            mask = np.abs(cc) > best_abs
            sim = np.where(mask, cc, sim)
            best_abs = np.where(mask, np.abs(cc), best_abs)

    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    sim = np.clip(sim, -1, 1)
    if clip_negative:
        sim = np.clip(sim, 0, 1)
    sim = 0.5 * (sim + sim.T)
    np.fill_diagonal(sim, 1.0)

    sim[~valid, :] = 0.0
    sim[:, ~valid] = 0.0
    np.fill_diagonal(sim, 1.0)

    return sim

# ---- STTC Similarity Matrix (Numba JIT) ----

@njit
def _tiled_time_fraction(times, dt, t_start, t_stop):
    total = t_stop - t_start
    if total <= 0 or len(times) == 0:
        return 0.0

    start = max(t_start, times[0] - dt)
    end = min(t_stop, times[0] + dt)
    covered = 0.0

    for i in range(1, len(times)):
        s = max(t_start, times[i] - dt)
        e = min(t_stop, times[i] + dt)

        if s <= end:
            if e > end:
                end = e
        else:
            covered += end - start
            start = s
            end = e

    covered += end - start
    return covered / total


@njit
def _proportion_near(a, b, dt):
    if len(a) == 0 or len(b) == 0:
        return 0.0

    count = 0
    j = 0
    nb = len(b)

    for i in range(len(a)):
        ai = a[i]

        while j < nb and b[j] < ai - dt:
            j += 1

        if j < nb and abs(b[j] - ai) <= dt:
            count += 1

    return count / len(a)


@njit
def _sttc_pair(a, b, dt, t_start, t_stop):
    if len(a) == 0 or len(b) == 0:
        return np.nan

    TA = _tiled_time_fraction(a, dt, t_start, t_stop)
    TB = _tiled_time_fraction(b, dt, t_start, t_stop)
    PA = _proportion_near(a, b, dt)
    PB = _proportion_near(b, a, dt)

    term1_denom = 1.0 - PA * TB
    term2_denom = 1.0 - PB * TA

    term1 = 0.0 if term1_denom == 0 else (PA - TB) / term1_denom
    term2 = 0.0 if term2_denom == 0 else (PB - TA) / term2_denom

    return 0.5 * (term1 + term2)


@njit
def compute_sttc_matrix(trains : list[np.ndarray], dt: float, t_start: float, t_stop: float) -> np.ndarray:
    """Compute the STTC similarity matrix for a list of spike trains from active neurons.

    Parameters
    ----------
    trains : list of np.ndarray
        Each element is an array of spike times for a neuron.
    dt : float
        Time window (in seconds) for considering spikes as coincident.
    t_start : float
        Start time of the recording (in seconds).
    t_stop : float
        End time of the recording (in seconds).

    Raises
    --------
    _sttc_pair : (a, b, dt, t_start, t_stop) -> float
        Compute the STTC similarity for a single pair of spike trains.

    Returns
    -------
    np.ndarray
        STTC similarity matrix indexed by active neurons, not all neurons nor all ROIs.
    """
    trains_nb = NumbaList(trains)
    n = len(trains_nb)
    out = np.eye(n, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            val = _sttc_pair(trains_nb[i], trains_nb[j], dt, t_start, t_stop)
            out[i, j] = val
            out[j, i] = val

    return out



# ---- DTW Similarity Matrix (GPU-accelerated) ----

def compute_dtw_matrix_from_traces(
    traces: np.ndarray,
    downsample_factor: int = 3,
    use_gpu: bool = True,
) -> np.ndarray:
    """GPU-accelerated SoftDTW distance matrix from a trace matrix."""
    trace_rows = np.asarray(traces, dtype=float)
    if trace_rows.ndim != 2 or trace_rows.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    try:
        import torch
    except (ImportError, OSError) as e:
        logger.warning("PyTorch not available (%s) - skipping DTW", e.__class__.__name__)
        n_rows = int(trace_rows.shape[0])
        return np.zeros((n_rows, n_rows), dtype=np.float32)

    if use_gpu and not torch.cuda.is_available():
        logger.warning("GPU not available - skipping DTW to avoid hangups")
        n_rows = int(trace_rows.shape[0])
        return np.zeros((n_rows, n_rows), dtype=np.float32)

    processed = []
    for row in trace_rows:
        t = row[::downsample_factor] if downsample_factor > 1 else row
        processed.append((t - np.mean(t)) / (np.std(t) + 1e-8))

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    max_len = max(len(t) for t in processed)
    padded = np.zeros((len(processed), max_len), dtype=np.float32)
    for i, t in enumerate(processed):
        padded[i, : len(t)] = t

    return _soft_dtw_pairwise(torch.from_numpy(padded).to(device), device, gamma=1.0)

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

# ---- Light Evoked Response Alignment ----

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

