from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

CorrMethod = Literal["pearson", "spearman"]


def _zscore_row(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < eps:
        return np.zeros_like(x)
    return (x - m) / s


def _rankdata_1d(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _corr_pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def _corr_spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata_1d(a)
    rb = _rankdata_1d(b)
    return _corr_pearson(ra, rb)


@dataclass
class TraceCorrelationSimilarity:
    """
    Similarity matrix based on fluorescence trace correlation.

    Goals:
      - focus on co-fluctuations (use_diff)
      - reduce global shared signal effects (remove_global)
      - make scale comparable across neurons (zscore_each)
      - ignore negative correlations (clip_negatives=True)
    """
    method: CorrMethod = "pearson"
    remove_global: bool = True
    use_diff: bool = True
    diff_order: int = 1
    zscore_each: bool = True
    clip_negatives: bool = True

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)

        # optional global signal removal (per-frame mean across neurons)
        if self.remove_global and X.size:
            X = X - np.nanmean(X, axis=0, keepdims=True)

        # focus on changes
        if self.use_diff:
            for _ in range(max(1, int(self.diff_order))):
                X = np.diff(X, axis=1)

        if self.zscore_each:
            X = np.vstack([_zscore_row(x) for x in X])

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def compute(self, traces: np.ndarray) -> np.ndarray:
        X = self._preprocess(traces)
        n = X.shape[0]
        S = np.eye(n, dtype=float)

        corr_fn = _corr_spearman if self.method == "spearman" else _corr_pearson

        for i in range(n):
            for j in range(i + 1, n):
                c = corr_fn(X[i], X[j])
                if self.clip_negatives and c < 0:
                    c = 0.0
                S[i, j] = S[j, i] = c

        return S
