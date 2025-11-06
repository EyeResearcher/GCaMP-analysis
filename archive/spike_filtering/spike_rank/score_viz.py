# score_viz.py
from __future__ import annotations
from typing import Iterable, Optional, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ---------- Utilities ----------
def _interp_nans_inplace(y: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs in-place (copy outside if you want y preserved)."""
    y = np.array(y, dtype=float)
    n = len(y)
    if n == 0:
        return y
    nan = ~np.isfinite(y)
    if nan.any():
        idx = np.arange(n)
        y[nan] = np.interp(idx[nan], idx[~nan], y[~nan])
    return y


def _sorted_peaks(peaks):
    """Sort Peak objects by time index; assumes each has .index and .rank_score."""
    return sorted(peaks, key=lambda p: p.index)


def _infer_styles(peaks) -> List[str]:
    styles = set()
    for p in peaks:
        rs = getattr(p, "rank_score", {}) or {}
        styles.update(rs.keys())
    return sorted(styles)


# ---------- 1) Build a (peaks × styles) score matrix ----------
def make_score_matrix(peaks, styles: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Returns a DataFrame with rows = peak frame indices (time-ordered),
    columns = sorting styles, values = peak.rank_score[style] (float or NaN).
    """
    peaks_sorted = _sorted_peaks(peaks)
    if styles is None:
        styles = _infer_styles(peaks_sorted)

    rows = []
    for p in peaks_sorted:
        row = {"peak_index": int(p.index)}
        rs: Dict[str, Any] = getattr(p, "rank_score", {}) or {}
        for s in styles:
            row[s] = rs.get(s, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("peak_index")
    return df[[*styles]]  # ensure column order


def save_scores_csv(peaks, styles: Optional[Iterable[str]] = None, out_csv: str = "peak_rank_scores.csv") -> str:
    df = make_score_matrix(peaks, styles)
    df.to_csv(out_csv, float_format="%.6f")
    return out_csv


# ---------- 2) Heatmap of peak rank scores ----------
def plot_score_heatmap(
    peaks,
    styles: Optional[Iterable[str]] = None,
    title: str = "Peak Rank Scores",
    save_path: Optional[str] = None,
    dpi: int = 150,
    cmap: str = "viridis",
):
    df = make_score_matrix(peaks, styles)
    # numeric (preserve NaN)
    mat = df.to_numpy(dtype=float)

    plt.figure(figsize=(max(6, 0.5 * mat.shape[1] + 3), max(4, 0.06 * mat.shape[0] + 2)))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar(im, label="rank score")
    plt.yticks(np.arange(len(df.index)), df.index.values, fontsize=8)
    plt.xticks(np.arange(len(df.columns)), df.columns, rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Sorting style")
    plt.ylabel("Peak index (frame)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------- 3) Score vs time (one line per style) ----------
def plot_scores_over_time(
    peaks,
    styles: Optional[Iterable[str]] = None,
    title: str = "Rank Scores vs Time",
    save_path: Optional[str] = None,
    dpi: int = 150,
    markers: bool = True,
):
    df = make_score_matrix(peaks, styles)
    plt.figure(figsize=(12, 4))
    xs = df.index.values
    for s in df.columns:
        ys = df[s].to_numpy(dtype=float)
        if markers:
            plt.plot(xs, ys, marker="o", linewidth=1, markersize=3, label=s)
        else:
            plt.plot(xs, ys, linewidth=1.25, label=s)
    plt.xlabel("Peak index (frame)")
    plt.ylabel("Rank score")
    plt.title(title)
    plt.legend(ncol=min(3, len(df.columns)))
    plt.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------- 4) Distributions per style (hist or violin) ----------
def plot_score_distributions(
    peaks,
    styles: Optional[Iterable[str]] = None,
    kind: str = "hist",  # 'hist' or 'violin'
    bins: int = 20,
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    df = make_score_matrix(peaks, styles)
    n = len(df.columns)
    n = max(n, 1)
    plt.figure(figsize=(min(16, 4 * n), 3.5))
    for i, s in enumerate(df.columns, 1):
        plt.subplot(1, n, i)
        vals = df[s].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if kind == "hist":
            plt.hist(vals, bins=bins, alpha=0.85)
        else:  # violin
            if len(vals) == 0:
                plt.text(0.5, 0.5, "No data", ha="center", va="center")
                plt.xticks([])
            else:
                plt.violinplot(vals, showmeans=True)
                plt.xticks([])
        plt.title(s)
        plt.ylim(-0.05, 1.05)
    plt.suptitle("Peak Rank Score Distributions")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------- 5) Trace plot with labels for ONE style ----------
def plot_trace_with_style_labels(
    trace: np.ndarray,
    peaks,
    style: str,
    title_prefix: str = "Trace",
    save_path: Optional[str] = None,
    linewidth: float = 0.8,
    figsize: tuple = (16, 5),
    dpi: int = 150,
    fontsize: int = 10,
    label_every: int = 1,
):
    """
    Plot the trace (with NaNs interpolated for display) and label each peak with
    its rank score for `style`. Labels use point-offset so they show reliably.
    """
    x = np.arange(len(trace))
    y = _interp_nans_inplace(np.array(trace, dtype=float))
    peaks_sorted = _sorted_peaks(peaks)
    px = [int(p.index) for p in peaks_sorted]
    py = [y[i] for i in px]

    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, y, linewidth=linewidth, color="k")
    plt.scatter(px, py, s=22, zorder=3)

    for ordinal, p in enumerate(peaks_sorted, start=1):
        if (ordinal - 1) % label_every != 0:
            continue
        score = (getattr(p, "rank_score", {}) or {}).get(style, None)
        lab = f"{ordinal}:{score:.3f}" if isinstance(score, (int, float)) else f"{ordinal}:NA"
        plt.annotate(
            lab,
            xy=(p.index, y[p.index]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            clip_on=False,
            zorder=4,
        )

    plt.title(f"{title_prefix} — {style}")
    plt.xlabel("Frame")
    plt.ylabel("Cascade")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------- 6) Convenience: one trace PNG per style ----------
def plot_trace_per_style(
    trace: np.ndarray,
    peaks,
    styles: Optional[Iterable[str]] = None,
    title_prefix: str = "Trace",
    save_dir: Optional[str] = None,
    dpi: int = 150,
    linewidth: float = 0.8,
    figsize: tuple = (16, 5),
    fontsize: int = 10,
    label_every: int = 1,
):
    if styles is None:
        styles = _infer_styles(peaks)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for s in styles:
        save_path = None
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"trace_{s}.png")
        plot_trace_with_style_labels(
            trace,
            peaks,
            s,
            title_prefix=title_prefix,
            save_path=save_path,
            linewidth=linewidth,
            figsize=figsize,
            dpi=dpi,
            fontsize=fontsize,
            label_every=label_every,
        )
