# batch_viz.py
from __future__ import annotations
import os, random
from typing import Iterable, Optional, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# import your pipeline pieces
from preprocess import find_peaks_and_valleys, assign_normalized_values, couple_peaks_to_valleys
from depth_ranking import compute_valley_depths, sort_and_rank_valleys
from rank_scoring import compute_all_peak_rank_scores


def _infer_styles_from_peaks(peaks) -> List[str]:
    styles = set()
    for p in peaks:
        rs = getattr(p, "rank_score", {}) or {}
        styles.update(rs.keys())
    return sorted(styles)


def _interp_nans(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    if n == 0:
        return y
    bad = ~np.isfinite(y)
    if bad.any():
        xi = np.arange(n)
        y[bad] = np.interp(xi[bad], xi[~bad], y[~bad])
    return y


def visualize_random_traces_per_style(
    cascade_array: np.ndarray,
    n: int = 10,
    sigma: float = 2.0,
    seed: Optional[int] = None,
    styles: Optional[Iterable[str]] = None,
    save_dir: Optional[str] = None,
    dpi: int = 150,
    linewidth: float = 0.8,
    fontsize: int = 10,
    label_every: int = 1,
):
    """
    Pick n random rows from a 2D cascade array and, for each row,
    produce one plot per sorting method. Each plot labels peaks with
    that method's rank score.

    Args:
        cascade_array: shape (n_rois, n_timepoints)
        n: number of random rows to visualize
        sigma: Gaussian sigma used in rank scoring
        seed: RNG seed for reproducibility
        styles: optional subset/order of styles to plot; if None, auto-infer
        save_dir: if provided, saves PNGs; otherwise shows figures
        dpi, linewidth, fontsize, label_every: figure aesthetics
    """
    if cascade_array.ndim != 2:
        raise ValueError("cascade_array must be 2D (ROIs × timepoints)")

    rng = random.Random(seed)
    rows = rng.sample(range(cascade_array.shape[0]), min(n, cascade_array.shape[0]))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for ridx in rows:
        trace_raw = cascade_array[ridx]
        trace = _interp_nans(trace_raw)

        # --- Pipeline on this row ---
        peaks, valleys = find_peaks_and_valleys(trace)
        peaks, valleys, _, _ = assign_normalized_values(peaks, valleys)
        peaks, valleys = couple_peaks_to_valleys(peaks, valleys)
        valleys = compute_valley_depths(valleys, trace)
        valleys, _ = sort_and_rank_valleys(valleys)
        valleys, peaks, styles = compute_all_peak_rank_scores(valleys, peaks, sigma)

        # figure out which styles we have
        styles_here = list(styles) if styles else _infer_styles_from_peaks(peaks)
        print(f"styles for ROI {ridx}: {styles_here}")
        if not styles_here:
            # nothing to plot
            continue

        # common x/peak positions
        x = np.arange(len(trace))
        peaks_sorted = sorted(peaks, key=lambda p: p.index)
        px = [int(p.index) for p in peaks_sorted]
        py = [float(trace[p.index]) for p in peaks_sorted]

        for style in styles_here:
            plt.figure(figsize=(32, 5), dpi=dpi)
            plt.plot(x, trace, linewidth=linewidth, color="k")
            plt.scatter(px, py, s=22, zorder=3)

            # label each peak with this style's score
            for ordinal, p in enumerate(peaks_sorted, start=1):
                if (ordinal - 1) % label_every != 0:
                    continue
                rs = getattr(p, "rank_score", {}) or {}
                val = rs.get(style, None)
                val = 0.0 if val is np.nan else val
                label = f"{val:3f}" if isinstance(val, (int, float)) and np.isfinite(val) else f"{ordinal}:NA"
                plt.annotate(
                    label,
                    xy=(p.index, trace[p.index]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                    clip_on=False,
                    zorder=4,
                )

            plt.title(f"ROI {ridx} — style: {style}  (σ={sigma})")
            plt.xlabel("Frame")
            plt.ylabel("Cascade")
            plt.tight_layout()

            if save_dir:
                fn = os.path.join(save_dir, f"roi_{ridx}_style_{style}.png")
                plt.savefig(fn, dpi=dpi, bbox_inches="tight")
                plt.close()
            else:
                plt.show()
