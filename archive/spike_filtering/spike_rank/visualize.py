import os
import numpy as np
import matplotlib.pyplot as plt

def plot_trace_by_style(
    trace,
    peaks,
    style: str,
    sigma_used: float = 2.0,
    save_dir: str | None = None,
    figsize=(12, 4),
    dpi=150,
    label_every: int = 1,
    fontsize: int = 10,
    headroom_frac: float = 0.12# extra space on top of y-limits
):
    title = style
    x = np.arange(len(trace))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, trace, lw=1.2)
    # Peaks in time order
    peaks_sorted = sorted(peaks, key=lambda p: p.index)
    peak_x = np.array([int(p.index) for p in peaks_sorted])
    peak_y = np.array([float(trace[p.index]) for p in peaks_sorted])
    ax.scatter(peak_x, peak_y, s=24, zorder=3)

    # Give headroom so labels aren’t clipped
    ymin, ymax = np.min(trace[40:-40]), np.max(trace[40:-40])
    yr = max(ymax - ymin, 1e-9)

    ax.set_ylim(ymin, ymax + headroom_frac * yr)
    print(f"Style : {style}")
    # Label each peak with its score for this style
    for ordinal, p in enumerate(peaks_sorted, start=1):
        if (ordinal - 1) % label_every != 0:
            continue

        score = (p.rank_score or {}).get(style, None)
        txt = f"{score:.3f}" if isinstance(score, (int, float)) else f"{ordinal}:NA"
        print(f"\tPeak {p.index}: {txt}")
        # Put text 6 points above the marker, independent of data scale
        ax.annotate(
            txt,
            xy=(p.index, trace[p.index]),
            xytext=(0, 6),                  # 6 points up
            textcoords="offset points",     # <- screen-space offset
            ha="center",
            va="bottom",
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            clip_on=False,                  # don't clip at axes border
            zorder=4,
        )

    ax.set_title(f"Sorting style: {title} | σ={sigma_used}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cascade")
    ax.grid(alpha=0.2)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"trace_{title}.png")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {out}")
        plt.close(fig)
    else:
        plt.show()


def plot_all_styles(trace, peaks, styles, sigma_used=2.0, save_dir="figs", dpi=150):
    """
    Call plot_trace_by_style once per style (each as its own figure).
    """
    os.makedirs(save_dir, exist_ok=True)
    for style in styles:
        plot_trace_by_style(
            trace,
            peaks,
            style,
            sigma_used=sigma_used,
            save_dir=save_dir,
            dpi=dpi
        )
