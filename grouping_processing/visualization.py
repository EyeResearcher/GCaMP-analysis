from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.visualization import visualize_neuron_groups, plot_matrix_heatmap  # your existing functions

if TYPE_CHECKING:
    from data_classes.video import Video


Which = Literal["corr", "dtw"]


def _infer_img_size(video: "Video", default=(1024, 1024)) -> tuple[int, int]:
    ops = getattr(video, "suite2p_data", {}).get("ops", {}) if getattr(video, "suite2p_data", None) else {}
    Ly = int(ops.get("Ly", default[0]))
    Lx = int(ops.get("Lx", default[1]))
    return (Ly, Lx)


def make_matrix_heatmap(
    matrix: np.ndarray,
    *,
    title: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize=(6, 5),
) -> Optional[Figure]:
    if matrix is None:
        return None
    m = np.asarray(matrix)
    if m.ndim != 2 or m.size == 0:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    plot_matrix_heatmap(m, title=title, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, show_colorbar=True)
    fig.tight_layout()
    return fig


def visualize_grouping(
    video: "Video",
    *,
    which: Which = "corr",
    config_label: Optional[str] = None,
    heatmap_cmap: str = "viridis",
    heatmap_vmin: Optional[float] = None,
    heatmap_vmax: Optional[float] = None,
) -> Tuple[Optional[Figure], Optional[Figure]]:
    """
    Returns:
        (overlay_fig, heatmap_fig)

    - overlay_fig: neuron group overlay on field-of-view using Suite2p stat xpix/ypix
    - heatmap_fig: STTC/DTW matrix heatmap (method-agnostic)
    """
    if which == "corr":
        groups = getattr(video, "corr_groups", [])
        matrix = getattr(video, "corr_matrix", None)
        label = config_label or "corr_grouping"
        heat_title = f"Correlation matrix ({label})"
    else:
        groups = getattr(video, "dtw_groups", [])
        matrix = getattr(video, "dtw_matrix", None)
        label = config_label or "dtw_grouping"
        heat_title = f"DTW matrix ({label})"

    # Overlay
    overlay_fig: Optional[Figure] = None
    if groups:
        img_size = _infer_img_size(video)
        stat = getattr(video, "suite2p_data", {}).get("stat", np.array([]))
        overlay_fig = visualize_neuron_groups(
            neuron_groups=groups,
            stat=stat,
            img_size=img_size,
            video_path=getattr(video, "path", None),
            config_label=label,
        )

    # Heatmap
    heatmap_fig = make_matrix_heatmap(
        matrix,
        title=heat_title,
        cmap=heatmap_cmap,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )

    return overlay_fig, heatmap_fig
