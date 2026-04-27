
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from typing import List, TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.neuron_group import NeuronGroup
import logging
logger = logging.getLogger(__name__)



def _safe_window_get(windows: Any, key: str, default: Optional[int] = None) -> Optional[int]:
    """Safely extract an integer from a possibly-dict windows object."""
    if isinstance(windows, dict):
        v = windows.get(key, default)
        try:
            return int(v) if v is not None else default
        except Exception:
            return default
    return default


def plot_trace_with_spikes(
    ax: "plt.Axes",
    y: np.ndarray,
    *,
    spike_idx: int,
    all_spike_indices: list[int],
    title: str,
    y_label: str,
    windows: list[np.ndarray] ,
    ylim : tuple[float, float]) -> None:
    """Full-trace plot with spike markers and window shading.
    
    Parameters
    ----------
    ax : matplotlib Axes
    y : 1-D signal array
    spike_idx : index of the current spike to highlight
    all_spike_indices : all spike peak indices to mark
    title, y_label : axis labels
    windows : dict with optional keys 'left_base', 'right_base', 'prev_min', 'next_min'
    """
    if windows is None:
        windows = {}

    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)

    ax.clear()
    ax.plot(x, y, linewidth=1)

    left_base = _safe_window_get(windows, "left_base", None)
    right_base = _safe_window_get(windows, "right_base", None)
    prev_min = _safe_window_get(windows, "prev_min", None)
    next_min = _safe_window_get(windows, "next_min", None)

    # Shade large window
    if left_base is not None and right_base is not None:
        lb = max(0, min(left_base, len(y) - 1))
        rb = max(0, min(right_base, len(y)))
        if rb > lb:
            ax.fill_between(x[lb:rb], y[lb:rb], alpha=0.15)

    # Thicken small window
    if prev_min is not None and next_min is not None:
        pm = max(0, min(prev_min, len(y) - 1))
        nm = max(0, min(next_min, len(y)))
        if nm > pm:
            ax.plot(x[pm:nm], y[pm:nm], linewidth=2)

    # All spike peaks
    
    for other_idx in all_spike_indices:
        oi = int(other_idx)
        if 0 <= oi < len(y):
            ax.plot([oi, oi], [ylim[0], y[oi]], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Current spike peak
    si = int(spike_idx)
    if 0 <= si < len(y):
        ax.plot([si, si], [ylim[0], y[si]], color="red", linestyle="-", linewidth=2, label="Current spike")

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(y_label)
    ax.set_xlim(0, len(y))
    ax.set_ylim(ylim)  # Keep consistent y-limits after adding lines
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")


def visualize_neuron_groups(neuron_groups: List[NeuronGroup], 
                            stat: np.ndarray, 
                            img_size: tuple = (512, 512), 
                            video_path: Path = None,
                            config_label: str = None,
                            save_path: Path = None):
    """
    Visualize neuron groups in a color-coordinated fashion.
    
    Each neuron group is assigned a unique color. Neurons are drawn using their
    xpix/ypix coordinates from the stat array.
    
    Args:
        neuron_groups (List[NeuronGroup]): List of neuron groups to visualize
        stat (np.ndarray): Array of stat dictionaries from Suite2p (one per ROI)
        img_size (tuple): Size of the output image (height, width)
        video_path (Path): Path to video directory for saving output
        config_label (str): Optional label for the configuration (e.g., "tw0.033_dt0.3")
        
    Returns:
        Path: Path to saved image
    """
    if not neuron_groups:
        logger.warning("No neuron groups to visualize")
        return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Create blank image (dark background)
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
    
    # Generate distinct colors for each group
    cmap = plt.cm.get_cmap('tab20', max(len(neuron_groups), 1))
    colors = [cmap(i)[:3] for i in range(len(neuron_groups))]  # RGB only
    
    # Track group info for legend
    group_info = []
    
    # Draw each neuron group
    for group_idx, group in enumerate(neuron_groups):
        color = colors[group_idx]
        n_neurons_drawn = 0
        
        # Get neuron indices from group
        neuron_indices = group.neuron_indices if hasattr(group, 'neuron_indices') else []
        
        # Get average correlation from pre-computed stats
        avg_corr = group.mean_spk_stats.get('mean_corr', None) if hasattr(group, 'mean_spk_stats') else None
        
        for neuron_idx in neuron_indices:
            # Get stat dict for this neuron
            if neuron_idx >= len(stat):
                logger.warning(f"Neuron index {neuron_idx} out of range for stat array (len={len(stat)})")
                continue
            
            neuron_stat = stat[neuron_idx]
            
            # Get pixel coordinates
            if 'ypix' not in neuron_stat or 'xpix' not in neuron_stat:
                logger.warning(f"Neuron {neuron_idx} missing ypix/xpix in stat")
                continue
            
            ypix = np.array(neuron_stat['ypix'])
            xpix = np.array(neuron_stat['xpix'])
            
            # Filter pixels within image bounds
            valid_mask = (ypix >= 0) & (ypix < img_size[0]) & (xpix >= 0) & (xpix < img_size[1])
            ypix = ypix[valid_mask]
            xpix = xpix[valid_mask]
            
            if len(ypix) == 0:
                continue
            
            # Color these pixels
            img[ypix, xpix, 0] = color[0]
            img[ypix, xpix, 1] = color[1]
            img[ypix, xpix, 2] = color[2]
            
            n_neurons_drawn += 1
        
        group_info.append({
            'group_id': group.group_id if hasattr(group, 'group_id') else f'group_{group_idx}',
            'color': color,
            'n_neurons': len(neuron_indices),
            'n_drawn': n_neurons_drawn,
            'avg_corr': avg_corr
        })
    
    # Display image
    ax.imshow(img, origin='upper')
    ax.set_xlim(0, img_size[1])
    ax.set_ylim(img_size[0], 0)  # Flip y-axis to match image coordinates
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    title = f"Neuron Groups (n={len(neuron_groups)})"
    if config_label:
        title += f"\n{config_label}"
    ax.set_title(title, fontsize=14, fontweight='bold', color='white')
    
    # Create legend with avg correlation
    if group_info:
        legend_elements = []
        for info in group_info:
            # Build label with optional avg correlation
            if info['avg_corr'] is not None and not np.isnan(info['avg_corr']):
                label = f"{info['group_id']} (n={info['n_neurons']}, corr={info['avg_corr']:.2f})"
            else:
                label = f"{info['group_id']} (n={info['n_neurons']})"
            
            legend_elements.append(
                Patch(
                    facecolor=info['color'],
                    edgecolor='white',
                    linewidth=0.5,
                    label=label
                )
            )
        
        # Place legend outside plot
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            framealpha=0.9,
            facecolor='black',
            labelcolor='white'
        )
    
    # Adjust layout to fit legend
    plt.tight_layout()
    
    # Resolve save path: explicit save_path takes precedence, then video_path
    _save_path = save_path
    if _save_path is None and video_path is not None:
        output_dir = Path(video_path) / 'metrics'
        output_dir.mkdir(exist_ok=True, parents=True)
        if config_label:
            _save_path = output_dir / f'neuron_groups_{config_label}.png'
        else:
            _save_path = output_dir / 'neuron_groups.png'

    if _save_path is not None:
        fig.savefig(_save_path, dpi=150, bbox_inches='tight', facecolor='black')
        logger.info(f"Saved neuron group visualization to {_save_path}")

    plt.close(fig)
    
    return fig


def plot_matrix_heatmap(
    matrix: np.ndarray,
    *,
    labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
) -> plt.Axes:
    """
    Plot any 2D matrix (STTC, DTW, etc.) as a heatmap.
    Returns the Axes the heatmap was drawn on.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)

    if title:
        ax.set_title(title)

    ax.set_xlabel("Index")
    ax.set_ylabel("Index")

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax


# =====================================================================
#  Treatment comparison spatial plots
# =====================================================================

_STATUS_COLORS = {
    "grouped": "#2ca02c",
    "ungrouped": "#d62728",
    "inactive": "#7f7f7f",
}


def plot_delta_corr_vs_dispersion(
    group_metrics: List[dict],
    *,
    ax: Optional[plt.Axes] = None,
    exclude_inactive_groups: bool = True,
    title: str = "ΔCorrelation vs. Baseline Spatial Dispersion",
) -> plt.Axes:
    """Scatter plot of delta_mean_corr vs baseline_mean_pairwise_dist.

    Each point is one baseline group. Groups where all neurons became
    inactive in treatment are excluded (or labelled) based on
    *exclude_inactive_groups*.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    xs, ys, sizes, colors = [], [], [], []
    for gm in group_metrics:
        x = gm.get("baseline_mean_pairwise_dist")
        y = gm.get("delta_mean_corr")
        n_active = gm.get("n_treatment_active", gm.get("n_neurons", 0))
        if x is None or y is None:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if n_active == 0 and exclude_inactive_groups:
            continue
        xs.append(x)
        ys.append(y)
        sizes.append(gm.get("n_neurons", 1))
        colors.append("#7f7f7f" if n_active == 0 else "#1f77b4")

    # Scale marker area by group size (min 20, max 300)
    if sizes:
        min_s, max_s = min(sizes), max(sizes)
        if max_s > min_s:
            scaled = [20 + 280 * (s - min_s) / (max_s - min_s) for s in sizes]
        else:
            scaled = [80] * len(sizes)
    else:
        scaled = []

    ax.scatter(xs, ys, c=colors, s=scaled, edgecolors="k", linewidths=0.5, zorder=3)

    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Baseline mean pairwise distance (px)")
    ax.set_ylabel("Δ mean correlation (treatment − baseline)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_neuron_centroid_distances(
    group_metrics: List[dict],
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Per-neuron distance from group centroids",
) -> plt.Axes:
    """Scatter of each neuron's distance from baseline centroid vs treatment
    subgroup centroid, coloured by treatment status.

    Only neurons from baseline groups that produced at least one treatment
    subgroup are included.  Ungrouped neurons (active but not re-clustered)
    are plotted along the top edge with a distinct marker.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    # Collect per-neuron rows — skip groups with no treatment subgroups
    bl_dists: dict[str, list] = {"grouped": [], "ungrouped": []}
    tx_dists: dict[str, list] = {"grouped": [], "ungrouped": []}

    for gm in group_metrics:
        if gm.get("n_treatment_subgroups", 0) == 0:
            continue
        detail = gm.get("neuron_spatial_detail", [])
        for nd in detail:
            status = nd.get("treatment_status", "ungrouped")
            if status == "inactive":
                continue
            d_bl = nd.get("dist_from_baseline_centroid")
            d_tx = nd.get("dist_from_treatment_centroid")
            if d_bl is None or not np.isfinite(d_bl):
                continue
            bl_dists[status].append(d_bl)
            tx_dists[status].append(d_tx if (d_tx is not None and np.isfinite(d_tx)) else np.nan)

    # Find y-axis ceiling for ungrouped/inactive strip
    all_finite_tx = [v for vs in tx_dists.values() for v in vs if np.isfinite(v)]
    y_ceil = max(all_finite_tx) * 1.15 if all_finite_tx else 100

    handles = []
    for status in ("grouped", "ungrouped"):
        xs = bl_dists[status]
        ys = tx_dists[status]
        if not xs:
            continue
        color = _STATUS_COLORS[status]
        # Replace NaN y with y_ceil for display
        ys_plot = [y if np.isfinite(y) else y_ceil for y in ys]
        marker = "o" if status == "grouped" else "X"
        ax.scatter(xs, ys_plot, c=color, marker=marker, s=40,
                   edgecolors="k", linewidths=0.3, alpha=0.8, zorder=3,
                   label=status)
        handles.append(Patch(facecolor=color, label=status))

    # Dashed line showing where ungrouped/inactive sit
    if any(not np.isfinite(y) for ys in tx_dists.values() for y in ys):
        ax.axhline(y_ceil, color="grey", ls=":", lw=0.8, alpha=0.5)
        ax.text(ax.get_xlim()[0], y_ceil, " no tx subgroup", fontsize=7,
                va="bottom", color="grey")

    nonempty = [bl_dists[s] for s in bl_dists if bl_dists[s]]
    if nonempty:
        max_bl = max(max(vs) for vs in nonempty)
        ax.plot([0, max_bl], [0, max_bl],
                "k--", lw=0.6, alpha=0.4, label="identity")
    else:
        ax.text(0.5, 0.5, "no treatment subgroups",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="grey")

    ax.set_xlabel("Distance from baseline centroid (px)")
    ax.set_ylabel("Distance from treatment subgroup centroid (px)")
    ax.set_title(title)
    if nonempty:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_delta_corr_vs_dispersion(
    group_metrics: List[dict],
    *,
    ax: Optional[plt.Axes] = None,
    exclude_inactive_groups: bool = True,
    title: str = "Delta Correlation vs. Baseline Spatial Dispersion",
) -> plt.Axes:
    """Scatter plot of delta_mean_corr vs baseline_mean_pairwise_dist."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    xs, ys, sizes, colors = [], [], [], []
    for gm in group_metrics:
        x = gm.get("baseline_mean_pairwise_dist")
        y = gm.get("delta_mean_corr")
        n_active = gm.get("n_section_active", gm.get("n_neurons", 0))
        if x is None or y is None:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if n_active == 0 and exclude_inactive_groups:
            continue
        xs.append(x)
        ys.append(y)
        sizes.append(gm.get("n_neurons", 1))
        colors.append("#7f7f7f" if n_active == 0 else "#1f77b4")

    if sizes:
        min_s, max_s = min(sizes), max(sizes)
        if max_s > min_s:
            scaled = [20 + 280 * (s - min_s) / (max_s - min_s) for s in sizes]
        else:
            scaled = [80] * len(sizes)
    else:
        scaled = []

    ax.scatter(xs, ys, c=colors, s=scaled, edgecolors="k", linewidths=0.5, zorder=3)
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Baseline mean pairwise distance (px)")
    ax.set_ylabel("Delta mean correlation (section - baseline)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_neuron_centroid_distances(
    group_metrics: List[dict],
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Per-neuron distance from group centroids",
) -> plt.Axes:
    """Scatter of each neuron's distance from baseline centroid vs section subgroup centroid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    bl_dists: dict[str, list] = {"grouped": [], "ungrouped": []}
    tx_dists: dict[str, list] = {"grouped": [], "ungrouped": []}

    for gm in group_metrics:
        if gm.get("n_section_subgroups", 0) == 0:
            continue
        detail = gm.get("neuron_spatial_detail", [])
        for nd in detail:
            status = nd.get("section_status", "ungrouped")
            if status == "inactive":
                continue
            d_bl = nd.get("dist_from_baseline_centroid")
            d_tx = nd.get("dist_from_section_centroid")
            if d_bl is None or not np.isfinite(d_bl):
                continue
            bl_dists[status].append(d_bl)
            tx_dists[status].append(d_tx if (d_tx is not None and np.isfinite(d_tx)) else np.nan)

    all_finite_tx = [v for values in tx_dists.values() for v in values if np.isfinite(v)]
    y_ceil = max(all_finite_tx) * 1.15 if all_finite_tx else 100

    for status in ("grouped", "ungrouped"):
        xs = bl_dists[status]
        ys = tx_dists[status]
        if not xs:
            continue
        color = _STATUS_COLORS[status]
        ys_plot = [y if np.isfinite(y) else y_ceil for y in ys]
        marker = "o" if status == "grouped" else "X"
        ax.scatter(
            xs,
            ys_plot,
            c=color,
            marker=marker,
            s=40,
            edgecolors="k",
            linewidths=0.3,
            alpha=0.8,
            zorder=3,
            label=status,
        )

    if any(not np.isfinite(y) for values in tx_dists.values() for y in values):
        ax.axhline(y_ceil, color="grey", ls=":", lw=0.8, alpha=0.5)
        ax.text(ax.get_xlim()[0], y_ceil, " no section subgroup", fontsize=7, va="bottom", color="grey")

    nonempty = [bl_dists[key] for key in bl_dists if bl_dists[key]]
    if nonempty:
        max_bl = max(max(values) for values in nonempty)
        ax.plot([0, max_bl], [0, max_bl], "k--", lw=0.6, alpha=0.4, label="identity")
    else:
        ax.text(
            0.5,
            0.5,
            "no section subgroups",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="grey",
        )

    ax.set_xlabel("Distance from baseline centroid (px)")
    ax.set_ylabel("Distance from section subgroup centroid (px)")
    ax.set_title(title)
    if nonempty:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax
