
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from typing import List, TYPE_CHECKING
from typing import Optional, Sequence

if TYPE_CHECKING:
    from data_classes.neuron_group import NeuronGroup
import logging
logger = logging.getLogger(__name__)

def visualize_neuron_groups(neuron_groups: List[NeuronGroup], 
                            stat: np.ndarray, 
                            img_size: tuple = (512, 512), 
                            video_path: Path = None,
                            config_label: str = None):
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
        
        # Get average STTC from pre-computed stats
        avg_sttc = group.mean_spk_stats.get('mean_sttc', None) if hasattr(group, 'mean_spk_stats') else None
        
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
            'avg_sttc': avg_sttc
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
    
    # Create legend with avg STTC
    if group_info:
        legend_elements = []
        for info in group_info:
            # Build label with optional avg STTC
            if info['avg_sttc'] is not None and not np.isnan(info['avg_sttc']):
                label = f"{info['group_id']} (n={info['n_neurons']}, STTC={info['avg_sttc']:.2f})"
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
    
    # Save figure
    
    plt.close(fig)
    

    
    return fig

def print_tree(node, prefix: str = "", is_last: bool = True):
    connector = "└── " if is_last else "├── "
    print(prefix + connector + node.name)

    new_prefix = prefix + ("    " if is_last else "│   ")
    children = list(node.children.values())

    for i, child in enumerate(children):
        print_tree(child, new_prefix, i == len(children) - 1)



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