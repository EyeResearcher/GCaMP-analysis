"""
Spike annotation module with persistent GUI for labeling spikes.

Provides a clean entry point `annotate_spikes()` that handles all data loading,
selection, and annotation in a single call.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from tkinter import Tk, Frame, Label, Button

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# Data I/O
# =============================================================================

def load_spike_data(npy_path: Path) -> tuple[dict, dict, Path]:
    """
    Load spike data from .npy and corresponding CSV.
    
    Args:
        npy_path: Path to ROI data .npy file
    
    Returns:
        Tuple of (npy_dict, key_labels dict, csv_path)
    """
    if not npy_path.exists():
        raise FileNotFoundError(f"ROI data file not found: {npy_path}")
    
    npy_dict = np.load(npy_path, allow_pickle=True).item()
    
    csv_path = npy_path.parent / f"{npy_path.stem}_spike_keys.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Spike keys CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    key_labels = dict(zip(df['spike_key'], df['label']))
    
    return npy_dict, key_labels, csv_path


def save_spike_data(npy_dict: dict, key_labels: dict, npy_path: Path) -> None:
    """
    Save spike data to .npy and CSV files.
    
    Args:
        npy_dict: ROI dictionary with spike data
        key_labels: Dictionary mapping spike_key -> label
        npy_path: Path to save .npy file
    """
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save .npy
    np.save(npy_path, npy_dict, allow_pickle=True)
    
    # Save CSV
    csv_path = npy_path.parent / f"{npy_path.stem}_spike_keys.csv"
    df = pd.DataFrame(list(key_labels.items()), columns=['spike_key', 'label'])
    df.to_csv(csv_path, index=False)
    
    print(f"Saved to {npy_path}")


# =============================================================================
# Spike Selection / Filtering
# =============================================================================

def filter_unlabeled_spikes(key_labels: dict) -> list[str]:
    """Return spike keys with label == -1."""
    return [k for k, v in key_labels.items() if v == -1]


def filter_non_manual_spikes(key_labels: dict) -> list[str]:
    """Return spike keys that haven't been manually verified (label == -1 or auto)."""
    return [k for k, v in key_labels.items() if v == -1]


def select_spikes_for_annotation(
    key_labels: dict,
    n_samples: int,
    unlabeled_only: bool = False
) -> list[str]:
    """
    Select spikes for annotation based on filtering criteria.
    
    Args:
        key_labels: Dictionary mapping spike_key -> label
        n_samples: Maximum number of spikes to select
        unlabeled_only: Only select spikes with label == -1
    
    Returns:
        List of selected spike keys
    
    Raises:
        ValueError: If no spikes match the filtering criteria
    """
    all_keys = list(key_labels.keys())
    
    if unlabeled_only:
        candidate_keys = filter_unlabeled_spikes(key_labels)
        filter_desc = "unlabeled"
    else:
        candidate_keys = all_keys
        filter_desc = "all"
    
    print(f"Filtering for {filter_desc} spikes: {len(candidate_keys)}/{len(all_keys)} available")
    
    if len(candidate_keys) == 0:
        raise ValueError(f"No {filter_desc} spikes found!")
    
    n_to_sample = min(n_samples, len(candidate_keys))
    selected = random.sample(candidate_keys, n_to_sample)
    print(f"Selected {n_to_sample} spikes for annotation")
    
    return selected


# =============================================================================
# Label Update Logic
# =============================================================================

def update_spike_label(npy_dict: dict, key_labels: dict, spike_key: str, new_label: int) -> bool:
    """
    Update a single spike's label in both dictionaries.
    
    Args:
        npy_dict: ROI dictionary (modified in place)
        key_labels: Spike key labels dict (modified in place)
        spike_key: Key of the spike to update (format: "roi_key-spike_idx")
        new_label: New label value (0 or 1)
    
    Returns:
        True if label value changed, False otherwise
    """
    roi_key, spike_idx_str = spike_key.rsplit('-', 1)
    spike_idx = int(spike_idx_str)
    
    current_label = npy_dict[roi_key]['spikes'][spike_idx]['label']
    changed = (new_label != current_label)
    
    # Update both dictionaries
    npy_dict[roi_key]['spikes'][spike_idx]['label'] = new_label
    key_labels[spike_key] = new_label
    
    return changed


# =============================================================================
# Spike Data Extraction
# =============================================================================

def extract_spike_data(npy_dict: dict, spike_key: str) -> dict:
    """
    Extract all data needed to display a spike for annotation.
    
    Args:
        npy_dict: ROI dictionary
        spike_key: Spike key (format: "roi_key-spike_idx")
    
    Returns:
        Dictionary with spike data for the annotator
    """
    roi_key, spike_idx_str = spike_key.rsplit('-', 1)
    spike_idx = int(spike_idx_str)
    
    roi_data = npy_dict[roi_key]
    spike_data = roi_data['spikes'][spike_idx]
    
    # Get all spike indices for context
    all_spike_indices = list(roi_data['spikes'].keys())
    
    return {
        'spike_key': spike_key,
        'roi_key': roi_key,
        'spike_idx': spike_idx,
        'raw_f': roi_data['raw_trace'],
        'smoothed_f': roi_data['smoothed_trace'],
        'features': spike_data['features'],
        'windows': spike_data['windows'],
        'current_label': spike_data['label'],
        'all_spike_indices': all_spike_indices
    }


# =============================================================================
# Session Summary
# =============================================================================

def print_session_summary(key_labels: dict, stats: dict) -> None:
    """Print summary statistics after annotation session."""
    total_good = sum(1 for v in key_labels.values() if v == 1)
    total_bad = sum(1 for v in key_labels.values() if v == 0)
    total_unlabeled = sum(1 for v in key_labels.values() if v == -1)
    
    print("\n" + "=" * 50)
    print("=== Annotation Complete ===")
    print(f"Spikes processed: {stats['total']}")
    print(f"  Updated:   {stats['updated']}")
    print(f"  Confirmed: {stats['confirmed']}")
    print(f"  Skipped:   {stats['skipped']}")
    print(f"\nDataset Summary:")
    print(f"  Total spikes:  {len(key_labels)}")
    print(f"  Good:          {total_good}")
    print(f"  Bad:           {total_bad}")
    print(f"  Unlabeled:     {total_unlabeled}")


# =============================================================================
# Main Entry Point
# =============================================================================

def annotate_spikes(
    data_path: Path,
    n_annotations: int = 100,
    unlabeled_only: bool = True,
    checkpoint_interval: int = 30
) -> dict:
    """
    Main function to run spike annotation.
    
    Args:
        data_path: Path to .npy file containing ROI/spike data
        n_annotations: Number of spikes to annotate
        unlabeled_only: Only annotate unlabeled spikes
        checkpoint_interval: Save every N annotations
    
    Returns:
        Updated ROI dictionary
    """
    # Load data
    npy_dict, key_labels, csv_path = load_spike_data(data_path)
    print(f"Loaded {len(key_labels)} spikes from {data_path}")
    
    # Select spikes
    try:
        selected_keys = select_spikes_for_annotation(
            key_labels,
            n_samples=n_annotations,
            unlabeled_only=unlabeled_only
        )
    except ValueError as e:
        print(f"❌ {e}")
        return npy_dict
    
    # Build spike data list for session
    spike_data_list = [extract_spike_data(npy_dict, key) for key in selected_keys]
    
    # Run annotation session
    session = SpikeAnnotationSession(
        spike_data_list=spike_data_list,
        npy_dict=npy_dict,
        key_labels=key_labels,
        save_path=data_path,
        checkpoint_interval=checkpoint_interval
    )
    stats = session.run()
    
    # Print summary
    print_session_summary(key_labels, stats)
    
    return npy_dict


# =============================================================================
# GUI Annotation Session (Persistent Window)
# =============================================================================

class SpikeAnnotationSession:
    """Persistent annotation GUI that stays open while cycling through spikes."""
    
    def __init__(self, spike_data_list: list[dict], npy_dict: dict, key_labels: dict,
                 save_path: Path, checkpoint_interval: int = 30):
        """
        Initialize the annotation session.
        
        Args:
            spike_data_list: List of spike data dicts from extract_spike_data()
            npy_dict: Reference to the full ROI dictionary (modified in place)
            key_labels: Reference to spike key labels (modified in place)
            save_path: Path to save checkpoints
            checkpoint_interval: Save every N annotations
        """
        self.spike_data_list = spike_data_list
        self.npy_dict = npy_dict
        self.key_labels = key_labels
        self.save_path = save_path
        self.checkpoint_interval = checkpoint_interval
        
        self.current_idx = 0
        self.stats = {
            'total': len(spike_data_list),
            'labeled': 0,
            'updated': 0,
            'skipped': 0,
            'confirmed': 0
        }
        
        # Build the GUI
        self.root = Tk()
        self.root.title("Spike Annotation")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self._create_info_panel()
        self._create_controls()
        self._create_plot()
        self._bind_shortcuts()
        
        # Load first spike
        self._update_display()
    
    def _create_info_panel(self):
        """Create the info panel with labels that can be updated."""
        self.info_frame = Frame(self.root)
        self.info_frame.pack(side="top", pady=10)
        
        self.progress_label = Label(self.info_frame, text="", font=('Arial', 14, 'bold'))
        self.progress_label.pack()
        
        self.spike_key_label = Label(self.info_frame, text="", font=('Arial', 12, 'bold'))
        self.spike_key_label.pack()
        
        self.current_label_label = Label(self.info_frame, text="", font=('Arial', 10))
        self.current_label_label.pack()
        
        # Feature labels
        self.feature_frame = Frame(self.info_frame)
        self.feature_frame.pack(pady=5)
        
        self.prom_label = Label(self.feature_frame, text="", font=('Arial', 9))
        self.prom_label.pack(side="left", padx=10)
        
        self.isolation_label = Label(self.feature_frame, text="", font=('Arial', 9))
        self.isolation_label.pack(side="left", padx=10)
        
        self.distance_label = Label(self.feature_frame, text="", font=('Arial', 9))
        self.distance_label.pack(side="left", padx=10)
        
        self.stats_label = Label(self.info_frame, text="", font=('Arial', 9), fg='gray')
        self.stats_label.pack(pady=5)
    
    def _create_controls(self):
        """Create the control buttons."""
        controls_frame = Frame(self.root)
        controls_frame.pack(pady=10)
        
        Label(controls_frame, text="Label this spike:", font=('Arial', 12, 'bold')).pack()
        
        button_frame = Frame(controls_frame)
        button_frame.pack()
        
        Button(button_frame, text="Good (1)", command=lambda: self._label_spike(1),
               bg='green', fg='white', font=('Arial', 12, 'bold'), width=15, height=2
               ).pack(side="left", padx=10)
        Button(button_frame, text="Bad (0)", command=lambda: self._label_spike(0),
               bg='red', fg='white', font=('Arial', 12, 'bold'), width=15, height=2
               ).pack(side="left", padx=10)
        Button(button_frame, text="Skip (Space)", command=self._skip_spike,
               bg='gray', fg='white', font=('Arial', 12, 'bold'), width=15, height=2
               ).pack(side="left", padx=10)
        
        # Save & Quit button
        quit_frame = Frame(controls_frame)
        quit_frame.pack(pady=10)
        
        Button(quit_frame, text="Save & Quit (Q)", command=self._save_and_quit,
               bg='orange', fg='white', font=('Arial', 12, 'bold'), width=20, height=2
               ).pack()
    
    def _create_plot(self):
        """Create the matplotlib figure."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 9))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.root.bind('1', lambda e: self._label_spike(1))
        self.root.bind('0', lambda e: self._label_spike(0))
        self.root.bind('<space>', lambda e: self._skip_spike())
        self.root.bind('<Right>', lambda e: self._skip_spike())
        self.root.bind('q', lambda e: self._save_and_quit())
        self.root.bind('Q', lambda e: self._save_and_quit())
        self.root.bind('<Escape>', lambda e: self._save_and_quit())
    
    def _get_current_spike(self) -> dict:
        """Get the current spike data."""
        return self.spike_data_list[self.current_idx]
    
    def _update_display(self):
        """Update the display with the current spike."""
        spike = self._get_current_spike()
        
        # Parse keys
        roi_key = spike['roi_key']
        spike_idx = spike['spike_idx']
        parts = roi_key.rsplit('_', 1)
        video_name = parts[0] if len(parts) > 1 else roi_key
        roi_idx = parts[1] if len(parts) > 1 else "0"
        
        current_label = spike['current_label']
        features = spike['features']
        
        # Update labels
        self.progress_label.config(
            text=f"Spike {self.current_idx + 1} / {self.stats['total']}"
        )
        self.spike_key_label.config(
            text=f"Spike {spike_idx} from ROI {roi_idx} in Video {video_name}"
        )
        
        label_text = {1: 'Good', 0: 'Bad', -1: 'Unlabeled'}.get(current_label, 'Unknown')
        label_color = {'Good': 'green', 'Bad': 'red', 'Unlabeled': 'gray'}.get(label_text, 'black')
        self.current_label_label.config(
            text=f"Current Label: {label_text}",
            fg=label_color
        )
        
        self.prom_label.config(text=f"Prominence: {features.get('spike_prom', 0):.4f}")
        self.isolation_label.config(text=f"Isolation: {features.get('isolation', 0):.4f}")
        self.distance_label.config(text=f"Distance: {features.get('distance', 0):.4f}")
        
        self.stats_label.config(
            text=f"Session: {self.stats['updated']} updated | {self.stats['confirmed']} confirmed | {self.stats['skipped']} skipped"
        )
        
        # Update plots
        self._update_plots(spike)
    
    def _update_plots(self, spike: dict):
        """Update the matplotlib plots."""
        self.ax1.clear()
        self.ax2.clear()
        
        windows = spike['windows']
        left_base, right_base = windows['large_window']['bounds']
        prev_min, next_min = windows['small_window']['bounds']
        
        # Raw fluorescence plot
        _highlight_windows(
            self.ax1,
            np.arange(len(spike['raw_f'])),
            spike['raw_f'],
            color="C0",
            left_base=left_base,
            right_base=right_base,
            prev_min=prev_min,
            next_min=next_min,
            spike_idx=spike['spike_idx'],
            all_spike_indices=spike['all_spike_indices'],
            title="Raw Fluorescence",
            y_label="F",
        )
        
        # Spike probability plot
        _highlight_windows(
            self.ax2,
            np.arange(len(spike['smoothed_f'])),
            spike['smoothed_f'],
            color="C1",
            left_base=left_base,
            right_base=right_base,
            prev_min=prev_min,
            next_min=next_min,
            spike_idx=spike['spike_idx'],
            all_spike_indices=spike['all_spike_indices'],
            title="Smoothed Fluorescence",
            y_label="Smoothed F",
        )
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _label_spike(self, label: int):
        """Apply a label to the current spike and move to next."""
        spike = self._get_current_spike()
        spike_key = spike['spike_key']
        
        changed = update_spike_label(self.npy_dict, self.key_labels, spike_key, label)
        
        if changed:
            self.stats['updated'] += 1
            label_name = 'Good' if label == 1 else 'Bad'
            print(f"[{self.current_idx + 1}/{self.stats['total']}] Updated: {spike_key} → {label_name}")
        else:
            self.stats['confirmed'] += 1
            print(f"[{self.current_idx + 1}/{self.stats['total']}] Confirmed: {spike_key}")
        
        self.stats['labeled'] += 1
        
        # Update the current_label in our local data
        self.spike_data_list[self.current_idx]['current_label'] = label
        
        self._checkpoint_if_needed()
        self._next_spike()
    
    def _skip_spike(self):
        """Skip the current spike and move to next."""
        spike = self._get_current_spike()
        self.stats['skipped'] += 1
        print(f"[{self.current_idx + 1}/{self.stats['total']}] Skipped: {spike['spike_key']}")
        
        self._next_spike()
    
    def _next_spike(self):
        """Move to the next spike or finish if done."""
        self.current_idx += 1
        
        if self.current_idx >= len(self.spike_data_list):
            print("\n✅ All spikes processed!")
            self._finish()
        else:
            self._update_display()
    
    def _checkpoint_if_needed(self):
        """Save checkpoint if interval reached."""
        if self.stats['labeled'] % self.checkpoint_interval == 0:
            save_spike_data(self.npy_dict, self.key_labels, self.save_path)
            print(f"📦 Checkpoint: Saved progress ({self.stats['labeled']} processed)")
    
    def _save_and_quit(self):
        """Save progress and close the session."""
        print("\n⏹️ Session ended by user. Saving progress...")
        save_spike_data(self.npy_dict, self.key_labels, self.save_path)
        self._finish()
    
    def _on_close(self):
        """Handle window close button."""
        self._save_and_quit()
    
    def _finish(self):
        """Clean up and close the window."""
        save_spike_data(self.npy_dict, self.key_labels, self.save_path)
        plt.close(self.fig)
        self.root.quit()
        self.root.destroy()
    
    def run(self) -> dict:
        """Run the annotation session and return stats."""
        self.root.mainloop()
        return self.stats


# =============================================================================
# Plot Helper
# =============================================================================

def _highlight_windows(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    left_base: int,
    right_base: int,
    prev_min: int,
    next_min: int,
    spike_idx: int,
    all_spike_indices: list,
    title: str,
    y_label: str,
) -> None:
    """Plot a trace with shaded large window and thick small window, showing all spikes."""
    ax.plot(x, y, color=color, linewidth=1)
    
    # Shade large window
    ax.fill_between(
        x[left_base:right_base],
        y[left_base:right_base],
        color=color,
        alpha=0.2,
    )
    
    # Highlight small window with thicker line
    ax.plot(
        x[prev_min:next_min],
        y[prev_min:next_min],
        color=color,
        linewidth=3,
    )
    
    # Auto-scale y-axis
    ax.set_xlim(0, len(x))
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    y_lim = ax.get_ylim()
    y_bottom = y_lim[0]
    
    # Mark the current spike with a red line
    if spike_idx < len(y):
        ax.plot([spike_idx, spike_idx], [y_bottom, y[spike_idx]], 
                color="red", linestyle="-", linewidth=2, label="Current Spike")
    
    # Mark all other spikes with thin gray lines
    for other_idx in all_spike_indices:
        if other_idx != spike_idx and other_idx < len(y):
            ax.plot([other_idx, other_idx], [y_bottom, y[other_idx]], 
                    color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(y_label)
    ax.set_xlim(0, len(x))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Annotate spike data")
    parser.add_argument("--data_path", type=str,
                        default="training_data/roi_filtering/all_roi_features.npy",
                        help="Path to ROI data file")
    parser.add_argument("--number_annotations", '-n', type=int, default=100,
                        help="Number of spikes to annotate")
    parser.add_argument("--unlabeled_only", action='store_true',
                        help="Only annotate spikes with label=-1")
    parser.add_argument("--checkpoint_interval", type=int, default=30,
                        help="Save checkpoint every N annotations")
    args = parser.parse_args()

    annotate_spikes(
        data_path=Path(args.data_path),
        n_annotations=args.number_annotations,
        unlabeled_only=args.unlabeled_only,
        checkpoint_interval=args.checkpoint_interval
    )


if __name__ == "__main__":
    main()