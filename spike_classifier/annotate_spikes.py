"""
Plot helpers for spike labeling GUI.

This focuses on the `LabelerApp.plot_spike` method, which highlights the
windows around a selected spike on both the raw fluorescence trace and the
spike probability trace. The class is intentionally minimal so it can be
adapted into an existing Tkinter-based GUI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Tuple
import tkinter as tk
from tkinter import Tk, Frame, Label, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save_data(npy_dict: dict, key_labels: dict, base_path: Path):
    """Save .npy and updated CSV with spike labels (skip JSON to avoid size issues)"""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
   
    npy_file = base_path.with_suffix('.npy')
    np.save(npy_file, npy_dict, allow_pickle=True)
    csv_file = base_path.parent / f"{base_path.stem}_spike_keys.csv"
    df = pd.DataFrame(list(key_labels.items()), columns=['spike_key', 'label'])
    df.to_csv(csv_file, index=False)
 
class Labeler:
    def __init__(self, roi_key: str, spike_idx: int, raw_f: np.ndarray, smoothed_sp: np.ndarray, smoothed_f: np.ndarray,
                 spike_features: dict, spike_windows: dict, current_label: int, all_spike_indices: list):
        self.roi_key = roi_key
        self.spike_idx = spike_idx
        self.raw_f = raw_f
        self.smoothed_f = smoothed_f
        self.spike_prob = smoothed_f  # Keep original with NaNs - bounds are relative to this
        self.all_spike_indices = all_spike_indices  # All spike indices in this ROI
        
        # Debug: Check trace data
        print(f"DEBUG: raw_f shape: {self.raw_f.shape}, min: {self.raw_f.min()}, max: {self.raw_f.max()}")
        print(f"DEBUG: spike_prob shape: {self.spike_prob.shape}, has NaN: {np.any(np.isnan(self.spike_prob))}")
        print(f"DEBUG: spike_prob valid range: {np.nanmin(self.spike_prob):.4f} to {np.nanmax(self.spike_prob):.4f}")
        
        # Extract window bounds from the new dictionary structure
        self.large_window_bounds = tuple(spike_windows['large_window']['bounds'])
        self.small_window_bounds = tuple(spike_windows['small_window']['bounds'])
        
        print(f"DEBUG: large_window_bounds: {self.large_window_bounds}")
        print(f"DEBUG: small_window_bounds: {self.small_window_bounds}")
        print(f"DEBUG: spike_idx: {self.spike_idx}")
        
        self.features = spike_features
        self.current_label = current_label
        self.selected_label = None  # Will be set when user clicks button
        

        parts = self.roi_key.split('_')
        self.video_name = parts[0]
        self.roi_idx = parts[1]
        
        self.root = Tk()
        self.title = f"ROI {self.roi_idx} from Video {self.video_name}"
        self.root.title(self.title)

        
        # Info panel
        info_frame = Frame(self.root)
        info_frame.pack(side ="top", pady=10)
        
        Label(info_frame, text=f"Spike {self.spike_idx} from ROI {self.roi_idx} in Video {self.video_name}", font=('Arial', 12, 'bold')).pack()
        Label(info_frame, text=f"Current Label: {self.current_label} ({'Good' if self.current_label == 1 else 'Bad/Unlabeled' if self.current_label == 0 else 'Unlabeled'})", 
              font=('Arial', 10)).pack()
        Label(info_frame, text=f"Spike Prominence: {self.features['spike_prom']:.4f}", font=('Arial', 10)).pack()
        Label(info_frame, text=f"Isolation: {self.features['isolation']:.4f}", font=('Arial', 10)).pack()
        Label(info_frame, text=f"Distance: {self.features['distance']:.4f}", font=('Arial', 10)).pack()
        Label(info_frame, text=f"Iso Skew: {self.features['iso_skew']:.4f}", font=('Arial', 10)).pack()
        Label(info_frame, text=f"Dist Skew: {self.features['dist_skew']:.4f}", font=('Arial', 10)).pack()
        
    
        
        # Controls
        controls_frame = Frame(self.root)
        controls_frame.pack(pady=10)
        
        Label(controls_frame, text="Label this ROI:", font=('Arial', 12, 'bold')).pack()
        
        button_frame = Frame(controls_frame)
        button_frame.pack()
        
        Button(button_frame, text="Good (1)", command=lambda: self.label_roi(1), 
               bg='green', fg='white', font=('Arial', 12, 'bold'), width=15, height=2).pack(side="left", padx=10)
        Button(button_frame, text="Bad (0)", command=lambda: self.label_roi(0), 
               bg='red', fg='white', font=('Arial', 12, 'bold'), width=15, height=2).pack(side="left", padx=10)
        Button(button_frame, text="Skip", command=self.skip_roi, 
               bg='gray', fg='white', font=('Arial', 12, 'bold'), width=15, height=2).pack(side="left", padx=10)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        # Keyboard shortcuts
        self.root.bind('1', lambda e: self.label_roi(1))
        self.root.bind('0', lambda e: self.label_roi(0))
        self.root.bind('<space>', lambda e: self.skip_roi())
        self.root.bind('<Right>', lambda e: self.skip_roi())
        
        self.plot_spike()
        
    def plot_spike(self):
        """
        Plot raw fluorescence and spike probability with highlighted windows.

        - Shades the large window (left/right bases from prominence calculation).
        - Re-plots the small window segment with a thicker line for emphasis.
        - Marks the spike frame with a vertical line.
        """
        if self.raw_f.size == 0 or self.spike_prob.size == 0:
            raise ValueError("raw_f and spike_prob must be populated before plotting.")

        left_base, right_base = self.large_window_bounds
        prev_min, next_min = self.small_window_bounds

        # Plot on the existing axes that are already connected to the canvas
        x_raw = np.arange(len(self.raw_f))
        _highlight_windows(
            self.ax1,  # Use existing axis
            x_raw,
            self.raw_f,
            color="C0",
            left_base=left_base,
            right_base=right_base,
            prev_min=prev_min,
            next_min=next_min,
            spike_idx=self.spike_idx,
            all_spike_indices=self.all_spike_indices,
            title="Raw Fluorescence",
            y_label="F",
        )

        # Spike probability plot on second axis
        x_prob = np.arange(len(self.spike_prob))
        _highlight_windows(
            self.ax2,  # Use existing axis
            x_prob,
            self.spike_prob,
            color="C1",
            left_base=left_base,
            right_base=right_base,
            prev_min=prev_min,
            next_min=next_min,
            spike_idx=self.spike_idx,
            all_spike_indices=self.all_spike_indices,
            title="Spike Probability",
            y_label="Prob.",
        )

        self.fig.tight_layout()
        self.canvas.draw()  # Refresh the canvas to show the plots


    def label_roi(self, label):
        self.selected_label = label
        self.root.quit()
        self.root.destroy()
    
    def skip_roi(self):
        self.selected_label = -1  # No change to label
        self.root.quit()
        self.root.destroy()
    
    def show(self):
        """Show the GUI and wait for user input"""
        self.root.mainloop()
        return self.selected_label
    
    
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
    ax.fill_between(
        x[left_base:right_base],
        y[left_base:right_base],
        color=color,
        alpha=0.2,
    )
    ax.plot(
        x[prev_min:next_min],
        y[prev_min:next_min],
        color=color,
        linewidth=3,
    )
    
    # Let matplotlib auto-scale y-axis, then get the actual limits
    ax.set_xlim(0, len(x))
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    y_lim = ax.get_ylim()
    y_bottom = y_lim[0]
    
    # Mark the current spike being annotated with a line from y-axis bottom to peak
    if spike_idx < len(y):
        ax.plot([spike_idx, spike_idx], [y_bottom, y[spike_idx]], color="red", linestyle="-", linewidth=1, label="Current Spike")

    # Mark all other spikes with thin lines from y-axis bottom to their peak values
    for other_spike_idx in all_spike_indices:
        if other_spike_idx != spike_idx and other_spike_idx < len(y):
            ax.plot([other_spike_idx, other_spike_idx], [y_bottom, y[other_spike_idx]], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(y_label)
    
    # Show the entire trace
    ax.set_xlim(0, len(x))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')


def main():
    parser = argparse.ArgumentParser(description="Annotate spike data")
    parser.add_argument("--number_annotations", '-n', type=int, help="Number of annotations to perform", default=1000)
    parser.add_argument("--unlabeled_only", action='store_true', 
                       help="Only annotate spikes with label=-1 (unlabeled)")
    args = parser.parse_args()

    base_path = Path("training_data/roi_filtering/all_roi_features")
    
    # Fix: Use parent / stem pattern instead of with_suffix
    keys_labels_path = base_path.parent / f"{base_path.stem}_spike_keys.csv"
    
    # Read CSV as DataFrame
    df = pd.read_csv(keys_labels_path)
    
    # Convert to dictionary with spike_key as keys
    key_labels = dict(zip(df['spike_key'], df['label']))
    
    # Load numpy dict
    npy_dict = np.load(base_path.with_suffix('.npy'), allow_pickle=True).item()

    all_spike_keys = list(key_labels.keys())
    
    if args.unlabeled_only:
        unlabeled_keys = [k for k in all_spike_keys if key_labels[k] == -1]
        if len(unlabeled_keys) == 0:
            print("❌ No unlabeled spikes found. Exiting.")
            return
        n_samples = min(args.number_annotations, len(unlabeled_keys))
        selected_keys = random.sample(unlabeled_keys, n_samples)
        print(f"Sampling {n_samples} unlabeled spikes")
    else:
        # Sample from all spikes
        total_available = len(all_spike_keys)
        n_samples = min(args.number_annotations, total_available)
        selected_keys = random.sample(all_spike_keys, n_samples)
        print(f"Sampling {n_samples} spikes from all {total_available} spikes")
    
    # We don't need json_dict anymore since we skip saving JSON files
    # Just use npy_dict for everything

    total_spikes = len(all_spike_keys)
    labeled_count = 0
    skipped_count = 0
    updated_count = 0

    for idx, spike_key in enumerate(selected_keys, start=1):
        roi_key = spike_key.rsplit('-', 1)[0]
        spike_idx = int(spike_key.rsplit('-', 1)[1])
        roi_data = npy_dict[roi_key]
        spike_data = roi_data['spikes'][spike_idx]
        current_label = spike_data['label']
        
        # Debug trace data
        print(f"\nDEBUG: roi_data keys: {roi_data.keys()}")
        print(f"DEBUG: raw_traces type: {type(roi_data.get('raw_traces'))}")
        print(f"DEBUG: smoothed_traces type: {type(roi_data.get('smoothed_traces'))}")
        if 'raw_traces' in roi_data:
            print(f"DEBUG: raw_traces length: {len(roi_data['raw_traces'])}")
            print(f"DEBUG: raw_traces[0] shape: {roi_data['raw_traces'][0].shape if hasattr(roi_data['raw_traces'][0], 'shape') else 'not array'}")
        if 'smoothed_traces' in roi_data:
            print(f"DEBUG: smoothed_traces length: {len(roi_data['smoothed_traces'])}")
            print(f"DEBUG: smoothed_traces[1] shape: {roi_data['smoothed_traces'][1].shape if hasattr(roi_data['smoothed_traces'][1], 'shape') else 'not array'}")
        
        print(f"\n[{idx}/{args.number_annotations}] Annotating Spike: {spike_key} | Current Label: {current_label} ({'Good' if current_label == 1 else 'Bad/Unlabeled' if current_label == 0 else 'Unlabeled'})")
        
        # Get all spike indices for this ROI
        all_spike_indices = list(roi_data['spikes'].keys())
        
        labeler = Labeler(roi_key, spike_idx, roi_data['raw_traces'][0], roi_data['smoothed_traces'][1], roi_data['smoothed_traces'][0],
                          spike_data['features'], spike_data['windows'], current_label, all_spike_indices
                        )
        selected_label = labeler.show()
        if selected_label is not None:
            if selected_label == -1:
                skipped_count += 1
                print(f"Skipped ROI: {roi_key}")
            else:
                if selected_label != current_label:
                    npy_dict[roi_key]['spikes'][spike_idx]['label'] = selected_label
                    key_labels[spike_key] = selected_label
                    updated_count += 1
                    print(f"Updated ROI: {roi_key} to Label: {selected_label} ({'Good' if selected_label == 1 else 'Bad'})")
                else:
                    print(f"No change for ROI: {roi_key}")
            labeled_count += 1
        else:
            print(f"No label selected for ROI: {roi_key}")
        if (labeled_count + updated_count) % 30 == 0:
                save_data(npy_dict, key_labels, base_path)
                print(f" Checkpoint: Saved progress ({labeled_count + updated_count} total changes)")
    print("\n" + "="*50)
    save_data(npy_dict, key_labels, base_path)
    
    print(f"\n=== Annotation Complete ===")
    print(f"ROIs sampled: {len(selected_keys)}")
    print(f"Newly labeled: {labeled_count}")
    print(f"Updated: {updated_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total labeled in dataset: {sum(1 for v in npy_dict.values() if v['label'] != -1)}/{total_available}")

    total_labeled = sum(1 for v in npy_dict.values() if v['label'] != -1)
    total_unlabeled = sum(1 for v in npy_dict.values() if v['label'] == -1)
    print(f"Total labeled in dataset: {total_labeled}/{total_spikes}")
    print(f"Total unlabeled remaining: {total_unlabeled}/{total_spikes}")

if __name__ == "__main__":
    main()