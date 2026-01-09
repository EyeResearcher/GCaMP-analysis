from tkinter import Button, Tk, Label, Frame
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from pathlib import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# Label Utilities
# =============================================================================

def get_label_value(label) -> int:
    """Extract numeric label value from either dict or int format."""
    return label['value'] if isinstance(label, dict) else label


def get_label_source(label) -> str:
    """Extract label source from dict format, or 'unknown' for int format."""
    return label.get('source', 'unknown') if isinstance(label, dict) else 'unknown'


def create_label_dict(value: int, source: str = 'manual') -> dict:
    """Create a standardized label dictionary."""
    return {'value': value, 'source': source}


# =============================================================================
# Data I/O
# =============================================================================

def load_roi_data(npy_path: Path) -> dict:
    """Load ROI data from .npy file."""
    if not npy_path.exists():
        raise FileNotFoundError(f"ROI data file not found: {npy_path}")
    return np.load(npy_path, allow_pickle=True).item()


def save_roi_data(npy_dict: dict, npy_path: Path) -> None:
    """Save ROI data to .npy file."""
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, npy_dict, allow_pickle=True)
    print(f"Saved to {npy_path}")


# =============================================================================
# ROI Selection / Filtering
# =============================================================================

def filter_unlabeled_rois(npy_dict: dict) -> list[str]:
    """Return keys of ROIs with label value == -1."""
    return [k for k in npy_dict.keys() if get_label_value(npy_dict[k]['label']) == -1]


def filter_non_manual_rois(npy_dict: dict) -> list[str]:
    """Return keys of ROIs that haven't been manually verified."""
    return [k for k in npy_dict.keys() if get_label_source(npy_dict[k]['label']) != 'manual']


def select_rois_for_annotation(
    npy_dict: dict,
    n_samples: int,
    unlabeled_only: bool = False,
    manual_only: bool = False
) -> list[str]:
    """
    Select ROIs for annotation based on filtering criteria.
    
    Args:
        npy_dict: Dictionary of ROI data
        n_samples: Maximum number of ROIs to select
        unlabeled_only: Only select ROIs with label == -1
        manual_only: Only select ROIs not yet manually verified
    
    Returns:
        List of selected ROI keys
    
    Raises:
        ValueError: If no ROIs match the filtering criteria
    """
    all_keys = list(npy_dict.keys())
    
    if unlabeled_only:
        candidate_keys = filter_unlabeled_rois(npy_dict)
        filter_desc = "unlabeled"
    elif manual_only:
        candidate_keys = filter_non_manual_rois(npy_dict)
        filter_desc = "non-manual"
    else:
        candidate_keys = all_keys
        filter_desc = "all"
    
    print(f"Filtering for {filter_desc} ROIs: {len(candidate_keys)}/{len(all_keys)} available")
    
    if len(candidate_keys) == 0:
        raise ValueError(f"No {filter_desc} ROIs found!")
    
    n_to_sample = min(n_samples, len(candidate_keys))
    selected = random.sample(candidate_keys, n_to_sample)
    print(f"Selected {n_to_sample} ROIs for annotation")
    
    return selected


# =============================================================================
# Label Update Logic
# =============================================================================

def update_roi_label(npy_dict: dict, roi_key: str, new_label: int) -> bool:
    """
    Update a single ROI's label in the dictionary.
    
    Args:
        npy_dict: Dictionary of ROI data (modified in place)
        roi_key: Key of the ROI to update
        new_label: New label value (0 or 1)
    
    Returns:
        True if label value changed, False if only source updated
    """
    current_label = get_label_value(npy_dict[roi_key]['label'])
    changed = (new_label != current_label)
    
    npy_dict[roi_key]['label'] = create_label_dict(new_label, source='manual')
    
    return changed


# =============================================================================
# Annotation Session
# =============================================================================
def run_annotation_session(
    npy_dict: dict,
    roi_keys: list[str],
    save_path: Path,
    checkpoint_interval: int = 30
) -> dict:
    """
    Run an interactive annotation session for the given ROIs.
    
    Args:
        npy_dict: Dictionary of ROI data (modified in place)
        roi_keys: List of ROI keys to annotate
        save_path: Path to save checkpoints and final data
        checkpoint_interval: Save every N annotations
    
    Returns:
        Dictionary with session statistics
    """
    # Build list of ROI data for the session
    roi_data_list = []
    for roi_key in roi_keys:
        roi_data = npy_dict[roi_key]
        roi_data_list.append({
            'key': roi_key,
            'raw_trace': roi_data['raw_trace'],
            'smoothed_trace': roi_data['smoothed_trace'],
            'features': roi_data['features'],
            'current_label': get_label_value(roi_data['label'])
        })
    
    # Create and run the persistent labeler
    labeler = AnnotationSession(
        roi_data_list=roi_data_list,
        npy_dict=npy_dict,
        save_path=save_path,
        checkpoint_interval=checkpoint_interval
    )
    stats = labeler.run()
    
    return stats


def print_session_summary(npy_dict: dict, stats: dict) -> None:
    """Print summary statistics after annotation session."""
    total_labeled = sum(1 for v in npy_dict.values() if get_label_value(v['label']) != -1)
    total_unlabeled = sum(1 for v in npy_dict.values() if get_label_value(v['label']) == -1)
    manual_count = sum(1 for v in npy_dict.values() if get_label_source(v['label']) == 'manual')
    auto_count = sum(1 for v in npy_dict.values() if get_label_source(v['label']) == 'auto')
    
    print("\n" + "=" * 50)
    print("=== Annotation Complete ===")
    print(f"ROIs processed: {stats['total']}")
    print(f"  Updated:   {stats['updated']}")
    print(f"  Confirmed: {stats['confirmed']}")
    print(f"  Skipped:   {stats['skipped']}")
    print(f"\nDataset Summary:")
    print(f"  Total ROIs:    {len(npy_dict)}")
    print(f"  Labeled:       {total_labeled}")
    print(f"  Unlabeled:     {total_unlabeled}")
    print(f"  Manual labels: {manual_count}")
    print(f"  Auto labels:   {auto_count}")


# =============================================================================
# Main Entry Point
# =============================================================================

def annotate_rois(
    data_path: Path,
    n_annotations: int = 1000,
    unlabeled_only: bool = False,
    manual_only: bool = False,
    checkpoint_interval: int = 30
) -> dict:
    """
    Main function to run ROI annotation.
    
    Args:
        data_path: Path to .npy file containing ROI data
        n_annotations: Number of ROIs to annotate
        unlabeled_only: Only annotate unlabeled ROIs
        manual_only: Only annotate ROIs without manual verification
        checkpoint_interval: Save every N annotations
    
    Returns:
        Updated ROI dictionary
    """
    # Load data
    npy_dict = load_roi_data(data_path)
    print(f"Loaded {len(npy_dict)} ROIs from {data_path}")
    
    # Select ROIs
    try:
        selected_keys = select_rois_for_annotation(
            npy_dict,
            n_samples=n_annotations,
            unlabeled_only=unlabeled_only,
            manual_only=manual_only
        )
    except ValueError as e:
        print(f"❌ {e}")
        return npy_dict
    
    # Run annotation session
    stats = run_annotation_session(
        npy_dict=npy_dict,
        roi_keys=selected_keys,
        save_path=data_path,
        checkpoint_interval=checkpoint_interval
    )
    
    # Print summary
    print_session_summary(npy_dict, stats)
    
    return npy_dict


# =============================================================================
# GUI Labeler Class (Persistent Window)
# =============================================================================

class AnnotationSession:
    """Persistent annotation GUI that stays open while cycling through ROIs."""
    
    def __init__(self, roi_data_list: list[dict], npy_dict: dict, 
                 save_path: Path, checkpoint_interval: int = 30):
        """
        Initialize the annotation session.
        
        Args:
            roi_data_list: List of dicts with keys: 'key', 'raw_trace', 'smoothed_trace', 
                          'features', 'current_label'
            npy_dict: Reference to the full ROI dictionary (modified in place)
            save_path: Path to save checkpoints
            checkpoint_interval: Save every N annotations
        """
        self.roi_data_list = roi_data_list
        self.npy_dict = npy_dict
        self.save_path = save_path
        self.checkpoint_interval = checkpoint_interval
        
        self.current_idx = 0
        self.stats = {
            'total': len(roi_data_list),
            'labeled': 0,
            'updated': 0,
            'skipped': 0,
            'confirmed': 0
        }
        
        # Build the GUI
        self.root = Tk()
        self.root.title("ROI Annotation")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self._create_info_panel()
        self._create_controls()
        self._create_plot()
        self._bind_shortcuts()
        
        # Load first ROI
        self._update_display()
    
    def _create_info_panel(self):
        """Create the info panel with labels that can be updated."""
        self.info_frame = Frame(self.root)
        self.info_frame.pack(side="top", pady=10)
        
        self.progress_label = Label(self.info_frame, text="", font=('Arial', 14, 'bold'))
        self.progress_label.pack()
        
        self.roi_key_label = Label(self.info_frame, text="", font=('Arial', 12, 'bold'))
        self.roi_key_label.pack()
        
        self.current_label_label = Label(self.info_frame, text="", font=('Arial', 10))
        self.current_label_label.pack()
        
        self.deriv_skew_label = Label(self.info_frame, text="", font=('Arial', 10))
        self.deriv_skew_label.pack()
        
        self.prom_mean_label = Label(self.info_frame, text="", font=('Arial', 10))
        self.prom_mean_label.pack()
        
        self.stats_label = Label(self.info_frame, text="", font=('Arial', 9), fg='gray')
        self.stats_label.pack(pady=5)
    
    def _create_controls(self):
        """Create the control buttons."""
        controls_frame = Frame(self.root)
        controls_frame.pack(pady=10)
        
        Label(controls_frame, text="Label this ROI:", font=('Arial', 12, 'bold')).pack()
        
        button_frame = Frame(controls_frame)
        button_frame.pack()
        
        Button(button_frame, text="Active (1)", command=lambda: self._label_roi(1),
               bg='green', fg='white', font=('Arial', 12, 'bold'), width=15, height=2
               ).pack(side="left", padx=10)
        Button(button_frame, text="Inactive (0)", command=lambda: self._label_roi(0),
               bg='red', fg='white', font=('Arial', 12, 'bold'), width=15, height=2
               ).pack(side="left", padx=10)
        Button(button_frame, text="Skip (Space)", command=self._skip_roi,
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
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.root.bind('1', lambda e: self._label_roi(1))
        self.root.bind('0', lambda e: self._label_roi(0))
        self.root.bind('<space>', lambda e: self._skip_roi())
        self.root.bind('<Right>', lambda e: self._skip_roi())
        self.root.bind('q', lambda e: self._save_and_quit())
        self.root.bind('Q', lambda e: self._save_and_quit())
        self.root.bind('<Escape>', lambda e: self._save_and_quit())
    
    def _get_current_roi(self) -> dict:
        """Get the current ROI data."""
        return self.roi_data_list[self.current_idx]
    
    def _update_display(self):
        """Update the display with the current ROI."""
        roi = self._get_current_roi()
        roi_key = roi['key']
        current_label = roi['current_label']
        features = roi['features']
        
        # Parse ROI key
        parts = roi_key.rsplit('_', 1)
        video_name = parts[0]
        roi_idx = parts[1]
        
        # Update labels
        self.progress_label.config(
            text=f"ROI {self.current_idx + 1} / {self.stats['total']}"
        )
        self.roi_key_label.config(text=f"ROI {roi_idx} from Video {video_name}")
        
        label_text = {1: 'Good', 0: 'Bad', -1: 'Unlabeled'}.get(current_label, 'Unknown')
        label_color = {'Good': 'green', 'Bad': 'red', 'Unlabeled': 'gray'}.get(label_text, 'black')
        self.current_label_label.config(
            text=f"Current Label: {label_text}", 
            fg=label_color
        )
        
        self.deriv_skew_label.config(
            text=f"Derivative Skew: {features['derivative_skew']:.4f}"
        )
        self.prom_mean_label.config(
            text=f"Spike Prom Mean: {features['spike_prom_mean']:.4f}"
        )
        
        self.stats_label.config(
            text=f"Session: {self.stats['updated']} updated | {self.stats['confirmed']} confirmed | {self.stats['skipped']} skipped"
        )
        
        # Update plots
        self._update_plots(roi)
    
    def _update_plots(self, roi: dict):
        """Update the matplotlib plots."""
        self.ax1.clear()
        self.ax2.clear()
        
        parts = roi['key'].rsplit('_', 1)
        title = f"ROI {parts[1]} from Video {parts[0]}"
        
        self.ax1.plot(roi['raw_trace'], color='blue', linewidth=1)
        self.ax1.set_title(f"Raw F Trace - {title}")
        self.ax1.set_xlabel("Frame #")
        self.ax1.set_ylabel("Raw F")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.plot(roi['smoothed_trace'], color='red', linewidth=1)
        self.ax2.set_title(f"Smoothed Trace - {title}")
        self.ax2.set_xlabel("Frame #")
        self.ax2.set_ylabel("Smoothed")
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _label_roi(self, label: int):
        """Apply a label to the current ROI and move to next."""
        roi = self._get_current_roi()
        roi_key = roi['key']
        
        changed = update_roi_label(self.npy_dict, roi_key, label)
        
        if changed:
            self.stats['updated'] += 1
            label_name = 'Good' if label == 1 else 'Bad'
            print(f"[{self.current_idx + 1}/{self.stats['total']}] Updated: {roi_key} → {label_name}")
        else:
            self.stats['confirmed'] += 1
            print(f"[{self.current_idx + 1}/{self.stats['total']}] Confirmed: {roi_key}")
        
        self.stats['labeled'] += 1
        
        # Update the current_label in our local data
        self.roi_data_list[self.current_idx]['current_label'] = label
        
        self._checkpoint_if_needed()
        self._next_roi()
    
    def _skip_roi(self):
        """Skip the current ROI and move to next."""
        roi = self._get_current_roi()
        self.stats['skipped'] += 1
        print(f"[{self.current_idx + 1}/{self.stats['total']}] Skipped: {roi['key']}")
        
        self._next_roi()
    
    def _next_roi(self):
        """Move to the next ROI or finish if done."""
        self.current_idx += 1
        
        if self.current_idx >= len(self.roi_data_list):
            print("\n✅ All ROIs processed!")
            self._finish()
        else:
            self._update_display()
    
    def _checkpoint_if_needed(self):
        """Save checkpoint if interval reached."""
        if self.stats['labeled'] % self.checkpoint_interval == 0:
            save_roi_data(self.npy_dict, self.save_path)
            print(f"📦 Checkpoint: Saved progress ({self.stats['labeled']} processed)")
    
    def _save_and_quit(self):
        """Save progress and close the session."""
        print("\n⏹️ Session ended by user. Saving progress...")
        save_roi_data(self.npy_dict, self.save_path)
        self._finish()
    
    def _on_close(self):
        """Handle window close button."""
        self._save_and_quit()
    
    def _finish(self):
        """Clean up and close the window."""
        save_roi_data(self.npy_dict, self.save_path)
        plt.close(self.fig)
        self.root.quit()
        self.root.destroy()
    
    def run(self) -> dict:
        """Run the annotation session and return stats."""
        self.root.mainloop()
        return self.stats


# Legacy Labeler class for single-ROI use (kept for backwards compatibility)
class Labeler:
    def __init__(self, roi_key: str, raw_f: np.ndarray, smoothed_sp: np.ndarray, 
                 features: dict, current_label: int):
        self.roi_key = roi_key
        self.f_trace = raw_f
        self.spike_prob = smoothed_sp
        self.features = features
        self.current_label = current_label
        self.selected_label = None
        self.quit_session = False  # Flag to exit entire session

        parts = roi_key.rsplit('_', 1)
        self.video_name = parts[0]
        self.roi_idx = parts[1]
        
        self.root = Tk()
        self.title = f"ROI {self.roi_idx} from Video {self.video_name}"
        self.root.title(self.title)

        self._create_info_panel()
        self._create_controls()
        self._create_plot()
        self._bind_shortcuts()
        self.plot_traces()
    
    def _create_info_panel(self):
        info_frame = Frame(self.root)
        info_frame.pack(side="top", pady=10)
        
        Label(info_frame, text=f"ROI Key: {self.roi_key}", font=('Arial', 12, 'bold')).pack()
        label_text = {1: 'Good', 0: 'Bad', -1: 'Unlabeled'}.get(self.current_label, 'Unknown')
        Label(info_frame, text=f"Current Label: {self.current_label} ({label_text})", 
              font=('Arial', 10)).pack()
        Label(info_frame, text=f"Derivative Skew: {self.features['derivative_skew']:.4f}", 
              font=('Arial', 10)).pack()
        Label(info_frame, text=f"Spike Prom Mean: {self.features['spike_prom_mean']:.4f}", 
              font=('Arial', 10)).pack()
    
    def _create_controls(self):
        controls_frame = Frame(self.root)
        controls_frame.pack(pady=10)
        
        Label(controls_frame, text="Label this ROI:", font=('Arial', 12, 'bold')).pack()
        
        button_frame = Frame(controls_frame)
        button_frame.pack()
        
        Button(button_frame, text="Active (1)", command=lambda: self.label_roi(1),
               bg='green', fg='white', font=('Arial', 12, 'bold'), width=15, height=2
               ).pack(side="left", padx=10)
        Button(button_frame, text="Inactive (0)", command=lambda: self.label_roi(0),
               bg='red', fg='white', font=('Arial', 12, 'bold'), width=15, height=2
               ).pack(side="left", padx=10)
        Button(button_frame, text="Skip", command=self.skip_roi,
               bg='gray', fg='white', font=('Arial', 12, 'bold'), width=15, height=2
               ).pack(side="left", padx=10)
        
        # Save & Quit button
        quit_frame = Frame(controls_frame)
        quit_frame.pack(pady=10)
        
        Button(quit_frame, text="Save & Quit (Q)", command=self.save_and_quit,
               bg='orange', fg='white', font=('Arial', 12, 'bold'), width=20, height=2
               ).pack()
    
    def _create_plot(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def _bind_shortcuts(self):
        self.root.bind('1', lambda e: self.label_roi(1))
        self.root.bind('0', lambda e: self.label_roi(0))
        self.root.bind('<space>', lambda e: self.skip_roi())
        self.root.bind('<Right>', lambda e: self.skip_roi())
        self.root.bind('q', lambda e: self.save_and_quit())
        self.root.bind('Q', lambda e: self.save_and_quit())
        self.root.bind('<Escape>', lambda e: self.save_and_quit())
        
    def plot_traces(self):
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.plot(self.f_trace, color='blue', linewidth=1)
        self.ax1.set_title(f"Raw F Trace - {self.title}")
        self.ax1.set_xlabel("Frame #")
        self.ax1.set_ylabel("Raw F")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.plot(self.spike_prob, color='red', linewidth=1)
        self.ax2.set_title(f"Smoothed Trace - {self.title}")
        self.ax2.set_xlabel("Frame #")
        self.ax2.set_ylabel("Smoothed")
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def label_roi(self, label):
        self.selected_label = label
        self.root.quit()
        self.root.destroy()
    
    def skip_roi(self):
        self.selected_label = -1
        self.root.quit()
        self.root.destroy()
    
    def save_and_quit(self):
        """Exit the annotation session and save progress."""
        self.selected_label = None
        self.quit_session = True
        self.root.quit()
        self.root.destroy()
    
    def show(self):
        self.root.mainloop()
        return self.selected_label, self.quit_session



def main():
    parser = argparse.ArgumentParser(description="Annotate ROI data")
    parser.add_argument("--data_path", type=str, 
                       default="training_data/roi_filtering/all_roi_features.npy",
                       help="Path to ROI data file")
    parser.add_argument("--number_annotations", '-n', type=int, default=1000,
                       help="Number of annotations to perform")
    parser.add_argument("--unlabeled_only", action='store_true',
                       help="Only annotate ROIs with label=-1")
    parser.add_argument("--manual_only", action='store_true',
                       help="Only annotate ROIs without manual verification")
    parser.add_argument("--checkpoint_interval", type=int, default=30,
                       help="Save checkpoint every N annotations")
    args = parser.parse_args()

    annotate_rois(
        data_path=Path(args.data_path),
        n_annotations=args.number_annotations,
        unlabeled_only=args.unlabeled_only,
        manual_only=args.manual_only,
        checkpoint_interval=args.checkpoint_interval
    )


if __name__ == "__main__":
    main()