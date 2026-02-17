from tkinter import Button, Frame
from typing import Any
import numpy as np
import random
import argparse
from pathlib import Path

from utils.label_utils import (
    get_label_value, get_label_source, create_label_dict,
    get_keys, compute_data_summary,
)
from classifier_pipeline.io_utils import load_roi_data
from classifier_pipeline.verbose_utils import print_keys, print_data_summary, print_session_summary
from classifier_pipeline.annotation import AnnotationSessionBase


# =============================================================================
# ROI selection
# =============================================================================
def select_rois_for_annotation(
    npy_dict: dict,
    n_samples: int,
    unlabeled_only: bool = False,
    labeled_only: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Select a random sample of ROIs for annotation."""
    roi_keys = get_keys(
        npy_dict,
        level="roi",
        unlabeled_only=unlabeled_only,
        labeled_only=labeled_only,
        verbose=verbose,
    )
    n_to_sample = min(n_samples, len(roi_keys))
    selected = random.sample(roi_keys, n_to_sample)

    if verbose:
        type_desc = "unlabeled" if unlabeled_only else "labeled" if labeled_only else "all"
        print_keys(len(roi_keys), type_desc, len(selected))

    return selected


# =============================================================================
# GUI Session
# =============================================================================

class AnnotationSession(AnnotationSessionBase):
    """ROI annotation GUI."""

    def __init__(self, npy_dict : dict, roi_keys : list[str], 
                 save_path: Path, checkpoint_interval=30, verbose=True):
        
        self.roi_data_list = []
        for roi_key in roi_keys:
            roi_data = npy_dict[roi_key]
            self.roi_data_list.append({
                'key': roi_key,
                'raw_trace': roi_data['raw_trace'],
                'smoothed_trace': roi_data['smoothed_trace'],
                'features': roi_data['features'],
                'current_label': get_label_value(roi_data['label']),
            })
        self.current_idx = 0

        super().__init__(
            npy_dict=npy_dict,
            save_path=save_path,
            checkpoint_interval=checkpoint_interval,
            n_rows=2,
            figsize=(12, 8),
            title="ROI Annotation",
            verbose=verbose,
        )
        self.stats["level"] = "roi"
        self.stats["queued"] = len(self.roi_data_list)
        self._update_display()

    def _build_controls(self):
        button_frame = Frame(self.controls_frame)
        button_frame.pack(pady=4)

        Button(button_frame, text="Active (1)", command=lambda: self._label_roi(1),
               bg='green', fg='white', font=('Arial', 12, 'bold'), width=15, height=2
               ).pack(side="left", padx=10)
        Button(button_frame, text="Inactive (0)", command=lambda: self._label_roi(0),
               bg='red', fg='white', font=('Arial', 12, 'bold'), width=15, height=2
               ).pack(side="left", padx=10)
        Button(button_frame, text="Skip (Space)", command=self._skip_roi,
               bg='gray', fg='white', font=('Arial', 12, 'bold'), width=15, height=2
               ).pack(side="left", padx=10)

        nav_frame = Frame(self.controls_frame)
        nav_frame.pack(pady=4)

        Button(nav_frame, text="Previous (Left)", command=self._prev_roi,
               bg="lightblue", fg="black", font=("Arial", 11, "bold"), width=18, height=2
               ).pack(side="left", padx=10)
        Button(nav_frame, text="Save & Quit (Q)", command=self._save_and_quit,
               bg='orange', fg='white', font=('Arial', 12, 'bold'), width=20, height=2
               ).pack(side="left", padx=10)

        self.root.bind('1', lambda e: self._label_roi(1))
        self.root.bind('0', lambda e: self._label_roi(0))
        self.root.bind('<space>', lambda e: self._skip_roi())
        self.root.bind('<Right>', lambda e: self._skip_roi())
        self.root.bind('q', lambda e: self._save_and_quit())
        self.root.bind('Q', lambda e: self._save_and_quit())
        self.root.bind('<Escape>', lambda e: self._save_and_quit())
        self.root.bind('<Left>', lambda e: self._prev_roi())

    def _update_display(self):
        roi = self.roi_data_list[self.current_idx]
        features = roi['features']
        parts = roi['key'].rsplit('_', 1)

        self.progress_var.set(f"ROI {self.current_idx + 1} / {self.stats['queued']}")

        label_text = {1: 'Good', 0: 'Bad', -1: 'Unlabeled'}.get(roi['current_label'], 'Unknown')
        self._set_status(
            f"ROI {parts[1]} from {parts[0]} | Label: {label_text} | "
            f"Session: {self.stats['updated']} updated, {self.stats['confirmed']} confirmed, {self.stats['skipped']} skipped"
        )

        ax_raw, ax_smooth = self.axes
        ax_raw.clear()
        ax_smooth.clear()

        title = f"ROI {parts[1]} from Video {parts[0]}"
        ax_raw.plot(roi['raw_trace'], color='blue', linewidth=1)
        ax_raw.set_title(f"Raw F Trace - {title}")
        ax_raw.set_xlabel("Frame #")
        ax_raw.set_ylabel("Raw F")
        ax_raw.grid(True, alpha=0.3)

        ax_smooth.plot(roi['smoothed_trace'], color='red', linewidth=1)
        ax_smooth.set_title(f"Smoothed F Trace - {title}")
        ax_smooth.set_xlabel("Frame #")
        ax_smooth.set_ylabel("Smoothed F")
        ax_smooth.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def _label_roi(self, label):
        roi = self.roi_data_list[self.current_idx]
        prev_label = roi["current_label"]
        changed = (label != prev_label)

        self.npy_dict[roi["key"]]["label"] = create_label_dict(label, source="manual")
        self.roi_data_list[self.current_idx]["current_label"] = label

        self._record_label(changed)

        if self.verbose:
            if changed:
                print(f"[{self.current_idx + 1}/{self.stats['queued']}] Updated: {roi['key']} → {'Good' if label == 1 else 'Bad'}")
            else:
                print(f"[{self.current_idx + 1}/{self.stats['queued']}] Confirmed: {roi['key']}")

        self._next_roi()

    def _skip_roi(self):
        self._record_skip()
        if self.verbose:
            print(f"[{self.current_idx + 1}/{self.stats['queued']}] Skipped: {self.roi_data_list[self.current_idx]['key']}")
        self._next_roi()

    def _prev_roi(self):
        if self.current_idx <= 0:
            return
        self.current_idx -= 1
        self._update_display()

    def _next_roi(self):
        self.current_idx += 1
        if self.current_idx >= self.stats['queued']:
            self._finish()
        else:
            self._update_display()


# =============================================================================
# Orchestration
# =============================================================================
def annotate_rois(
    data_path: Path,
    n_samples: int = 1000,
    unlabeled_only: bool = False,
    labeled_only: bool = False,
    checkpoint_interval: int = 30,
    verbose: bool = True,
) -> dict:
    """Main function to run ROI annotation."""
    npy_dict = load_roi_data(data_path, verbose=verbose)

    selected_keys = select_rois_for_annotation(
        npy_dict,
        n_samples=n_samples,
        unlabeled_only=unlabeled_only,
        labeled_only=labeled_only,
        verbose=verbose,
    )

    session = AnnotationSession(
        npy_dict=npy_dict,
        roi_keys=selected_keys,
        save_path=data_path,
        checkpoint_interval=checkpoint_interval,
        verbose=verbose,
    )
    stats = session.run()

    if verbose:
        print_session_summary(stats)
        s = compute_data_summary(npy_dict, level="roi")
        print_data_summary(s)

    return stats
# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Annotate ROI data")
    parser.add_argument("--data_path", type=str,
                        default="training_data/roi_filtering/all_roi_features.npy",
                        help="Path to ROI data file")
    parser.add_argument("--number_annotations", '-n', type=int, default=1000,
                        help="Number of annotations to perform")
    parser.add_argument("--unlabeled_only", action='store_true',
                        help="Only annotate ROIs with label=-1")
    parser.add_argument("--labeled_only", action='store_true',
                        help="Only annotate already-labeled ROIs")
    parser.add_argument("--checkpoint_interval", type=int, default=30,
                        help="Save checkpoint every N annotations")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Enable verbose output")
    args = parser.parse_args()

    annotate_rois(
        data_path=Path(args.data_path),
        n_annotations=args.number_annotations,
        unlabeled_only=args.unlabeled_only,
        labeled_only=args.labeled_only,
        checkpoint_interval=args.checkpoint_interval,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()