"""
ROI-centric spike annotation module.

Presents one ROI at a time, lets the user label ALL candidate spikes
(good/bad) while seeing the full trace with spike window highlights.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt

from tkinter import Frame, Label, Button, Listbox, Scrollbar, StringVar, END, SINGLE

from utils.label_utils import (
    create_label_dict, get_label_value,
    update_spike_label, label_to_text,
    matches_label_mode as _spike_matches_mode,
    compute_data_summary, parse_spike_key, make_spike_key
)
from utils.visualization import plot_trace_with_spikes, _safe_window_get
from classifier_pipeline.io_utils import load_roi_data
from classifier_pipeline.annotation import AnnotationSessionBase
from classifier_pipeline.verbose_utils import print_session_summary, print_data_summary


# =============================================================================
# Helpers
# =============================================================================

def collect_candidate_rois(
    npy_dict: dict,
    *,
    unlabeled_only: bool = False,
    labeled_only: bool = False,
) -> list[str]:
    """Return ROI keys that have at least one spike matching the mode."""
    roi_keys: list[str] = []
    for roi_key, roi_data in npy_dict.items():
        spikes = roi_data.get("spikes", {})
        if not isinstance(spikes, dict) or len(spikes) == 0:
            continue
        for spk_data in spikes.values():
            lbl = spk_data.get("label", create_label_dict(-1, "unlabeled"))
            if _spike_matches_mode(lbl, unlabeled_only=unlabeled_only, labeled_only=labeled_only):
                roi_keys.append(str(roi_key))
                break
    return roi_keys


def collect_candidate_spike_indices(
    npy_dict: dict,
    roi_key: str,
    *,
    unlabeled_only: bool = False,
    labeled_only: bool = False,
) -> list[int]:
    spikes = npy_dict[roi_key].get("spikes", {})
    if not isinstance(spikes, dict) or len(spikes) == 0:
        return []
    idxs: list[int] = []
    for spk_idx, spk_data in spikes.items():
        lbl = spk_data.get("label", create_label_dict(-1, "unlabeled"))
        if _spike_matches_mode(lbl, unlabeled_only=unlabeled_only, labeled_only=labeled_only):
            idxs.append(int(spk_idx))
    idxs.sort()
    return idxs


# =============================================================================
# Plot helpers
# =============================================================================

def _safe_window_get(windows: Any, key: str, default: Optional[int] = None) -> Optional[int]:
    """Safely extract an integer from a possibly-dict windows object."""
    if isinstance(windows, dict):
        v = windows.get(key, default)
        try:
            return int(v) if v is not None else default
        except Exception:
            return default
    return default


def _plot_trace_with_context(
    ax: plt.Axes,
    y: np.ndarray,
    *,
    spike_idx: int,
    all_spike_indices: list[int],
    title: str,
    y_label: str,
    windows: Any,
) -> None:
    """Full-trace plot with spike markers and window shading."""
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
    y_lim = ax.get_ylim()
    y_bottom = y_lim[0]
    for other_idx in all_spike_indices:
        oi = int(other_idx)
        if 0 <= oi < len(y):
            ax.plot([oi, oi], [y_bottom, y[oi]], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Current spike peak
    si = int(spike_idx)
    if 0 <= si < len(y):
        ax.plot([si, si], [y_bottom, y[si]], color="red", linestyle="-", linewidth=2, label="Current spike")

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(y_label)
    ax.set_xlim(0, len(y))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")


# =============================================================================
# Config
# =============================================================================

@dataclass
class SessionConfig:
    data_path: Path
    unlabeled_only: bool = False
    labeled_only: bool = False
    checkpoint_interval: int = 30
    max_rois: Optional[int] = None


# =============================================================================
# GUI Session
# =============================================================================

class SpikeAnnotationByROISession(AnnotationSessionBase):
    """
    Two-level annotation GUI: ROI → spikes within ROI.
    Inherits window lifecycle, checkpoint, save, and stats from base.
    """

    def __init__(self, npy_dict: dict, cfg: SessionConfig):
        self.cfg = cfg

        self.roi_keys_all = collect_candidate_rois(
            npy_dict,
            unlabeled_only=cfg.unlabeled_only,
            labeled_only=cfg.labeled_only,
        )
        if cfg.max_rois is not None:
            self.roi_keys_all = self.roi_keys_all[: int(cfg.max_rois)]

        self.roi_pos = 0
        self.spike_pos = 0
        self.current_spike_indices: list[int] = []

        super().__init__(
            npy_dict=npy_dict,
            save_path=cfg.data_path,
            checkpoint_interval=cfg.checkpoint_interval,
            n_rows=2,
            figsize=(10, 6),
            title="Spike Annotation (ROI-centric)",
            verbose=True,
        )
        self.stats["level"] = "spike"
        self.stats["total"] = len(self.roi_keys_all)

        # --- Extra info labels (spike-specific) ---
        self.roi_var = StringVar(value="")
        Label(self.info_frame, textvariable=self.roi_var, font=("Arial", 11)).pack()

        self.spike_var = StringVar(value="")
        Label(self.info_frame, textvariable=self.spike_var, font=("Arial", 10)).pack()

        self.features_var = StringVar(value="")
        Label(self.info_frame, textvariable=self.features_var, font=("Arial", 9), fg="gray").pack(pady=2)

        # --- Spike listbox (before plot_frame, on the left) ---
        self.list_frame = Frame(self.root)
        self.list_frame.pack(side="left", fill="y", padx=8, pady=8, before=self.plot_frame)

        Label(self.list_frame, text="Spikes in ROI (filtered):", font=("Arial", 10, "bold")).pack()

        self.scrollbar = Scrollbar(self.list_frame)
        self.scrollbar.pack(side="right", fill="y")

        self.spike_listbox = Listbox(
            self.list_frame, width=34, height=20, selectmode=SINGLE,
            yscrollcommand=self.scrollbar.set,
        )
        self.spike_listbox.pack(side="left", fill="y")
        self.scrollbar.config(command=self.spike_listbox.yview)
        self.spike_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Load first ROI
        if self.stats["total"] > 0:
            self._load_roi(0)
        else:
            self._set_status("No ROIs match the current filter settings.")

    # ----- controls -----

    def _build_controls(self):
        # Spike labeling
        btn_row1 = Frame(self.controls_frame)
        btn_row1.pack(pady=2)

        Button(btn_row1, text="Good (G)", width=18,
               command=lambda: self._label_current_spike(1)).pack(side="left", padx=6)
        Button(btn_row1, text="Bad (B)", width=18,
               command=lambda: self._label_current_spike(0)).pack(side="left", padx=6)
        Button(btn_row1, text="Skip (S)", width=18,
               command=self._skip_current_spike).pack(side="left", padx=6)

        # Spike nav + bulk
        btn_row2 = Frame(self.controls_frame)
        btn_row2.pack(pady=2)

        Button(btn_row2, text="Prev spike (←)", width=18,
               command=self._prev_spike).pack(side="left", padx=6)
        Button(btn_row2, text="Next spike (→)", width=18,
               command=self._next_spike).pack(side="left", padx=6)
        Button(btn_row2, text="All remaining → Bad (X)", width=24,
               command=self._label_all_remaining_bad).pack(side="left", padx=6)

        # ROI nav + save
        btn_row3 = Frame(self.controls_frame)
        btn_row3.pack(pady=2)

        Button(btn_row3, text="Prev ROI (↑)", width=18,
               command=self._prev_roi).pack(side="left", padx=6)
        Button(btn_row3, text="Next ROI (↓)", width=18,
               command=self._next_roi).pack(side="left", padx=6)
        Button(btn_row3, text="Save (Ctrl+S)", width=18,
               command=self._save).pack(side="left", padx=6)
        Button(btn_row3, text="Save & Quit", width=18,
               command=self._save_and_quit).pack(side="left", padx=6)

        # Keyboard shortcuts
        self.root.bind("g", lambda _e: self._label_current_spike(1))
        self.root.bind("b", lambda _e: self._label_current_spike(0))
        self.root.bind("s", lambda _e: self._skip_current_spike())
        self.root.bind("<Left>", lambda _e: self._prev_spike())
        self.root.bind("<Right>", lambda _e: self._next_spike())
        self.root.bind("<Up>", lambda _e: self._prev_roi())
        self.root.bind("<Down>", lambda _e: self._next_roi())
        self.root.bind("x", lambda _e: self._label_all_remaining_bad())
        self.root.bind("<Control-s>", lambda _e: self._save())

    # ----- state helpers -----

    def _current_roi_key(self) -> str:
        return self.roi_keys_all[self.roi_pos]

    def _current_spike_idx(self) -> int:
        return self.current_spike_indices[self.spike_pos]

    def _get_spike_data(self, roi_key: str, spike_idx: int) -> dict:
        return self.npy_dict[roi_key]["spikes"][int(spike_idx)]

    def _get_current_label(self, roi_key: str, spike_idx: int) -> int:
        try:
            label = self._get_spike_data(roi_key, spike_idx).get(
                "label", create_label_dict(-1, "unlabeled")
            )
            return get_label_value(label)
        except Exception:
            return -1

    # ----- ROI loading -----

    def _load_roi(self, roi_pos: int) -> None:
        self.roi_pos = max(0, min(int(roi_pos), len(self.roi_keys_all) - 1))
        roi_key = self._current_roi_key()

        self.current_spike_indices = collect_candidate_spike_indices(
            self.npy_dict,
            roi_key,
            unlabeled_only=self.cfg.unlabeled_only,
            labeled_only=self.cfg.labeled_only,
        )

        if len(self.current_spike_indices) == 0:
            self._set_status(f"ROI {roi_key} has no spikes matching current filters. Skipping.")
            self._auto_advance_roi()
            return

        self.spike_pos = 0
        self._refresh_spike_listbox()

        self.spike_listbox.selection_clear(0, END)
        self.spike_listbox.selection_set(0)
        self.spike_listbox.activate(0)

        self._update_display()

    def _auto_advance_roi(self) -> None:
        for rp in range(self.roi_pos + 1, len(self.roi_keys_all)):
            idxs = collect_candidate_spike_indices(
                self.npy_dict, self.roi_keys_all[rp],
                unlabeled_only=self.cfg.unlabeled_only,
                labeled_only=self.cfg.labeled_only,
            )
            if len(idxs) > 0:
                self._load_roi(rp)
                return
        self._finish()

    # ----- listbox -----

    def _refresh_spike_listbox(self) -> None:
        self.spike_listbox.delete(0, END)
        roi_key = self._current_roi_key()
        for spk_idx in self.current_spike_indices:
            lbl = self._get_current_label(roi_key, spk_idx)
            self.spike_listbox.insert(END, f"spike {spk_idx:>4} | {label_to_text(lbl)}")

    def _update_listbox_row(self, spike_pos: int) -> None:
        if not (0 <= spike_pos < len(self.current_spike_indices)):
            return
        roi_key = self._current_roi_key()
        spk_idx = self.current_spike_indices[spike_pos]
        lbl = self._get_current_label(roi_key, spk_idx)
        self.spike_listbox.delete(spike_pos)
        self.spike_listbox.insert(spike_pos, f"spike {spk_idx:>4} | {label_to_text(lbl)}")

    def _on_listbox_select(self, _event: Any) -> None:
        sel = self.spike_listbox.curselection()
        if not sel:
            return
        self.spike_pos = int(sel[0])
        self._update_display()

    # ----- display -----

    def _update_display(self) -> None:
        if self.stats["total"] == 0:
            return

        roi_key = self._current_roi_key()
        spk_idx = self._current_spike_idx()

        roi_data = self.npy_dict[roi_key]
        raw_f = np.asarray(roi_data.get("raw_trace", []), dtype=float)
        smooth_f = np.asarray(roi_data.get("smoothed_trace", []), dtype=float)

        spike_data = self._get_spike_data(roi_key, spk_idx)
        windows = spike_data.get("windows", {})
        features = spike_data.get("features", {})

        all_spike_indices = sorted(int(k) for k in roi_data.get("spikes", {}).keys())

        # Info panel
        self.progress_var.set(
            f"ROI {self.roi_pos + 1}/{self.stats['total']} | "
            f"Spike {self.spike_pos + 1}/{len(self.current_spike_indices)}"
        )
        self.roi_var.set(f"ROI key: {roi_key}")

        current_label = self._get_current_label(roi_key, spk_idx)
        self.spike_var.set(
            f"Current spike: {spk_idx} | label: {label_to_text(current_label)}"
        )

        # Feature summary
        if isinstance(features, dict) and len(features) > 0:
            preferred = ["prominence", "isolation", "distance", "width", "height"]
            parts = [f"{k}={features[k]}" for k in preferred if k in features]
            if not parts:
                parts = [f"{k}={v}" for i, (k, v) in enumerate(features.items()) if i < 4]
            self.features_var.set(" | ".join(parts))
        else:
            self.features_var.set("")

        # Plots
        ax_raw, ax_smooth = self.axes

        if raw_f.size > 0:
            _plot_trace_with_context(
                ax_raw, raw_f,
                spike_idx=spk_idx, all_spike_indices=all_spike_indices,
                title="Raw trace", y_label="F", windows=windows,
            )
        else:
            ax_raw.clear()
            ax_raw.set_title("Raw trace (missing)")

        if smooth_f.size > 0:
            _plot_trace_with_context(
                ax_smooth, smooth_f,
                spike_idx=spk_idx, all_spike_indices=all_spike_indices,
                title="Smoothed trace", y_label="F (smoothed)", windows=windows,
            )
        else:
            ax_smooth.clear()
            ax_smooth.set_title("Smoothed trace (missing)")

        self.fig.tight_layout()
        self.canvas.draw()

    # ----- labeling actions -----

    def _label_current_spike(self, label: int) -> None:
        roi_key = self._current_roi_key()
        spk_idx = self._current_spike_idx()

        changed = update_spike_label(self.npy_dict, roi_key, spk_idx, int(label))
        self._record_label(changed)

        self._update_listbox_row(self.spike_pos)
        self._set_status(f"Labeled spike {spk_idx} as {label_to_text(label)} (changed={changed}).")

        self._next_spike()

    def _skip_current_spike(self) -> None:
        self._record_skip()
        roi_key = self._current_roi_key()
        spk_idx = self._current_spike_idx()
        self._set_status(f"Skipped spike {spk_idx} in ROI {roi_key}.")
        self._next_spike()

    def _label_all_remaining_bad(self) -> None:
        """Label current + all later spikes in this ROI as bad (0)."""
        roi_key = self._current_roi_key()
        n_changed = 0
        n_total = 0

        for pos in range(self.spike_pos, len(self.current_spike_indices)):
            spk_idx = self.current_spike_indices[pos]
            changed = update_spike_label(self.npy_dict, roi_key, spk_idx, 0)
            n_total += 1
            self._record_label(changed)
            if changed:
                n_changed += 1

        self._refresh_spike_listbox()
        self._set_status(f"Labeled {n_total} remaining spikes as bad (changed={n_changed}).")

        self._next_roi()

    # ----- navigation -----

    def _prev_spike(self) -> None:
        if len(self.current_spike_indices) == 0:
            return
        self.spike_pos = max(0, self.spike_pos - 1)
        self.spike_listbox.selection_clear(0, END)
        self.spike_listbox.selection_set(self.spike_pos)
        self.spike_listbox.activate(self.spike_pos)
        self._update_display()

    def _next_spike(self) -> None:
        if len(self.current_spike_indices) == 0:
            return
        self.spike_pos += 1
        if self.spike_pos >= len(self.current_spike_indices):
            self._next_roi()
            return
        self.spike_listbox.selection_clear(0, END)
        self.spike_listbox.selection_set(self.spike_pos)
        self.spike_listbox.activate(self.spike_pos)
        self._update_display()

    def _prev_roi(self) -> None:
        if self.stats["total"] == 0:
            return
        new_pos = max(0, self.roi_pos - 1)
        self._load_roi(new_pos)

    def _next_roi(self) -> None:
        if self.stats["total"] == 0:
            return
        new_pos = self.roi_pos + 1
        if new_pos >= self.stats["total"]:
            self._finish()
            return
        self._load_roi(new_pos)


# =============================================================================
# Public entry point
# =============================================================================

def annotate_spikes(
    data_path: Path,
    max_rois: Optional[int] = None,
    unlabeled_only: bool = False,
    labeled_only: bool = False,
    checkpoint_interval: int = 30,
    verbose: bool = True,
) -> dict:
    """ROI-centric spike annotation."""
    npy_dict = load_roi_data(data_path, verbose=verbose)

    cfg = SessionConfig(
        data_path=data_path,
        unlabeled_only=unlabeled_only,
        labeled_only=labeled_only,
        checkpoint_interval=checkpoint_interval,
        max_rois=max_rois,
    )

    session = SpikeAnnotationByROISession(npy_dict=npy_dict, cfg=cfg)
    stats = session.run()

    if verbose:
        print_session_summary(stats)
        s = compute_data_summary(npy_dict, level="spike")
        print_data_summary(s)

    return stats


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-centric spike annotation GUI")
    parser.add_argument("--data_path", type=str, required=True, help="Path to spike .npy file")
    parser.add_argument("--max_rois", type=int, default=None, help="Annotate at most N ROIs")
    parser.add_argument("--unlabeled_only", action="store_true", help="Only annotate label == -1 spikes")
    parser.add_argument("--labeled_only", action="store_true", help="Only show already-labeled spikes")
    parser.add_argument("--checkpoint_interval", type=int, default=30, help="Auto-save every N labeled spikes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    annotate_spikes(
        data_path=Path(args.data_path),
        max_rois=args.max_rois,
        unlabeled_only=args.unlabeled_only,
        labeled_only=args.labeled_only,
        checkpoint_interval=args.checkpoint_interval,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()