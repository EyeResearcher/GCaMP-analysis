"""
ROI-centric spike annotation module.

What this module does
- Loads your existing spike-training .npy dict + the companion *_spike_keys.csv
- Presents one ROI at a time
- Lets the user label ALL candidate spikes for that ROI (good/bad), while seeing:
  - the full trace (raw + smoothed)
  - all spike peak locations
  - the current spike window highlight
  - the current stored label (so you can quickly "skip" if it’s already correct)
- Supports:
  - Previous/Next spike within an ROI
  - Previous/Next ROI
  - "Label all remaining as bad" for the current ROI
  - unlabeled_only and labeled_only filtering modes
  - checkpoint saving every N labeled spikes

Data format assumptions (matches your current annotate_spikes.py):
npy_dict = {
  "<roi_key>": {
    "raw_trace": np.ndarray,
    "smoothed_trace": np.ndarray,
    "spikes": {
      <spike_idx:int>: {
        "label": int,                # -1 unlabeled, 0 bad, 1 good
        "features": dict | any,
        "windows": dict              # contains window indices (see safe access in code)
      },
      ...
    }
  },
  ...
}

CSV format:
  columns: spike_key,label
  spike_key looks like: "<roi_key>-<spike_idx>"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tkinter import Tk, Frame, Label, Button, Listbox, Scrollbar, StringVar, END, SINGLE
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# Data I/O
# =============================================================================

def load_spike_data(npy_path: Path) -> tuple[dict, dict[str, int], Path]:
    if not npy_path.exists():
        raise FileNotFoundError(f"Spike .npy not found: {npy_path}")

    npy_dict = np.load(npy_path, allow_pickle=True).item()

    csv_path = npy_path.parent / f"{npy_path.stem}_spike_keys.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Spike keys CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "spike_key" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must have columns ['spike_key','label'], got: {list(df.columns)}")

    key_labels = dict(zip(df["spike_key"].astype(str), df["label"].astype(int)))
    return npy_dict, key_labels, csv_path


def save_spike_data(npy_dict: dict, key_labels: dict[str, int], npy_path: Path) -> None:
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    # Save .npy
    np.save(npy_path, npy_dict, allow_pickle=True)

    # Save CSV
    csv_path = npy_path.parent / f"{npy_path.stem}_spike_keys.csv"
    df = pd.DataFrame(list(key_labels.items()), columns=["spike_key", "label"])
    df.to_csv(csv_path, index=False)

    print(f"Saved: {npy_path}")
    print(f"Saved: {csv_path}")


# =============================================================================
# Keys + label updates
# =============================================================================

def parse_spike_key(spike_key: str) -> tuple[str, int]:
    roi_key, spike_idx_str = spike_key.rsplit("-", 1)
    return roi_key, int(spike_idx_str)


def make_spike_key(roi_key: str, spike_idx: int) -> str:
    return f"{roi_key}-{int(spike_idx)}"


def update_spike_label(npy_dict: dict, key_labels: dict[str, int], roi_key: str, spike_idx: int, new_label: int) -> bool:
    """
    Updates BOTH:
      - npy_dict[roi_key]['spikes'][spike_idx]['label']
      - key_labels["roi_key-spike_idx"]
    Returns True if changed.
    """
    spike_idx = int(spike_idx)
    spike_key = make_spike_key(roi_key, spike_idx)

    current_label = int(npy_dict[roi_key]["spikes"][spike_idx].get("label", -1))
    changed = (int(new_label) != current_label)

    npy_dict[roi_key]["spikes"][spike_idx]["label"] = int(new_label)
    key_labels[spike_key] = int(new_label)

    return changed


def label_to_text(label: int) -> str:
    if label == 1:
        return "good"
    if label == 0:
        return "bad"
    return "unlabeled"


# =============================================================================
# ROI selection
# =============================================================================

def _spike_matches_mode(label: int, *, unlabeled_only: bool, labeled_only: bool) -> bool:
    if unlabeled_only and labeled_only:
        raise ValueError("Choose at most one of unlabeled_only or labeled_only.")

    if unlabeled_only:
        return int(label) == -1
    if labeled_only:
        return int(label) != -1
    return True


def collect_candidate_rois(
    npy_dict: dict,
    *,
    unlabeled_only: bool = False,
    labeled_only: bool = False,
) -> list[str]:
    """
    Returns ROI keys that have at least one spike matching the mode.
    """
    roi_keys: list[str] = []
    for roi_key, roi_data in npy_dict.items():
        spikes = roi_data.get("spikes", {})
        if not isinstance(spikes, dict) or len(spikes) == 0:
            continue

        keep_any = False
        for spk_idx, spk_data in spikes.items():
            try:
                lbl = int(spk_data.get("label", -1))
            except Exception:
                lbl = -1
            if _spike_matches_mode(lbl, unlabeled_only=unlabeled_only, labeled_only=labeled_only):
                keep_any = True
                break

        if keep_any:
            roi_keys.append(str(roi_key))

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
        try:
            lbl = int(spk_data.get("label", -1))
        except Exception:
            lbl = -1
        if _spike_matches_mode(lbl, unlabeled_only=unlabeled_only, labeled_only=labeled_only):
            idxs.append(int(spk_idx))

    idxs.sort()
    return idxs


# =============================================================================
# Plot helpers
# =============================================================================

def _safe_window_get(windows: Any, key: str, default: Optional[int] = None) -> Optional[int]:
    """
    windows is expected to be dict-like, but we guard because older saved data can vary.
    """
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
    """
    Full-trace plot with:
      - all spikes marked (gray dashed)
      - current spike marked (red)
      - optional shaded large window + thicker small window if indices exist
    """
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)

    ax.clear()
    ax.plot(x, y, linewidth=1)

    # Try to use your stored window indices (names match your current annotate_spikes usage)
    left_base = _safe_window_get(windows, "left_base", None)
    right_base = _safe_window_get(windows, "right_base", None)
    prev_min = _safe_window_get(windows, "prev_min", None)
    next_min = _safe_window_get(windows, "next_min", None)

    # Shade "large window" if available
    if left_base is not None and right_base is not None:
        lb = max(0, min(left_base, len(y) - 1))
        rb = max(0, min(right_base, len(y)))
        if rb > lb:
            ax.fill_between(x[lb:rb], y[lb:rb], alpha=0.15)

    # Thicken "small window" if available (prev_min -> next_min)
    if prev_min is not None and next_min is not None:
        pm = max(0, min(prev_min, len(y) - 1))
        nm = max(0, min(next_min, len(y)))
        if nm > pm:
            ax.plot(x[pm:nm], y[pm:nm], linewidth=2)

    # Mark all spike peaks
    y_lim = ax.get_ylim()
    y_bottom = y_lim[0]
    for other_idx in all_spike_indices:
        oi = int(other_idx)
        if 0 <= oi < len(y):
            ax.plot([oi, oi], [y_bottom, y[oi]], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Mark current spike peak
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
# Session state
# =============================================================================

@dataclass
class SessionConfig:
    data_path: Path
    unlabeled_only: bool = False
    labeled_only: bool = False
    checkpoint_interval: int = 30
    max_rois: Optional[int] = None


@dataclass
class SessionStats:
    rois_total: int = 0
    rois_done: int = 0
    spikes_total_in_session: int = 0
    spikes_labeled: int = 0
    spikes_updated: int = 0
    spikes_confirmed: int = 0
    spikes_skipped: int = 0


# =============================================================================
# GUI
# =============================================================================

class SpikeAnnotationByROISession:
    """
    Persistent GUI:
      - selects ROIs based on mode
      - within each ROI, iterates its candidate spikes
      - lets user relabel with full ROI context
    """

    def __init__(self, npy_dict: dict, key_labels: dict[str, int], cfg: SessionConfig):
        self.npy_dict = npy_dict
        self.key_labels = key_labels
        self.cfg = cfg

        self.roi_keys_all = collect_candidate_rois(
            self.npy_dict,
            unlabeled_only=cfg.unlabeled_only,
            labeled_only=cfg.labeled_only,
        )
        if cfg.max_rois is not None:
            self.roi_keys_all = self.roi_keys_all[: int(cfg.max_rois)]

        self.stats = SessionStats()
        self.stats.rois_total = len(self.roi_keys_all)

        # Indices into roi list + spike list
        self.roi_pos = 0
        self.spike_pos = 0

        # Current ROI spike indices (filtered by mode)
        self.current_spike_indices: list[int] = []

        # GUI
        self.root = Tk()
        self.root.title("Spike Annotation (ROI-centric)")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()
        self._bind_shortcuts()

        # Matplotlib figure
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax_raw = self.fig.add_subplot(211)
        self.ax_smooth = self.fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Load first ROI
        if self.stats.rois_total == 0:
            self._set_status_text("No ROIs match the current filter settings.")
        else:
            self._load_roi(self.roi_pos)

    # ----- layout -----

    def _build_layout(self) -> None:
        # Top info
        self.info_frame = Frame(self.root)
        self.info_frame.pack(side="top", fill="x", pady=8)

        self.progress_var = StringVar(value="")
        self.progress_label = Label(self.info_frame, textvariable=self.progress_var, font=("Arial", 12, "bold"))
        self.progress_label.pack()

        self.roi_var = StringVar(value="")
        self.roi_label = Label(self.info_frame, textvariable=self.roi_var, font=("Arial", 11))
        self.roi_label.pack()

        self.spike_var = StringVar(value="")
        self.spike_label = Label(self.info_frame, textvariable=self.spike_var, font=("Arial", 10))
        self.spike_label.pack()

        self.features_var = StringVar(value="")
        self.features_label = Label(self.info_frame, textvariable=self.features_var, font=("Arial", 9), fg="gray")
        self.features_label.pack(pady=2)

        self.status_var = StringVar(value="")
        self.status_label = Label(self.info_frame, textvariable=self.status_var, font=("Arial", 9), fg="gray")
        self.status_label.pack(pady=2)

        # Middle controls
        self.controls_frame = Frame(self.root)
        self.controls_frame.pack(side="top", fill="x", pady=6)

        # Spike labeling buttons
        btn_row1 = Frame(self.controls_frame)
        btn_row1.pack(pady=2)

        Button(btn_row1, text="Good (1)", width=18, command=lambda: self._label_current_spike(1)).pack(side="left", padx=6)
        Button(btn_row1, text="Bad (0)", width=18, command=lambda: self._label_current_spike(0)).pack(side="left", padx=6)
        Button(btn_row1, text="Skip spike", width=18, command=self._skip_current_spike).pack(side="left", padx=6)

        btn_row2 = Frame(self.controls_frame)
        btn_row2.pack(pady=2)

        Button(btn_row2, text="Prev spike", width=18, command=self._prev_spike).pack(side="left", padx=6)
        Button(btn_row2, text="Next spike", width=18, command=self._next_spike).pack(side="left", padx=6)
        Button(btn_row2, text="Label all remaining as bad", width=24, command=self._label_all_remaining_bad).pack(side="left", padx=6)

        btn_row3 = Frame(self.controls_frame)
        btn_row3.pack(pady=2)

        Button(btn_row3, text="Prev ROI", width=18, command=self._prev_roi).pack(side="left", padx=6)
        Button(btn_row3, text="Next ROI", width=18, command=self._next_roi).pack(side="left", padx=6)
        Button(btn_row3, text="Save", width=18, command=self._save).pack(side="left", padx=6)
        Button(btn_row3, text="Save & Quit", width=18, command=self._save_and_quit).pack(side="left", padx=6)

        # Spike listbox
        self.list_frame = Frame(self.root)
        self.list_frame.pack(side="left", fill="y", padx=8, pady=8)

        Label(self.list_frame, text="Spikes in ROI (filtered):", font=("Arial", 10, "bold")).pack()

        self.scrollbar = Scrollbar(self.list_frame)
        self.scrollbar.pack(side="right", fill="y")

        self.spike_listbox = Listbox(self.list_frame, width=34, height=20, selectmode=SINGLE, yscrollcommand=self.scrollbar.set)
        self.spike_listbox.pack(side="left", fill="y")
        self.scrollbar.config(command=self.spike_listbox.yview)

        self.spike_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Plot frame
        self.plot_frame = Frame(self.root)
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=8, pady=8)

    def _bind_shortcuts(self) -> None:
        # Spike labels
        self.root.bind("g", lambda _e: self._label_current_spike(1))
        self.root.bind("b", lambda _e: self._label_current_spike(0))
        self.root.bind("s", lambda _e: self._skip_current_spike())

        # Navigation
        self.root.bind("<Left>", lambda _e: self._prev_spike())
        self.root.bind("<Right>", lambda _e: self._next_spike())
        self.root.bind("<Up>", lambda _e: self._prev_roi())
        self.root.bind("<Down>", lambda _e: self._next_roi())

        # Bulk
        self.root.bind("x", lambda _e: self._label_all_remaining_bad())

        # Save
        self.root.bind("<Control-s>", lambda _e: self._save())

    # ----- state helpers -----

    def _set_status_text(self, msg: str) -> None:
        self.status_var.set(msg)

    def _current_roi_key(self) -> str:
        return self.roi_keys_all[self.roi_pos]

    def _current_spike_idx(self) -> int:
        return self.current_spike_indices[self.spike_pos]

    def _get_spike_data(self, roi_key: str, spike_idx: int) -> dict:
        return self.npy_dict[roi_key]["spikes"][int(spike_idx)]

    def _get_current_label(self, roi_key: str, spike_idx: int) -> int:
        try:
            return int(self._get_spike_data(roi_key, spike_idx).get("label", -1))
        except Exception:
            return -1

    def _checkpoint_if_needed(self) -> None:
        if self.cfg.checkpoint_interval <= 0:
            return
        if self.stats.spikes_labeled > 0 and (self.stats.spikes_labeled % self.cfg.checkpoint_interval == 0):
            self._save()
            self._set_status_text(f"Checkpoint saved ({self.stats.spikes_labeled} labeled spikes).")

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

        # If filter produced an empty ROI, skip forward until you find something or stop.
        if len(self.current_spike_indices) == 0:
            self._set_status_text(f"ROI {roi_key} has no spikes matching current filters. Skipping.")
            self._auto_advance_roi()
            return

        # Update ROI-level stats
        self.stats.rois_done = max(self.stats.rois_done, self.roi_pos)

        # Reset spike position to first spike for this ROI
        self.spike_pos = 0

        # Fill listbox
        self._refresh_spike_listbox()

        # Select first item
        self.spike_listbox.selection_clear(0, END)
        self.spike_listbox.selection_set(0)
        self.spike_listbox.activate(0)

        # Render
        self._update_display()

    def _auto_advance_roi(self) -> None:
        # Move forward until we find an ROI with spikes; else end.
        start = self.roi_pos
        for rp in range(start + 1, len(self.roi_keys_all)):
            roi_key = self.roi_keys_all[rp]
            idxs = collect_candidate_spike_indices(
                self.npy_dict, roi_key,
                unlabeled_only=self.cfg.unlabeled_only,
                labeled_only=self.cfg.labeled_only,
            )
            if len(idxs) > 0:
                self._load_roi(rp)
                return

        # none found
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

    # ----- display update -----

    def _update_display(self) -> None:
        if self.stats.rois_total == 0:
            return

        roi_key = self._current_roi_key()
        spk_idx = self._current_spike_idx()

        roi_data = self.npy_dict[roi_key]
        raw_f = np.asarray(roi_data.get("raw_trace", []), dtype=float)
        smooth_f = np.asarray(roi_data.get("smoothed_trace", []), dtype=float)

        spike_data = self._get_spike_data(roi_key, spk_idx)
        windows = spike_data.get("windows", {})
        features = spike_data.get("features", {})

        # All spikes for context (regardless of filter) so user sees everything in ROI
        all_spike_indices = sorted([int(k) for k in roi_data.get("spikes", {}).keys()])

        # Info panel
        self.progress_var.set(
            f"ROI {self.roi_pos + 1}/{self.stats.rois_total} | "
            f"Spike {self.spike_pos + 1}/{len(self.current_spike_indices)}"
        )
        self.roi_var.set(f"ROI key: {roi_key}")

        current_label = self._get_current_label(roi_key, spk_idx)
        self.spike_var.set(f"Current spike: {spk_idx} | current label: {label_to_text(current_label)}")

        # Feature summary (robust to dict / non-dict)
        if isinstance(features, dict) and len(features) > 0:
            # Prefer these if they exist, else show first few items
            preferred = ["prominence", "isolation", "distance", "width", "height"]
            parts = []
            for k in preferred:
                if k in features:
                    parts.append(f"{k}={features[k]}")
            if not parts:
                # fallback: show up to 4 items
                for i, (k, v) in enumerate(features.items()):
                    if i >= 4:
                        break
                    parts.append(f"{k}={v}")
            self.features_var.set(" | ".join(parts))
        else:
            self.features_var.set("")

        # Plots
        if raw_f.size > 0:
            _plot_trace_with_context(
                self.ax_raw,
                raw_f,
                spike_idx=spk_idx,
                all_spike_indices=all_spike_indices,
                title="Raw trace",
                y_label="F",
                windows=windows,
            )
        else:
            self.ax_raw.clear()
            self.ax_raw.set_title("Raw trace (missing)")

        if smooth_f.size > 0:
            _plot_trace_with_context(
                self.ax_smooth,
                smooth_f,
                spike_idx=spk_idx,
                all_spike_indices=all_spike_indices,
                title="Smoothed trace",
                y_label="F (smoothed)",
                windows=windows,
            )
        else:
            self.ax_smooth.clear()
            self.ax_smooth.set_title("Smoothed trace (missing)")

        self.fig.tight_layout()
        self.canvas.draw()

    # ----- actions -----

    def _label_current_spike(self, label: int) -> None:
        roi_key = self._current_roi_key()
        spk_idx = self._current_spike_idx()

        changed = update_spike_label(self.npy_dict, self.key_labels, roi_key, spk_idx, int(label))
        self.stats.spikes_labeled += 1

        if changed:
            self.stats.spikes_updated += 1
        else:
            self.stats.spikes_confirmed += 1

        # Update listbox row + display text
        self._update_listbox_row(self.spike_pos)
        self._set_status_text(f"Labeled spike {spk_idx} as {label_to_text(label)} (changed={changed}).")

        self._checkpoint_if_needed()

        # Auto-advance within ROI
        self._next_spike()

    def _skip_current_spike(self) -> None:
        self.stats.spikes_skipped += 1
        roi_key = self._current_roi_key()
        spk_idx = self._current_spike_idx()
        self._set_status_text(f"Skipped spike {spk_idx} in ROI {roi_key}.")
        self._next_spike()

    def _label_all_remaining_bad(self) -> None:
        """
        Labels current spike + all later spikes in this ROI as bad (0).
        """
        roi_key = self._current_roi_key()
        start_pos = self.spike_pos

        n_changed = 0
        n_total = 0
        for pos in range(start_pos, len(self.current_spike_indices)):
            spk_idx = self.current_spike_indices[pos]
            changed = update_spike_label(self.npy_dict, self.key_labels, roi_key, spk_idx, 0)
            n_total += 1
            self.stats.spikes_labeled += 1
            if changed:
                n_changed += 1
                self.stats.spikes_updated += 1
            else:
                self.stats.spikes_confirmed += 1

        self._refresh_spike_listbox()
        self._set_status_text(f"Labeled {n_total} remaining spikes as bad (changed={n_changed}).")

        self._checkpoint_if_needed()
        # Move to next ROI after bulk labeling
        self._next_roi()

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
            # finished ROI
            self._next_roi()
            return

        self.spike_listbox.selection_clear(0, END)
        self.spike_listbox.selection_set(self.spike_pos)
        self.spike_listbox.activate(self.spike_pos)
        self._update_display()

    def _prev_roi(self) -> None:
        if self.stats.rois_total == 0:
            return
        new_pos = max(0, self.roi_pos - 1)
        self._load_roi(new_pos)

    def _next_roi(self) -> None:
        if self.stats.rois_total == 0:
            return
        new_pos = self.roi_pos + 1
        if new_pos >= self.stats.rois_total:
            self._finish()
            return
        self._load_roi(new_pos)

    def _save(self) -> None:
        save_spike_data(self.npy_dict, self.key_labels, self.cfg.data_path)

    def _save_and_quit(self) -> None:
        self._save()
        self._finish()

    def _on_close(self) -> None:
        # Default to saving on close
        self._save_and_quit()

    def _finish(self) -> None:
        try:
            self._save()
        except Exception as e:
            print(f"Save failed during finish: {e}")

        try:
            plt.close(self.fig)
        except Exception:
            pass

        self.root.quit()
        self.root.destroy()

    # ----- public -----

    def run(self) -> SessionStats:
        self.root.mainloop()
        return self.stats


# =============================================================================
# Public entry point
# =============================================================================

def annotate_spikes_by_roi(
    data_path: Path,
    *,
    max_rois: Optional[int] = None,
    unlabeled_only: bool = False,
    labeled_only: bool = False,
    checkpoint_interval: int = 30,
) -> SessionStats:
    """
    ROI-centric spike annotation.

    Parameters
    ----------
    data_path:
        Path to your ROI/spike .npy file.
    max_rois:
        If set, only annotate the first N candidate ROIs (after filtering).
    unlabeled_only:
        Only include spikes with label == -1.
    labeled_only:
        Only include spikes with label != -1 (lets you review/spot-check labels).
    checkpoint_interval:
        Auto-save after every N labeled spikes.
    """
    npy_dict, key_labels, _csv_path = load_spike_data(data_path)

    cfg = SessionConfig(
        data_path=data_path,
        unlabeled_only=unlabeled_only,
        labeled_only=labeled_only,
        checkpoint_interval=checkpoint_interval,
        max_rois=max_rois,
    )

    session = SpikeAnnotationByROISession(npy_dict=npy_dict, key_labels=key_labels, cfg=cfg)
    stats = session.run()

    print(
        f"Done. ROIs: {stats.rois_total}. "
        f"Labeled spikes: {stats.spikes_labeled} "
        f"(updated={stats.spikes_updated}, confirmed={stats.spikes_confirmed}, skipped={stats.spikes_skipped})."
    )
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
    args = parser.parse_args()

    annotate_spikes_by_roi(
        data_path=Path(args.data_path),
        max_rois=args.max_rois,
        unlabeled_only=args.unlabeled_only,
        labeled_only=args.labeled_only,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
