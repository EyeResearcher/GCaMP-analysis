"""Base class for annotation GUI sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from tkinter import Tk, Frame, Label, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from classifier_pipeline.io_utils import save_roi_data


class AnnotationSessionBase:
    """
    Shared skeleton for ROI and spike annotation GUIs.

    Subclasses must implement:
        _build_controls()     — buttons and keybindings
        _update_display()     — refresh info panel + plots
        _on_session_complete() — optional hook when all items are done
    """

    def __init__(
        self,
        npy_dict: dict,
        save_path: Path,
        *,
        checkpoint_interval: int = 30,
        n_rows: int = 2,
        figsize: tuple = (12, 8),
        title: str = "Annotation",
        verbose: bool = True,
    ):
        self.npy_dict = npy_dict
        self.save_path = save_path
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose

        self.stats = {
            "total": 0,
            "labeled": 0,
            "updated": 0,
            "confirmed": 0,
            "skipped": 0,
        }

        # --- Tk root ---
        self.root = Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Info panel (subclasses add to this) ---
        self.info_frame = Frame(self.root)
        self.info_frame.pack(side="top", fill="x", pady=8)

        self.progress_var = StringVar(value="")
        self.progress_label = Label(
            self.info_frame, textvariable=self.progress_var, font=("Arial", 12, "bold")
        )
        self.progress_label.pack()

        self.status_var = StringVar(value="")
        self.status_label = Label(
            self.info_frame, textvariable=self.status_var, font=("Arial", 9), fg="gray"
        )
        self.status_label.pack(pady=2)

        # --- Controls (subclass fills this) ---
        self.controls_frame = Frame(self.root)
        self.controls_frame.pack(side="top", fill="x", pady=6)
        self._build_controls()

        # --- Matplotlib canvas ---
        self.plot_frame = Frame(self.root)
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        self.fig, self.axes = plt.subplots(n_rows, 1, figsize=figsize)
        if n_rows == 1:
            self.axes = [self.axes]
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        """Create buttons and bind keyboard shortcuts. Override in subclass."""
        raise NotImplementedError

    def _update_display(self) -> None:
        """Refresh info panel and plots for current item. Override in subclass."""
        raise NotImplementedError

    def _on_session_complete(self) -> None:
        """Called when all items have been visited. Override if needed."""
        pass

    # ------------------------------------------------------------------
    # Shared logic
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def _record_label(self, changed: bool) -> None:
        """Update stats after a label action and checkpoint if needed."""
        self.stats["labeled"] += 1
        if changed:
            self.stats["updated"] += 1
        else:
            self.stats["confirmed"] += 1
        self._checkpoint_if_needed()

    def _record_skip(self) -> None:
        self.stats["skipped"] += 1

    def _checkpoint_if_needed(self) -> None:
        if self.checkpoint_interval <= 0:
            return
        if self.stats["labeled"] > 0 and (self.stats["labeled"] % self.checkpoint_interval == 0):
            self._save()
            if self.verbose:
                self._set_status(f"Checkpoint saved ({self.stats['labeled']} labeled).")

    def _save(self) -> None:
        save_roi_data(self.npy_dict, self.save_path, verbose=self.verbose)

    def _save_and_quit(self) -> None:
        if self.verbose:
            print("Session ended by user. Saving progress...")
        self._save()
        self._finish()

    def _on_close(self) -> None:
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

    def run(self) -> dict:
        """Run the GUI event loop and return session stats."""
        self.root.mainloop()
        return self.stats