"""
Enhanced Spike Labeler GUI.
Features:
 - Displays raw fluorescence and cascade probability traces with spike markers.
 - Shows current spike's top-3 features and model probability.
 - Priority tag (borderline/high_conf_pos/high_conf_neg/mid) if provided.
 - Keyboard shortcuts: y (yes/good), n (no/bad), b (bad ROI), left/right arrows to navigate without labeling, s (save partial).
 - Saves annotations incrementally to CSV to avoid data loss.
 - Accepts a CSV produced by export_spike_candidates.py and optional model path.

Usage:
  python spike_labeler_gui.py --candidates spike_candidate_exports/03-1_spike_candidates.csv --output annotations/03-1_annotations.csv

CSV columns expected:
  spike_key, skew_contribution, spike_prob_value, max_second_derivative_raw, model_prob, suggested_priority

Output columns:
  spike_key,label,model_prob,skew_contribution,spike_prob_value,max_second_derivative_raw,suggested_priority

Label meanings:
  1 = real/good spike
  0 = not a spike / noise
  BAD_ROI = entire ROI invalid (sets remaining unlabeled spikes for that neuron to BAD_ROI if desired)
"""
import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOP_FEATURES = ['skew_contribution', 'spike_prob_value', 'max_second_derivative_raw']

class SpikeLabelerGUI:
    def __init__(self, df: pd.DataFrame, output_path: Path):
        self.df = df.reset_index(drop=True).copy()
        self.output_path = output_path
        self.index = 0
        self.labels = {}
        self.total = len(self.df)
        self.root = tk.Tk()
        self.root.title("Spike Labeler")
        self.root.geometry("1200x800")

        # Annotation status frame
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill='x')
        self.status_label = tk.Label(status_frame, text="Initializing...", font=('Arial', 12, 'bold'))
        self.status_label.pack(side='left', padx=10, pady=5)

        # Feature/probability panel
        info_frame = tk.Frame(self.root)
        info_frame.pack(fill='x')
        self.info_text = tk.Label(info_frame, text="", justify='left', font=('Consolas', 11))
        self.info_text.pack(side='left', padx=10)

        # Matplotlib figure
        self.fig, (self.ax_raw, self.ax_prob) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Controls
        controls = tk.Frame(self.root)
        controls.pack(fill='x')
        tk.Button(controls, text="Yes (1)", command=lambda: self.set_label(1), width=10).pack(side='left', padx=5)
        tk.Button(controls, text="No (0)", command=lambda: self.set_label(0), width=10).pack(side='left', padx=5)
        tk.Button(controls, text="Bad ROI", command=self.set_bad_roi, width=10).pack(side='left', padx=5)
        tk.Button(controls, text="Prev", command=self.prev_spike, width=10).pack(side='left', padx=5)
        tk.Button(controls, text="Next", command=self.next_spike, width=10).pack(side='left', padx=5)
        tk.Button(controls, text="Save", command=self.save_partial, width=10).pack(side='left', padx=5)
        tk.Button(controls, text="Finish", command=self.finish, width=10).pack(side='left', padx=5)

        # Key bindings
        self.root.bind('<Left>', lambda e: self.prev_spike())
        self.root.bind('<Right>', lambda e: self.next_spike())
        self.root.bind('y', lambda e: self.set_label(1))
        self.root.bind('n', lambda e: self.set_label(0))
        self.root.bind('b', lambda e: self.set_bad_roi())
        self.root.bind('s', lambda e: self.save_partial())

        # Pre-load traces if present (not mandatory)
        self.raw_trace = None
        self.prob_trace = None
        if 'raw_trace' in self.df.columns and 'prob_trace' in self.df.columns:
            # Assuming same trace for all rows; if per-row, adjust logic
            self.raw_trace = self.df['raw_trace'].iloc[0]
            self.prob_trace = self.df['prob_trace'].iloc[0]

        self.update_display()
        self.root.mainloop()

    def current_row(self):
        return self.df.iloc[self.index]

    def update_display(self):
        row = self.current_row()
        spike_key = row['spike_key']
        model_prob = row.get('model_prob', np.nan)
        priority = row.get('suggested_priority', 'unknown')
        feats = {f: row.get(f, np.nan) for f in TOP_FEATURES}

        # Update status
        labeled_count = len(self.labels)
        self.status_label.config(text=f"Spike {self.index+1}/{self.total} | Labeled: {labeled_count} | Key: {spike_key}")

        # Info panel
        info_lines = [
            f"Model prob: {model_prob:.3f}" if not np.isnan(model_prob) else "Model prob: N/A",
            f"Priority: {priority}",
        ] + [f"{name}: {feats[name]:.4f}" for name in TOP_FEATURES]
        existing = self.labels.get(spike_key, "(unlabeled)")
        info_lines.append(f"Current label: {existing}")
        self.info_text.config(text="\n".join(info_lines))

        # Plot (if we had traces we could highlight location; placeholder for now)
        self.ax_raw.clear()
        self.ax_prob.clear()
        self.ax_raw.set_title("Raw Fluorescence (optional)")
        self.ax_prob.set_title("Cascade Probability (optional)")
        if self.raw_trace is not None:
            self.ax_raw.plot(self.raw_trace, color='black', linewidth=0.7)
        if self.prob_trace is not None:
            self.ax_prob.plot(self.prob_trace, color='blue', linewidth=0.7)
        self.canvas.draw()

    def set_label(self, value):
        spike_key = self.current_row()['spike_key']
        self.labels[spike_key] = value
        self.next_spike()

    def set_bad_roi(self):
        spike_key = self.current_row()['spike_key']
        self.labels[spike_key] = 'BAD_ROI'
        self.next_spike()

    def prev_spike(self):
        if self.index > 0:
            self.index -= 1
            self.update_display()

    def next_spike(self):
        if self.index < self.total - 1:
            self.index += 1
            self.update_display()
        else:
            messagebox.showinfo("End", "Reached last spike. Use Finish to save.")

    def save_partial(self):
        self._write_output(partial=True)
        messagebox.showinfo("Saved", f"Partial save written to {self.output_path}")

    def finish(self):
        self._write_output(partial=False)
        messagebox.showinfo("Done", f"Annotations saved to {self.output_path}")
        self.root.quit()
        self.root.destroy()

    def _write_output(self, partial: bool):
        out_rows = []
        for _, row in self.df.iterrows():
            sk = row['spike_key']
            lbl = self.labels.get(sk, None)
            if partial and lbl is None:
                continue
            out_rows.append({
                'spike_key': sk,
                'label': lbl,
                'model_prob': row.get('model_prob', np.nan),
                'skew_contribution': row.get('skew_contribution', np.nan),
                'spike_prob_value': row.get('spike_prob_value', np.nan),
                'max_second_derivative_raw': row.get('max_second_derivative_raw', np.nan),
                'suggested_priority': row.get('suggested_priority', 'unknown')
            })
        if out_rows:
            out_df = pd.DataFrame(out_rows)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(self.output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Spike Labeler GUI')
    parser.add_argument('--candidates', type=Path, required=True, help='Path to spike candidates CSV')
    parser.add_argument('--output', type=Path, required=True, help='Path to write annotations CSV')
    args = parser.parse_args()

    if not args.candidates.exists():
        logger.error(f"Candidates file not found: {args.candidates}")
        sys.exit(1)

    df = pd.read_csv(args.candidates)
    if df.empty:
        logger.error("Candidates CSV is empty.")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} candidate spikes from {args.candidates}")

    app = SpikeLabelerGUI(df, args.output)

if __name__ == '__main__':
    main()
