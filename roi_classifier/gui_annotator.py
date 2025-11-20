"""ROI Annotation GUI

Annotate ROIs (cells) as good (1) or bad (0) to build robust training data for the ROI classifier.

Expected Suite2p directory structure per video:
  <video_folder>/suite2p/plane0/F.npy
  <video_folder>/suite2p/plane0/cascade_spike_prob.npy   (optional; if missing we compute probabilities externally or skip probability plot)

Usage examples:
  python roi_classifier/gui_annotator.py --video_dir D:/Datasets/03-1 --labels_out training_data/roi__filtering/roi_labels.csv
  python roi_classifier/gui_annotator.py --videos_root D:/Datasets --labels_out training_data/roi__filtering/roi_labels.csv --pattern "^[0-9]+-[0-9]+$"

The labels CSV schema:
  source_file,roi_index,label
Where:
  source_file = absolute path to F.npy (so we can reload traces later)
  roi_index = integer index of ROI in F.npy
  label = 1 (good) or 0 (bad)

Features are NOT extracted here; this tool purely provides labels.
Run feature extraction afterward:
  python -c "from pathlib import Path; from roi_classifier.feature_extraction import prepare_roi_training_data; prepare_roi_training_data(Path('training_data/roi__filtering/roi_labels.csv'), Path('training_data/roi__filtering/roi_features_minmax.csv'), normalization='minmax')"

Key shortcuts:
  g -> mark good (1)
  b -> mark bad (0)
  Left/Right arrows -> navigate without labeling
  s -> save partial
  f -> finish & save

If a previous labels CSV exists, its entries are loaded; previously labeled ROIs are skipped by default (unless --include_labeled is passed to allow relabeling).
"""

import argparse
import logging
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Ensure project root is on sys.path when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def _minmax_normalize(trace: np.ndarray) -> np.ndarray:
	tmin = float(np.min(trace))
	tmax = float(np.max(trace))
	if tmax - tmin <= 1e-12:
		return np.zeros_like(trace, dtype=float)
	return (trace - tmin) / (tmax - tmin)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_suite2p_traces(plane_path: Path):
	"""Load Suite2p F.npy (required) and cascade_spike_prob.npy (optional)."""
	f_path = plane_path / 'F.npy'
	if not f_path.exists():
		raise FileNotFoundError(f"F.npy not found in {plane_path}")
	F = np.load(f_path)

	spike_prob_path = plane_path / 'cascade_spike_prob.npy'
	spike_prob = None
	if spike_prob_path.exists():
		try:
			spike_prob = np.load(spike_prob_path)
		except Exception as e:
			logger.warning(f"Failed loading cascade_spike_prob.npy: {e}")
	return F, spike_prob, f_path


class ROIAnnotatorGUI:
	def __init__(self, videos: list[Path], labels_out: Path, include_labeled: bool = False, roi_model_path: Path | None = None,
			 cascade_model_name: str | None = None, cascade_model_dir: Path | None = None,
			 save_cascade: bool = False):
		self.videos = videos
		self.labels_out = labels_out
		self.include_labeled = include_labeled
		self.roi_model = None
		self.roi_norm = 'minmax'
		self.cascade_model = None
		self.cascade_model_name = cascade_model_name
		self.cascade_model_dir = cascade_model_dir
		self.save_cascade = save_cascade
		if roi_model_path is not None and roi_model_path.exists():
			try:
				from joblib import load
				mdl = load(roi_model_path)
				# Support dict or bare estimator
				if isinstance(mdl, dict):
					self.roi_model = mdl.get('classifier') or mdl.get('pipeline') or mdl
					
				else:
					self.roi_model = mdl
				logger.info(f"Loaded ROI model from {roi_model_path}")
			except Exception as e:
				logger.warning(f"Failed to load ROI model {roi_model_path}: {e}")

		# Load CASCADE model if requested
		if self.cascade_model_name:
			try:
				from utils import load_cascade_model
				self.cascade_model = load_cascade_model(self.cascade_model_name, str(self.cascade_model_dir))
				logger.info(f"Loaded CASCADE model: {self.cascade_model_name}")
			except Exception as e:
				logger.warning(f"Could not load CASCADE model ({self.cascade_model_name}): {e}")
				self.cascade_model = None

		# Load existing labels if present
		self.existing_df = pd.read_csv(labels_out) if labels_out.exists() else pd.DataFrame(columns=['label', 'source_file', 'roi_index'])
		self.existing_set = set(zip(self.existing_df.source_file.astype(str), self.existing_df.roi_index.astype(int)))

		# Build work list of (source_file, roi_index, video_dir)
		self.work_items = []
		for v in self.videos:
			plane0 = v / 'suite2p' / 'plane0'
			if not plane0.exists():
				logger.warning(f"Skipping {v}, missing suite2p/plane0")
				continue
			try:
				F, spike_prob, f_path = load_suite2p_traces(plane0)
			except Exception as e:
				logger.error(f"Failed loading traces for {v}: {e}")
				continue
# ======= Compute CASCADE on RAW fluorescence (no normalization) if not present and model available ====== 

			if spike_prob is None and self.cascade_model is not None:
				try:
					from pipeline.preprocessing import compute_cascade_probabilities
					spike_prob = compute_cascade_probabilities(F, self.cascade_model)
					logger.info(f"Computed CASCADE probabilities for video {v.name}")
					if self.save_cascade:
						out_path = plane0 / 'cascade_spike_prob.npy'
						try:
							np.save(out_path, spike_prob)
							logger.info(f"Saved CASCADE probabilities to {out_path}")
						except Exception as se:
							logger.warning(f"Failed saving cascade_spike_prob.npy for {v.name}: {se}")
				except Exception as ce:
					logger.warning(f"CASCADE computation failed for {v.name}: {ce}")
					spike_prob = None

			# Precompute model probabilities/features for all ROIs if model available
			precomputed = None
			"""if self.roi_model is not None:
				try:
					from roi_classifier.feature_extraction import extract_roi_features
					feats_list = []
					# Prepare spike_prob handling: broadcast 1D traces if needed
					if spike_prob is None:
						spike_prob_use = None
					else:
						spike_prob_use = spike_prob
						# If prob is 1D (n_frames,), broadcast to all ROIs
						if spike_prob_use.ndim == 1 and spike_prob_use.shape[0] == F.shape[1]:
							spike_prob_use = np.tile(spike_prob_use[None, :], (F.shape[0], 1))
						# If prob is (n_frames, n_rois), transpose
						if spike_prob_use.ndim == 2 and spike_prob_use.shape[0] == F.shape[1] and spike_prob_use.shape[1] == F.shape[0]:
							spike_prob_use = spike_prob_use.T

					for r_idx in range(F.shape[0]):
						f_trace_r = F[r_idx]
						# If spike_prob is None, fallback to zeros so derivative_skew still computed
						if spike_prob_use is None:
							prob_trace_r = np.zeros_like(f_trace_r)
						else:
							prob_trace_r = spike_prob_use[r_idx] if spike_prob_use.ndim == 2 else spike_prob_use
						feats = extract_roi_features(f_trace_r, prob_trace_r, normalization=self.roi_norm)
						feats_list.append([feats['derivative_skew'], feats['spike_prom_mean']])
					X = np.array(feats_list)
					if hasattr(self.roi_model, 'predict_proba'):
						probs = self.roi_model.predict_proba(X)[:, 1]
					else:
						preds = self.roi_model.predict(X)
						probs = preds.astype(float)
					precomputed = probs
				except Exception as e:
					logger.warning(f"Failed to precompute ROI probabilities for {v.name}: {e}")
					precomputed = None"""

			for roi_idx in range(F.shape[0]):
				key = (str(f_path.resolve()), roi_idx)
				if key in self.existing_set and not self.include_labeled:
					continue
				self.work_items.append({
					'video': v.name,
					'plane_path': plane0,
					'f_path': f_path,
					'roi_index': roi_idx,
					'F': F,
					'spike_prob_all': spike_prob,
					'model_prob': None if precomputed is None else float(precomputed[roi_idx])
				})

		if not self.work_items:
			logger.info("Nothing to label (all ROIs already labeled or no videos found).")
			sys.exit(0)

		self.current_idx = 0
		self.new_labels = []  # accumulate dicts

		# GUI setup
		self.root = tk.Tk()
		self.root.title("ROI Annotator")
		self.root.geometry("1100x750")

		top_frame = tk.Frame(self.root)
		top_frame.pack(fill='x')
		self.status_label = tk.Label(top_frame, text="Starting...", font=('Arial', 12, 'bold'))
		self.status_label.pack(side='left', padx=10, pady=5)

		self.info_label = tk.Label(top_frame, text="", justify='left', font=('Consolas', 11))
		self.info_label.pack(side='right', padx=10, pady=5)

		# Matplotlib figure for current ROI traces
		self.fig, (self.ax_f, self.ax_prob) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
		self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
		self.canvas.get_tk_widget().pack(fill='both', expand=True)

		# Control buttons
		buttons = tk.Frame(self.root)
		buttons.pack(fill='x')
		tk.Button(buttons, text="Good (1)", command=lambda: self.set_label(1), width=12).pack(side='left', padx=5, pady=5)
		tk.Button(buttons, text="Bad (0)", command=lambda: self.set_label(0), width=12).pack(side='left', padx=5, pady=5)
		tk.Button(buttons, text="Prev", command=self.prev_item, width=10).pack(side='left', padx=5)
		tk.Button(buttons, text="Next", command=self.next_item, width=10).pack(side='left', padx=5)
		tk.Button(buttons, text="Save", command=self.save_partial, width=10).pack(side='left', padx=5)
		tk.Button(buttons, text="Finish", command=self.finish, width=10).pack(side='left', padx=5)

		# Key bindings
		self.root.bind('g', lambda e: self.set_label(1))
		self.root.bind('b', lambda e: self.set_label(0))
		self.root.bind('<Left>', lambda e: self.prev_item())
		self.root.bind('<Right>', lambda e: self.next_item())
		self.root.bind('s', lambda e: self.save_partial())
		self.root.bind('f', lambda e: self.finish())

		self.update_display()
		self.root.mainloop()

	def current(self):
		return self.work_items[self.current_idx]

	def update_display(self):
		item = self.current()
		total = len(self.work_items)
		self.status_label.config(text=f"Video: {item['video']} | ROI {self.current_idx+1}/{total} (global) | roi_index={item['roi_index']}")

		# Compute optional features and model probability
		model_prob_text = "Model prob: N/A"
		feat_text = ""
		if self.roi_model is not None:
			try:
				# Use precomputed prob if available for speed
				if item.get('model_prob') is not None:
					p = item['model_prob']
					model_prob_text = f"Model prob: {p:.3f}"
					from roi_classifier.feature_extraction import extract_roi_features
					f_trace = item['F'][item['roi_index']]
					sp_all = item['spike_prob_all']
					if sp_all is None:
						prob_trace = np.zeros_like(f_trace)
					else:
						# Broadcast/transpose if necessary
						if sp_all.ndim == 1 and sp_all.shape[0] == item['F'].shape[1]:
							prob_trace = sp_all
						elif sp_all.ndim == 2 and sp_all.shape[0] == item['F'].shape[1] and sp_all.shape[1] == item['F'].shape[0]:
							prob_trace = sp_all[:, item['roi_index']]
						else:
							prob_trace = sp_all[item['roi_index']]
					feats = extract_roi_features(f_trace, prob_trace, normalization=self.roi_norm)
					feat_text = f"derivative_skew: {feats['derivative_skew']:.4f} | spike_prom_mean: {feats['spike_prom_mean']:.4f}"
				else:
					from roi_classifier.feature_extraction import extract_roi_features
					f_trace = item['F'][item['roi_index']]
					sp_all = item['spike_prob_all']
					if sp_all is None:
						prob_trace = np.zeros_like(f_trace)
					else:
						if sp_all.ndim == 1 and sp_all.shape[0] == item['F'].shape[1]:
							prob_trace = sp_all
						elif sp_all.ndim == 2 and sp_all.shape[0] == item['F'].shape[1] and sp_all.shape[1] == item['F'].shape[0]:
							prob_trace = sp_all[:, item['roi_index']]
						else:
							prob_trace = sp_all[item['roi_index']]
					feats = extract_roi_features(f_trace, prob_trace, normalization=self.roi_norm)
					X = np.array([[feats['derivative_skew'], feats['spike_prom_mean']]])
					if hasattr(self.roi_model, 'predict_proba'):
						p = float(self.roi_model.predict_proba(X)[:, 1][0])
						model_prob_text = f"Model prob: {p:.3f}"
					else:
						pred = int(self.roi_model.predict(X)[0])
						model_prob_text = f"Model pred: {pred}"
					feat_text = f"derivative_skew: {feats['derivative_skew']:.4f} | spike_prom_mean: {feats['spike_prom_mean']:.4f}"
			except Exception as e:
				logger.debug(f"Feature/model compute failed: {e}")

		# Info label
		labeled = len(self.new_labels)
		extra = f" | {feat_text}" if feat_text else ""
		self.info_label.config(text=f"New labeled: {labeled} | Existing loaded: {len(self.existing_df)} | {model_prob_text}{extra}")

		# Plot traces
		self.ax_f.clear(); self.ax_prob.clear()
		# Fluorescence trace for this ROI
		f_trace = item['F'][item['roi_index']]
		self.ax_f.plot(f_trace, color='black', linewidth=0.7)
		self.ax_f.set_title('Fluorescence (raw)')
		# Spike probability if available
		if item['spike_prob_all'] is not None:
			prob_trace = item['spike_prob_all'][item['roi_index']]
			self.ax_prob.plot(prob_trace, color='blue', linewidth=0.7)
			self.ax_prob.set_title('Cascade spike probability')
		else:
			self.ax_prob.text(0.5,0.5,'Cascade missing (zeros)', ha='center', va='center')
			self.ax_prob.set_title('Spike probability (missing)')
		self.ax_prob.set_xlabel('Frame')
		self.canvas.draw()

	def set_label(self, value: int):
		item = self.current()
		entry = {
			'source_file': str(item['f_path'].resolve()),
			'roi_index': item['roi_index'],
			'label': value
		}
		self.new_labels.append(entry)
		self.next_item()

	def prev_item(self):
		if self.current_idx > 0:
			self.current_idx -= 1
			self.update_display()

	def next_item(self):
		if self.current_idx < len(self.work_items) - 1:
			self.current_idx += 1
			self.update_display()
		else:
			messagebox.showinfo("End", "Reached last ROI. Use Finish to save.")

	def save_partial(self):
		self._write_labels(partial=True)
		messagebox.showinfo("Saved", f"Partial labels written to {self.labels_out}")

	def finish(self):
		self._write_labels(partial=False)
		messagebox.showinfo("Done", f"Final labels saved to {self.labels_out}")
		self.root.quit(); self.root.destroy()

	def _write_labels(self, partial: bool):
		if not self.new_labels:
			return
		new_df = pd.DataFrame(self.new_labels)
		if self.labels_out.exists():
			base_df = pd.read_csv(self.labels_out)
			combined = pd.concat([base_df, new_df], ignore_index=True)
			# Deduplicate, keep last annotation per (source_file, roi_index)
			combined.sort_values(by=['source_file', 'roi_index']).drop_duplicates(subset=['source_file','roi_index'], keep='last', inplace=True)
			combined.to_csv(self.labels_out, index=False)
		else:
			new_df.to_csv(self.labels_out, index=False)


def collect_videos(videos_root: Path, pattern: str, recursive: bool) -> list[Path]:
	"""Collect video directories that contain a suite2p/plane0 folder.

	If recursive is True, walk the entire tree under videos_root and pick any directory
	that has the structure <video_dir>/suite2p/plane0. Otherwise only inspect the first level.

	Pattern (regex) filters on the video directory's name (the parent of suite2p).
	"""
	regex = re.compile(pattern) if pattern else None
	vids: set[Path] = set()
	if recursive:
		for plane0 in videos_root.rglob('plane0'):
			# Require parent suite2p
			if plane0.parent.name != 'suite2p':
				continue
			video_dir = plane0.parent.parent
			if not video_dir.is_dir():
				continue
			if regex and not regex.match(video_dir.name):
				continue
			vids.add(video_dir)
	else:
		# Non-recursive: just look at immediate children
		for child in videos_root.iterdir():
			if not child.is_dir():
				continue
			if regex and not regex.match(child.name):
				continue
			plane0 = child / 'suite2p' / 'plane0'
			if plane0.exists():
				vids.add(child)
	return sorted(vids)


def main():
	ap = argparse.ArgumentParser(description="ROI Annotation GUI")
	group = ap.add_mutually_exclusive_group(required=True)
	group.add_argument('--video_dir', type=Path, help='Single video directory containing suite2p/plane0')
	group.add_argument('--videos_root', type=Path, help='Root directory of multiple video folders')
	ap.add_argument('--pattern', type=str, default=r'^[0-9]+-[0-9]+$', help='Regex to select video folder names when using --videos_root')
	ap.add_argument('--recursive', action='store_true', help='Recursively search for any suite2p/plane0 folders under videos_root')
	ap.add_argument('--labels_out', type=Path, default=Path('training_data/roi__filtering/roi_labels.csv'))
	ap.add_argument('--include_labeled', action='store_true', help='Include already labeled ROIs for relabeling')
	ap.add_argument('--roi_model', type=Path, default=None, help='Optional ROI classifier to display probability for current ROI')
	ap.add_argument('--cascade_model_name', type=str, default='Global_EXC_30Hz_smoothing100ms_high_noise', help='CASCADE model name to compute spike probabilities when missing')
	ap.add_argument('--cascade_model_dir', type=Path, default=Path('Cascade/Pretrained_models'), help='CASCADE models directory')
	ap.add_argument('--save_cascade', action='store_true', help='Persist computed cascade_spike_prob.npy into each plane0 folder if missing')
	ap.add_argument('--no_cascade', action='store_true', help='Skip CASCADE computation entirely; only use existing cascade_spike_prob.npy if present')
	args = ap.parse_args()

	if args.video_dir:
		videos = [args.video_dir]
	else:
		if not args.videos_root.exists():
			logger.error(f"videos_root not found: {args.videos_root}")
			sys.exit(1)
		videos = collect_videos(args.videos_root, args.pattern, recursive=args.recursive)
		if not videos:
			logger.error("No matching video folders found.")
			sys.exit(1)

	args.labels_out.parent.mkdir(parents=True, exist_ok=True)
	logger.info(f"Starting ROI annotation for {len(videos)} video(s). Output: {args.labels_out}")
	ROIAnnotatorGUI(
		videos,
		args.labels_out,
		include_labeled=args.include_labeled,
		roi_model_path=args.roi_model,
		cascade_model_name=None if args.no_cascade else args.cascade_model_name,
		cascade_model_dir=args.cascade_model_dir,
		save_cascade=args.save_cascade,
	)


if __name__ == '__main__':
	main()

