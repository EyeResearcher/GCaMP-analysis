r"""
Export spike candidates with top 3 features and model probabilities for annotation.

Optionally gate spikes by ROI classifier predictions so that only spikes from GOOD ROIs
are included for annotation. This prevents spending time on clearly bad ROIs.

Usage:
    python export_spike_candidates.py --datasets_root C:\Users\mzinn1\Desktop\Datasets --output spike_candidate_exports \
        --model spike_classifier/models/spike_classifier.pkl \
        --cascade_model_name Global_EXC_30Hz_smoothing100ms_high_noise \
        --cascade_model_dir Cascade/Pretrained_models \
        --roi_model roi_classifier/models/roi_classifier.pkl  # optional gating

Directory expectation:
  datasets_root/
      03-1/ (video folder)
          suite2p/plane0/F.npy, iscell.npy, ...
      03-2/
      ...

We:
  1. Iterate video folders (pattern: */suite2p/plane0)
  2. Load Suite2p fluorescence F.npy
  3. MinMax normalize F and compute cascade probabilities
  4. Detect spikes per neuron (ROI)
  5. Extract top-3 spike features (skew_contribution, spike_prob_value, max_second_derivative_raw)
  6. Per-video minmax scale features (to match training regime if model expects it)
  7. Compute model probability (LogisticRegression predict_proba) for class 1 (good spike)
  8. Write CSV: <output>/<video_id>_spike_candidates.csv
    Columns: spike_key, roi_index, skew_contribution, spike_prob_value, max_second_derivative_raw, model_prob, suggested_priority

Priority tiers:
  - borderline (0.35 <= prob <= 0.65)
  - high_conf_pos (prob > 0.9)
  - high_conf_neg (prob < 0.1)
  - mid (else)

"""
from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd
from joblib import load
import re
import sys

# Ensure project root is on path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.preprocessing import load_suite2p_data, compute_cascade_probabilities
from pipeline.spike_detection import detect_spikes_from_cascade
from pipeline.spike_filtering import extract_spike_features, _minmax_scale_array  # internal scaling helper
from roi_classifier.feature_extraction import extract_roi_features  # for ROI gating

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOP_FEATURES = ['skew_contribution', 'spike_prob_value', 'max_second_derivative_raw']

VIDEO_PATTERN = re.compile(r'^(?P<vid>[0-9]+-[0-9]+)$')


def minmax_normalize(trace: np.ndarray) -> np.ndarray:
    tmin = trace.min()
    tmax = trace.max()
    return (trace - tmin) / (tmax - tmin + 1e-10) if tmax > tmin else trace


def process_video(video_dir: Path, cascade_model, spike_model, roi_model, roi_threshold: float) -> pd.DataFrame:
    suite2p_path = video_dir / 'suite2p' / 'plane0'
    if not suite2p_path.exists():
        logger.warning(f"Skipping {video_dir}, suite2p/plane0 not found")
        return pd.DataFrame()

    data = load_suite2p_data(suite2p_path)
    F = data['F']  # shape (n_rois, n_frames)
    fs = data.get('fs', 30.0)

    # Per-video MinMax normalization of fluorescence traces
    F_norm = np.zeros_like(F, dtype=float)
    for i in range(F.shape[0]):
        F_norm[i] = minmax_normalize(F[i])

    # Compute cascade probabilities on normalized traces
    cascade_prob = compute_cascade_probabilities(F_norm, cascade_model)

    rows = []
    # Spike model details
    classifier = None
    expects_scale = False
    if spike_model is not None:
        if isinstance(spike_model, dict):
            classifier = spike_model.get('classifier') or spike_model.get('pipeline')
            expects_scale = spike_model.get('expects_per_video_minmax', False)
        else:
            classifier = spike_model

    # ROI model details (for gating)
    roi_classifier = None
    roi_norm = 'minmax'
    if roi_model is not None:
        if isinstance(roi_model, dict):
            roi_classifier = roi_model.get('classifier') or roi_model.get('pipeline')
            roi_norm = roi_model.get('normalization', 'minmax')
        else:
            roi_classifier = roi_model

    # If gating, compute ROI features & predictions first
    good_roi_mask = None
    if roi_classifier is not None:
        roi_features = []
        # Use cascade_prob for feature extraction normalization (minmax inside extractor)
        for r_idx in range(F.shape[0]):
            f_trace = F[r_idx]
            prob_trace = np.zeros_like(f_trace)  # placeholder if cascade_prob missing
            # We'll use the normalized fluorescence and computed cascade probabilities if available
            # For feature extraction we follow training path: pass raw f_trace + cascade_prob row
            prob_trace = cascade_prob[r_idx]
            feats = extract_roi_features(f_trace, prob_trace, normalization=roi_norm)
            roi_features.append([feats['derivative_skew'], feats['spike_prom_mean']])
        roi_features_arr = np.array(roi_features)
        roi_preds = roi_classifier.predict_proba(roi_features_arr) if hasattr(roi_classifier, 'predict_proba') else None
        if roi_preds is not None:
            # Class 1 probability
            good_probs = roi_preds[:, 1]
            good_roi_mask = good_probs >= roi_threshold
        else:
            # Fallback to direct class prediction (1 good, 0 bad)
            pred_labels = roi_classifier.predict(roi_features_arr)
            good_roi_mask = pred_labels == 1

    video_id = video_dir.name  # e.g. 03-1

    for roi_idx in range(F.shape[0]):
        if good_roi_mask is not None and not good_roi_mask[roi_idx]:
            continue  # skip bad ROI for spike annotation
        f_trace = F[roi_idx]
        f_norm = F_norm[roi_idx]
        prob_trace = cascade_prob[roi_idx]

        # Detect spikes on normalized probability & raw fluorescence (raw used for index) - consistent with pipeline
        spikes = detect_spikes_from_cascade(f_trace, prob_trace)  # pipeline uses raw_fluorescence; here we used original f_trace
        if len(spikes) == 0:
            continue

        # Extract features (uses raw_fluorescence + cascade_prob already normalized)
        feat_array = extract_spike_features(spikes, f_trace, prob_trace)
        if feat_array.size == 0:
            continue

        # Build spike_keys and DataFrame for this ROI
        spike_frames = [s.frame_index for s in spikes]
        spike_keys = [f"{video_id}_{roi_idx}_{frame}" for frame in spike_frames]

        df_roi = pd.DataFrame(feat_array, columns=TOP_FEATURES)
        df_roi.insert(0, 'roi_index', roi_idx)
        df_roi.insert(0, 'spike_key', spike_keys)

        # Optional per-video scaling (column-wise) if model expects it
        if classifier is not None and (expects_scale):
            scaled_vals, _, _ = _minmax_scale_array(df_roi[TOP_FEATURES].values)
            df_roi[TOP_FEATURES] = scaled_vals

        # Predict probabilities
        if classifier is not None:
            probs = classifier.predict_proba(df_roi[TOP_FEATURES].values)[:, 1]
            df_roi['model_prob'] = probs
            # Priority assignment
            conditions = []
            for p in probs:
                if 0.35 <= p <= 0.65:
                    conditions.append('borderline')
                elif p > 0.9:
                    conditions.append('high_conf_pos')
                elif p < 0.1:
                    conditions.append('high_conf_neg')
                else:
                    conditions.append('mid')
            df_roi['suggested_priority'] = conditions
        else:
            df_roi['model_prob'] = np.nan
            df_roi['suggested_priority'] = 'unscored'

    rows.append(df_roi)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=['spike_key'] + TOP_FEATURES + ['model_prob', 'suggested_priority'])


def main():
    parser = argparse.ArgumentParser(description='Export spike candidates for annotation')
    parser.add_argument('--datasets_root', type=Path, required=True, help='Root datasets folder')
    parser.add_argument('--output', type=Path, required=True, help='Output folder for candidate CSVs')
    parser.add_argument('--model', type=Path, default=Path('spike_classifier/models/spike_classifier.pkl'), help='Spike classifier model path')
    parser.add_argument('--cascade_model_name', type=str, default='Global_EXC_30Hz_smoothing100ms_high_noise')
    parser.add_argument('--cascade_model_dir', type=Path, default=Path('Cascade/Pretrained_models'))
    parser.add_argument('--limit_videos', type=int, default=None, help='Optional limit on number of videos processed')
    parser.add_argument('--roi_model', type=Path, default=None, help='Optional ROI classifier model for gating good ROIs')
    parser.add_argument('--roi_threshold', type=float, default=0.5, help='Probability threshold for good ROI (if ROI model has predict_proba)')
    args = parser.parse_args()

    # Load spike model if exists
    spike_model = None
    if args.model.exists():
        try:
            spike_model = load(args.model)
            logger.info(f"Loaded spike classifier: {args.model}")
        except Exception as e:
            logger.warning(f"Failed to load spike model: {e}")

    # Load cascade model
    from utils import load_cascade_model
    cascade_model = load_cascade_model(args.cascade_model_name, args.cascade_model_dir)

    # Load ROI model if provided
    roi_model = None
    if args.roi_model and args.roi_model.exists():
        try:
            roi_model = load(args.roi_model)
            logger.info(f"Loaded ROI classifier: {args.roi_model}")
        except Exception as e:
            logger.warning(f"Failed to load ROI model: {e}")

    args.output.mkdir(parents=True, exist_ok=True)

    video_dirs = [d for d in args.datasets_root.iterdir() if d.is_dir() and VIDEO_PATTERN.match(d.name)]
    if args.limit_videos:
        video_dirs = video_dirs[:args.limit_videos]

    logger.info(f"Found {len(video_dirs)} video folders to process")

    for vdir in video_dirs:
        logger.info(f"Processing video folder: {vdir.name}")
        df_video = process_video(vdir, cascade_model, spike_model, roi_model, args.roi_threshold)
        if df_video.empty:
            logger.info(f"No spikes found for {vdir.name}")
            continue
        out_file = args.output / f"{vdir.name}_spike_candidates.csv"
        df_video.to_csv(out_file, index=False)
        logger.info(f"Wrote {len(df_video)} candidates to {out_file}")

    logger.info("Export complete")

if __name__ == '__main__':
    main()
