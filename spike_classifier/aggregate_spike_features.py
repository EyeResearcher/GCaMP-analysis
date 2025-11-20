r"""
Aggregate spike features for ALL spikes from GOOD (positive) ROIs into a single CSV.

Workflow:
- Iterate dataset videos under --datasets_root (expects */suite2p/plane0 structure)
- Load Suite2p F.npy and metadata
- Compute Cascade probabilities (or use provided model)
- Gate ROIs using the ROI classifier (prob >= --roi_threshold)
- Detect spikes for each GOOD ROI
- Extract both top-3 spike features and detailed metrics
- Write a unified CSV suitable for the spike annotation GUI

Output schema (columns):
- spike_key: string "<video_id>_<roi_index>_<frame_index>"
- video_id: video folder name
- roi_index: int
- frame_index: int (fluorescence peak frame)
- cascade_peak_idx: int (peak on cascade prob used to align)
- prob_value: float (cascade prob at peak)
- f_value: float (fluorescence value at peak)
- skew_contribution, spike_prob_value, max_second_derivative_raw: top-3 features
- amplitude, relative_amplitude, rise_time, decay_time, fwhm, auc: detailed metrics

Notes:
- ROI gating uses the same feature extraction as training (minmax/deltaf per model dict)
- NaN-safe and shape-safe handling is leveraged from existing pipeline modules

"""
from pathlib import Path
import argparse
import logging
import sys
import numpy as np
import pandas as pd
from joblib import load
import re

# Ensure project root is on path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.preprocessing import load_suite2p_data, compute_cascade_probabilities
from pipeline.spike_detection import detect_spikes_from_cascade
from pipeline.spike_filtering import extract_spike_features
from pipeline.spike_filtering import compute_spike_metrics  # detailed metrics
from roi_classifier.feature_extraction import extract_roi_features
from sklearn.preprocessing import MinMaxScaler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Accept basic 2-1 plus names with suffixes like 3-2_Glut_2m
VIDEO_PATTERN = re.compile(r'^(?P<vid>[0-9]+-[0-9]+)(?:_.+)?$')

TOP_FEATURES = ['skew_contribution', 'spike_prob_value', 'max_second_derivative_raw']
DETAIL_FEATURES = ['amplitude', 'relative_amplitude', 'rise_time', 'decay_time', 'fwhm', 'auc']



def process_video(video_dir: Path, cascade_model, roi_model, roi_threshold: float, dataset_tag: str) -> pd.DataFrame:
    suite2p_path = video_dir / 'suite2p' / 'plane0'
    if not suite2p_path.exists():
        logger.warning(f"Skipping {video_dir}, suite2p/plane0 not found")
        return pd.DataFrame()

    data = load_suite2p_data(suite2p_path)
    F = data['F']  # (n_rois, n_frames)
    fs = data.get('fs', 30.0)

    # Per-ROI MinMax normalization for Cascade input and feature extraction
    f_minmax = MinMaxScaler().fit_transform(F)
    # Compute Cascade probabilities on normalized traces
    cascade_prob = compute_cascade_probabilities(F, cascade_model)

    # Setup ROI classifier
    roi_classifier = None
    roi_norm = 'minmax'
    if roi_model is not None:
        if isinstance(roi_model, dict):
            roi_classifier = roi_model.get('classifier') or roi_model.get('pipeline')
            roi_norm = roi_model.get('normalization', 'minmax')
        else:
            roi_classifier = roi_model

    # Gate ROIs
    good_roi_mask = np.ones(F.shape[0], dtype=bool) if roi_classifier is None else None
    if roi_classifier is not None:
        roi_features = []
        for r_idx in range(F.shape[0]):
            f_trace = F[r_idx]
            prob_trace = cascade_prob[r_idx]
            feats = extract_roi_features(f_trace, prob_trace, normalization=roi_norm)
            roi_features.append([feats['derivative_skew'], feats['spike_prom_mean']])
        roi_features_arr = np.array(roi_features)
        if hasattr(roi_classifier, 'predict_proba'):
            probs = roi_classifier.predict_proba(roi_features_arr)[:, 1]
            good_roi_mask = probs >= roi_threshold
        else:
            labels = roi_classifier.predict(roi_features_arr)
            good_roi_mask = labels == 1

    video_id = video_dir.name

    # Aggregate spikes for GOOD ROIs
    rows = []
    for roi_idx in range(F.shape[0]):
        if not good_roi_mask[roi_idx]:
            continue
        # Use per-ROI MinMax normalized fluorescence throughout
        f_trace = F_norm[roi_idx]
        prob_trace = cascade_prob[roi_idx]

        # Detect spikes
        spikes = detect_spikes_from_cascade(f_trace, prob_trace)
        if len(spikes) == 0:
            continue

        # Top-3 features
        feat_array = extract_spike_features(spikes, f_trace, prob_trace)
        if feat_array.size == 0:
            continue

        # Build per-spike rows with detailed metrics
        for i, sp in enumerate(spikes):
            frame_idx = sp.frame_index
            metrics = compute_spike_metrics(sp, f_trace)
            row = {
                'spike_key': f"{dataset_tag}_{video_id}_{roi_idx}_{frame_idx}",
                'video_id': video_id,
                'roi_index': roi_idx,
                'frame_index': frame_idx,
                'cascade_peak_idx': sp.cascade_peak_idx,
                'prob_value': float(sp.prob_height),
                # Store normalized fluorescence value at the spike
                'f_value': float(f_trace[frame_idx]) if 0 <= frame_idx < len(f_trace) else float(sp.f_value),
                # top-3 features
                'skew_contribution': float(feat_array[i, 0]),
                'spike_prob_value': float(feat_array[i, 1]),
                'max_second_derivative_raw': float(feat_array[i, 2]),
                # detailed metrics
                'amplitude': float(metrics.get('amplitude', 0.0)),
                'relative_amplitude': float(metrics.get('relative_amplitude', 0.0)),
                'rise_time': float(metrics.get('rise_time', np.nan)) if metrics.get('rise_time', np.nan) is not None else np.nan,
                'decay_time': float(metrics.get('decay_time', np.nan)) if metrics.get('decay_time', np.nan) is not None else np.nan,
                'fwhm': float(metrics.get('fwhm', 0.0)),
                'auc': float(metrics.get('auc', 0.0)),
            }
            rows.append(row)

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=['spike_key'])


def main():
    ap = argparse.ArgumentParser(description='Aggregate spike features from GOOD ROIs to a single CSV')
    ap.add_argument('--datasets_root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='Path to write unified spike_features CSV')
    ap.add_argument('--cascade_model_name', type=str, default='Global_EXC_30Hz_smoothing100ms_high_noise')
    ap.add_argument('--cascade_model_dir', type=Path, default=Path('Cascade/Pretrained_models'))
    ap.add_argument('--roi_model', type=Path, default=Path('roi_classifier/models/roi_classifier.pkl'))
    ap.add_argument('--roi_threshold', type=float, default=0.5)
    ap.add_argument('--limit_videos', type=int, default=None)
    args = ap.parse_args()

    
    from utils import load_cascade_model
    cascade_model = load_cascade_model(args.cascade_model_name, args.cascade_model_dir)

    roi_model = None
    if args.roi_model and args.roi_model.exists():
        try:
            roi_model = load(args.roi_model)
            logger.info(f"Loaded ROI classifier: {args.roi_model}")
        except Exception as e:
            logger.warning(f"Failed to load ROI model: {e}")

    # Discover videos
    video_dirs = [d for d in args.datasets_root.iterdir() if d.is_dir() and VIDEO_PATTERN.match(d.name)]
    if args.limit_videos:
        video_dirs = video_dirs[:args.limit_videos]

    logger.info(f"Found {len(video_dirs)} video folders to process")

    # Process each video and append
    all_rows = []
    # Derive dataset tag from datasets_root folder name (last two digits if present)
    root_name = args.datasets_root.name
    dataset_tag_match = re.search(r'(\d{2})$', root_name)
    dataset_tag = dataset_tag_match.group(1) if dataset_tag_match else root_name[:2]

    for vdir in video_dirs:
        logger.info(f"Processing video: {vdir.name}")
        df_video = process_video(vdir, cascade_model, roi_model, args.roi_threshold, dataset_tag)
        if not df_video.empty:
            all_rows.append(df_video)
        else:
            logger.info(f"No spikes from GOOD ROIs for {vdir.name}")

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
    else:
        df_all = pd.DataFrame(columns=['spike_key'] + TOP_FEATURES + DETAIL_FEATURES)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(args.output, index=False)
    logger.info(f"Wrote {len(df_all)} spikes to {args.output}")


if __name__ == '__main__':
    main()
