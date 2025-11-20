"""
Prepare per-video MinMax scaled spike training data.
Groups spikes by video using spike_key pattern and applies MinMax scaling
per video to all numeric feature columns (excluding spike_key and label).

spike_key format example: '01_1-2_62_56'
- dataset folder: '01' (maps to C:\Users\...\Datasets\01)
- video id: '1-2' (sub-video within folder)
- neuron index: '62' (unused for grouping)
- spike index: '56' (unused)

Output: training_data/spike_filtering/spike_training_data_scaled.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / 'training_data' / 'spike_filtering' / 'spike_training_data.csv'
OUT_PATH = ROOT / 'training_data' / 'spike_filtering' / 'spike_training_data_scaled.csv'

VIDEO_ID_RE = re.compile(r'^(?P<dataset>[^_]+)_(?P<video>[^_]+)_')

def extract_video_id(spike_key: str) -> str:
    """Return group id '<dataset>_<video>' from spike_key."""
    m = VIDEO_ID_RE.match(spike_key)
    if m:
        return f"{m.group('dataset')}_{m.group('video')}"
    # Fallback: use first two underscore-separated parts
    parts = spike_key.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else spike_key


def per_video_minmax_scale(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Determine feature columns (numeric, excluding spike_key and label)
    exclude = {'spike_key', 'label'}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # Add video_id column
    df['video_id'] = df['spike_key'].apply(extract_video_id)

    # Apply scaling per video_id
    def scale_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for c in feature_cols:
            col = g[c].astype(float)
            cmin = float(np.nanmin(col)) if len(col) else 0.0
            cmax = float(np.nanmax(col)) if len(col) else 1.0
            if np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin:
                g[c] = (col - cmin) / (cmax - cmin)
            else:
                # Constant or non-finite: set to 0.0
                g[c] = 0.0
        return g

    scaled = df.groupby('video_id', group_keys=False).apply(scale_group)
    # Drop helper column
    scaled = scaled.drop(columns=['video_id'])
    return scaled


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {IN_PATH}")
    df = pd.read_csv(IN_PATH)
    scaled = per_video_minmax_scale(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scaled.to_csv(OUT_PATH, index=False)
    print(f"Wrote scaled training data to {OUT_PATH} ({len(scaled)} rows)")

if __name__ == '__main__':
    main()
