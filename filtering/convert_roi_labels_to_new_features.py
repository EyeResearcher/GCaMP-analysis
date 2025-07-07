import pandas as pd
import numpy as np
from pathlib import Path
from feature_utils import extract_features
from joblib import load

# === CONFIG ===
LABELS_CSV = Path(r"C:\Users\mzinn1\Desktop\Scripts\post_suite2p_analysis\model_runs\GCaMP8s_Olympus_Glass\roi_filtering\roi_labels.csv")  # original
SAVE_CSV = LABELS_CSV.parent / "roi_labels_v2.csv"
MODEL_DIR = Path(r"C:\Users\mzinn1\Desktop\Scripts\post_suite2p_analysis\model_runs\GCaMP8s_Olympus_Glass")  # where template_traces.npz lives
# ==============

# Load template traces
template_path = MODEL_DIR / "roi_filtering" / "template_traces.npz"
templates = np.load(template_path)
template_tuple = (templates["few"], templates["med"], templates["many"])

# Load old CSV
df_old = pd.read_csv(LABELS_CSV)
print("Loaded:", df_old.shape)

# Prepare new features
new_features = []
missing = []

for i, row in df_old.iterrows():
    file_path = Path(row["source_file"])
    roi = int(row.name) if "roi" not in row else int(row["roi"])  # fallback
    try:
        F = np.load(file_path)
        raw = F[roi]
        norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
        feats = extract_features(norm, raw, template_tuple)
        new_features.append(feats)
    except Exception as e:
        print(f"Skipping ROI {roi} in {file_path}: {e}")
        missing.append((file_path, roi))
        continue

# Assemble new DataFrame
df_feats = pd.DataFrame(new_features)
df_feats["label"] = df_old["label"].values
df_feats["source_file"] = df_old["source_file"].values
df_feats.to_csv(SAVE_CSV, index=False)
print("Saved upgraded features to:", SAVE_CSV)
