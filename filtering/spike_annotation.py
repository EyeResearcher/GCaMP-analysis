import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from utils.io_utils import SummaryFiles, save_cascade_predictions
from Cascade.cascade2p.cascade_wrapper import CascadePredictor
from utils.spike_utils import find_spikes

DATASET_ROOT = Path(input("Enter path to dataset folder: "))
CASCADE_MODEL_NAME = input("Enter Cascade model name: ")
N_ANNOTATE = int(input("How many spikes do you want to annotate? "))
CSV_PATH = Path(input("Enter path to save CSV file: "))

# 1. Find all suite2p folders
suite2p_folders = [p for p in DATASET_ROOT.rglob('suite2p') if (p / 'plane0' / 'F.npy').exists()]

rows = []
for suite2p_folder in suite2p_folders:
    plane0 = suite2p_folder / 'plane0'
    F_path = plane0 / 'F.npy'
    spike_prob_path = plane0 / 'cascade_spike_prob.npy'
    # 2. Generate spike_prob if missing
    if not spike_prob_path.exists():
        print(f"Generating spike_prob for {plane0}")
        raw_fluorescence = np.load(F_path)
        cascade_model = CascadePredictor(CASCADE_MODEL_NAME)
        spike_prob = cascade_model.predict(raw_fluorescence)
        save_cascade_predictions(plane0, spike_prob)
    else:
        print(f"Spike_prob already exists for {plane0}")

    # 3. For each ROI, find spikes and write to CSV
    raw_fluorescence = np.load(F_path)
    spike_prob = np.load(spike_prob_path)
    for roi_idx in range(raw_fluorescence.shape[0]):
        roi_f = raw_fluorescence[roi_idx]
        roi_spike_prob = spike_prob[roi_idx]
        # Use your find_spikes function (adapt as needed)
        spike_indices_prob, _, spike_indices_raw, _ = find_spikes(roi_spike_prob, roi_f)
        for idx_prob, idx_raw in zip(spike_indices_prob, spike_indices_raw):
            rows.append([
                str(plane0),  # suite2p path
                roi_idx,      # ROI index
                idx_prob,     # Spike index in spike_prob trace
                idx_raw,      # Spike index in raw_f trace
                "",           # Feature placeholder
                ""            # Label placeholder
    ])
# 4. Save all rows to CSV
df = pd.DataFrame(rows, columns=["suite2p_path", "roi_index", "spike_prob_index", "raw_f_index", "feature", "label"])
df.to_csv(CSV_PATH, index=False)
print(f"Wrote {len(df)} spikes to {CSV_PATH}")

# 5. Annotation: randomly sample N spikes
df = pd.read_csv(CSV_PATH)
sampled = df.sample(n=min(N_ANNOTATE, len(df)), random_state=42).reset_index(drop=True)

# 6. Launch annotation GUI for each sampled spike
from .spike_labeler_gui import SpikeLabelerApp  # You need to implement this as in previous answers
bad_rois = set()
labels = []

def on_spike_labels_done(result, roi_idx):
    if result == "BAD_ROI":
        bad_rois.add(roi_idx)
    else:
        labels.extend(result)

for i, row in sampled.iterrows():
    suite2p_path = Path(row['suite2p_path'])
    roi_idx = int(row['roi_index'])
    spike_prob_idx = int(row['spike_prob_index'])  # or spike_prob_index if you renamed
    raw_f = np.load(suite2p_path / 'F.npy')[roi_idx]
    spike_prob = np.load(suite2p_path / 'cascade_spike_prob.npy')[roi_idx]

    if roi_idx in bad_rois:
        labels.append("")  # or np.nan, or skip appending
        continue

    # The callback now takes both the result and the roi_idx
    SpikeLabelerApp(
        raw_f, spike_prob, [spike_prob_idx], [spike_prob_idx],  # pass both indices if needed
        lambda l, roi_idx=roi_idx: on_spike_labels_done(l, roi_idx),
        neuron_id=roi_idx
    )

# Remove all rows with roi_index in bad_rois
sampled = sampled[~sampled['roi_index'].isin(bad_rois)]
sampled['label'] = labels[:len(sampled)]
sampled.to_csv("spike_annotations_labeled.csv", index=False)
print("Annotation complete and saved.")