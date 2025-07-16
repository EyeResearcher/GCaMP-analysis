import os
import numpy as np
import pandas as pd
from pathlib import Path
from utils.io_utils import save_cascade_predictions, SummaryFiles
from data_classes.neuron import Neuron
from Cascade.cascade2p.cascade_wrapper import CascadePredictor
from utils.spike_utils import find_spikes
from .feature_utils import compute_spike_features, zscore_features  # <-- Import your feature function
from scipy.signal import peak_prominences
from scipy.stats import skew, zscore    
from .spike_annotation import main_annotate
from .dataset_utils import load_good_rois, spike_dataset_feature_computation

# ---- USER INPUTS ----
DATASET_ROOT = Path(input("Enter path to dataset folder: "))
CASCADE_MODEL_NAME = input("Enter Cascade model name: ")
N_ANNOTATE = int(input("How many spikes do you want to annotate? "))
MODEL_VERSION_FOLDER = Path(input("Enter path to the umbrella folder for this model and information: "))
ROI_LABELS_PATH = Path(input("Enter path to roi_labels.csv: "))

# ---- CREATE FOLDERS FOR THIS RUN ----
SPIKE_FILTERING_FOLDER = MODEL_VERSION_FOLDER / 'spike_filtering'
SPIKE_FILTERING_FOLDER.mkdir(parents=True, exist_ok=True)
FEATURES_CSV_PATH = SPIKE_FILTERING_FOLDER / 'spike_features.csv'
ANNOTATION_PATH = SPIKE_FILTERING_FOLDER / 'spike_annotations.csv'

# ---- LOAD GOOD ROIS ----
good_rois = load_good_rois(ROI_LABELS_PATH)

# ---- FIND SUITE2P FOLDERS ----

features_df, features_df_spk_instance = spike_dataset_feature_computation(DATASET_ROOT, good_rois)
print(f"Computed features for {len(features_df)} spikes.")
annotate_yn = input("Do you want to annotate spikes? (y/n): ").strip().lower()
if annotate_yn == 'y':
    # ---- ANNOTATE SPIKES ----
    features_df = pd.read_csv(FEATURES_CSV_PATH) if features_df_spk_instance is None else features_df_spk_instance
    annotated_df = main_annotate(features_df)
    annotated_df.to_csv(FEATURES_CSV_PATH, index=False)
    print(f"Annotation complete and saved to {FEATURES_CSV_PATH}")