from utils.io_utils import SummaryFiles
from Cascade.cascade2p.cascade_wrapper import CascadePredictor
import os 
from pathlib import Path
from data_classes.neuron import Neuron
import pandas as pd
def load_good_rois(roi_labels_path: Path):
    """
    Load good ROIs from a CSV file.
    """
    roi_labels = pd.read_csv(roi_labels_path)
    good_rois = set(
        (os.path.normpath(str(row['source_file'])), int(row['roi_index']))
        for _, row in roi_labels.iterrows() if row['label'] == 1
    )
    return good_rois
def spike_dataset_feature_computation(dataset_root: Path, good_rois: set,
                                      model_name = "Global_EXC_30Hz_smoothing100ms_high_noise", 
                                      edge=32):
    """
    This function computes features for spikes in Suite2p folders.
    It generates spike probabilities using a Cascade model, computes features for each spike,
    and saves the results to a CSV file.
    """
   
    suite2p_folders = [p for p in dataset_root.rglob('suite2p') if (p / 'plane0' / 'F.npy').exists()]
    feature_list = []
    rows = []
    spk_instances= []
    for suite2p_folder in suite2p_folders:
        summary = SummaryFiles(suite2p_folder, CascadePredictor(model_name))
      

        # 2. For each GOOD ROI, find spikes and compute features
        for roi_idx in range(summary.raw_fluorescence.shape[0]):
            roi_key = (os.path.normpath(str(suite2p_folder / "plane0" / "F.npy")), roi_idx)
            if roi_key not in good_rois:
                print(f"Skipping ROI: {roi_key} (not in good_rois)")

                continue  # Skip ROIs not labeled as good
            #Create instance of Neuron
            roi = Neuron(roi_idx, {}, summary, fs=summary.sampling_rate)

            #Get roi features
            roi._get_features()
            
            #Get roi spikes
            roi._find_spikes()
            
           
            # 3. Compute features for each spike
            for spike in roi.spikes:
                # --- Compute features for this spike ---
                spike._set_roi_index(roi_idx)  # Set the ROI index for this spike
                raw_features = list(spike.features.values())
                feature_names = list(spike.features.keys())
                feature_list.append(raw_features)
                spike_key = f"{suite2p_folder.parts[-3].split('_')[1]}_{suite2p_folder.parts[-2][:3]}_{roi_idx}_{spike.idx_prob}"
                rows.append([
                    spike_key,  # Unique identifier for the spike
                    str(suite2p_folder / "plane0"),  # suite2p path
                    roi_idx,      # ROI index
                    spike.idx_prob,     # Spike index in spike_prob trace
                    spike.idx_raw,        # Spike index in raw fluorescence trac
                    feature_names,  # Feature names
                    raw_features,      # Raw features list
                    None,      # Placeholder for z scored features
                    None,  # Placeholder for label
                ])
                spk_instances.append(spike)