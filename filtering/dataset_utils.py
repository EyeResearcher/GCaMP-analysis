from utils.io_utils import SummaryFiles
from Cascade.cascade2p.cascade_wrapper import CascadePredictor
import os 
from pathlib import Path
from data_classes.neuron import Neuron
import pandas as pd
from filtering.feature_utils import zscore_features
from data_classes.spike import Spike
import numpy as np
from typing import Type, TypeVar, Dict, Any
import ast
from joblib import parallel, delayed
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def reinstantiate_objects(class_type : Type[TypeVar], instance_dict : dict):
    """
    Reinstantiate an object of class `class_type` from its dictionary representation.
    Args:
        class_type (Type[TypeVar]): The class type to instantiate.
        instance_dict (dict): A dictionary containing the attributes of the instance.
    Returns:
        TypeVar: An instance of `class_type` with attributes set from `instance_dict`.
    """
    if not isinstance(instance_dict, dict):
        instance_dict = ast.literal_eval(instance_dict)  # Convert string representation to dict if needed
    instance = class_type.__new__(class_type)  # Create an uninitialized instance
    instance.__dict__.update(instance_dict)  # Assign the dictionary to the instance
    return instance

def features_only(features_df: pd.DataFrame):
    """ This function takes an existing features DataFrame and only computes new features for the spikes.
        It returns a DataFrame with the new features and feature names.
    Args:
        features_df (pd.DataFrame): The DataFrame containing spike data with existing features.
    Returns:
        features_df (pd.DataFrame): A DataFrame containing the new features and feature names.
    """
    
    spk_instances_new = []
  
    features_new = []
    feature_names_new = []
    video_summaries = {}
    features_by_roi = {}
    # Iterate through each row in the DataFrame
    for __ , row in features_df.iterrows():
        #Intialize spike key 
        spike_key : str = row["spike_key"]
        video_id  : str = f"{spike_key.split('_')[0]}_{spike_key.split('_')[1]}"
        roi_id    : str = f"{video_id}_{row['roi_index']}"
        # Check if the video summary is already loaded
        if video_id not in video_summaries.keys():
            summary_instance : SummaryFiles = SummaryFiles(row['suite2p_path'])
            summary_instance.load_files()  # Load the summary files for this instance
            summary_instance._create_spike_prob(new_model=False)  # Ensure spike probabilities are created
            video_summaries[video_id] = summary_instance
        
        if roi_id not in features_by_roi.keys():
            features_by_roi[roi_id] = {}
        # Redefine neuron features
        neuron_features : dict        = ast.literal_eval(row["neuron_features"]) if isinstance(row["neuron_features"], str) else row["neuron_features"]
        lb_prom         : np.ndarray  = np.array(ast.literal_eval(row["left_based_prominences"])) if isinstance(row["left_based_prominences"], str) else row["left_based_prominences"]

        # Reinstantiate Spike and Neuron objects from their stored dictionaries
        spk_instance    : Spike       = reinstantiate_objects(Spike, row['spk_instance'])
        
        neuron_instance : Neuron = Neuron(row['roi_index'], neuron_features, video_summaries[video_id], fs=15)
        neuron_instance.left_base_prominences = lb_prom

        #Compute features for the spike instance
        spk_instance.compute_features(spk_instance.i, neuron_instance.raw_fluorescence,
                                    neuron_instance.spike_prob, neuron_instance.left_base_prominences, 
                                    neuron_instance.features["spike_prom_skew"])
        spk_instances_new.append(spk_instance.__dict__)
        # Append the features and feature names to the lists
        features_by_roi[roi_id][spike_key] = list(spk_instance.features.values())
        features_new.append(list(spk_instance.features.values()))
        feature_names_new.append(list(spk_instance.features.keys()))

    #Recreate features z scored within a neuron
    for spike_features_dict in list(features_by_roi.values()):
        z_features = zscore_features(np.array(list(spike_features_dict.values())))
        for i, key in enumerate(spike_features_dict.keys()):
            row_idx = features_df.index[features_df["spike_key"] == key][0]
            features_df.at[row_idx, "raw_z_features"] = z_features[i]
    features_df.reset_index()

    # Add the new features and feature names to the DataFrame
    features_df["spk_instance"] = spk_instances_new
    features_df["z_scored_z_features"] = zscore_features(np.array(features_df["raw_z_features"].tolist()))
    features_df["z_scored_raw_features"] = zscore_features(np.array(features_new))
    features_df["raw_features"] = features_new
    features_df["feature_names"] = feature_names_new
    return features_df
def _process_one_suite2p(suite2p_folder, good_rois, model_name, edge, new_model):
    summary = SummaryFiles(suite2p_folder.parent, CascadePredictor(model_name=model_name), new_model = new_model)
    summary.load_files()  # Load the summary files for this instance
    summary._create_spike_prob(new_model=new_model)  # Ensure spike probabilities are created
    feature_list = []
    z_features_list = []
    rows = []
    skipped = 0
    # 2. For each GOOD ROI, find spikes and compute features
    for roi_idx in range(summary.raw_fluorescence.shape[0]):
        roi_key = (os.path.normpath(str(suite2p_folder / "plane0" / "F.npy")), roi_idx)
        if roi_key not in good_rois:
            skipped += 1

            continue  # Skip ROIs not labeled as good
        #Create instance of Neuron
        roi = Neuron(roi_idx, {}, summary, fs=summary.sampling_rate)

        #Get roi features
        roi._get_features()
        
        #Get roi spikes
        roi._find_spikes()

        #Get all spike features
        roi._compute_spike_features()
        
        #Get spike features z scored within the roi 
        roi._zscore_spike_features()

        # 3. Compute features for each spike
        for spike in roi.spikes:
            #Set the ROI index for the spike
            spike._set_roi_index(roi_idx)  

            #Get the raw features and feature names
            raw_features = list(spike.features.values())
            z_features = list(spike.z_features) #this is stored as an array whose order matches order of storage for regular features
            feature_names = list(spike.features.keys())

            #Add the raw features to the feature list for zscoring later
            feature_list.append(raw_features)
            z_features_list.append(z_features)

            #Create a unique spike key
            spike_key = f"{suite2p_folder.parts[-3].split('_')[1]}_{suite2p_folder.parts[-2][:3]}_{roi_idx}_{spike.idx_prob}"

            # Create a dictionary representation of the spike instance
            spike_dict = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in spike.__dict__.items()}
            for key, value in spike_dict.items():
                spike_dict[key] = list(value) if isinstance(value,np.ndarray) else value

            # Append the spike data to the rows list
            rows.append([
                spike_key,  # Unique identifier for the spike
                str(suite2p_folder / "plane0"),  # suite2p path
                roi_idx,      # ROI index
                spike_dict,  # Spike instance attributes
                spike.idx_prob,     # Spike index in spike_prob trace
                spike.idx_raw,        # Spike index in raw fluorescence trac
                roi.features,       #Neuron level information
                list(roi.left_base_prominences),  # Left base prominences
                feature_names,  # Feature names
                raw_features,      # Raw features list
                z_features,
                None,      # Placeholder for z scored features
                None,

            ])
    return rows, feature_list, z_features_list, skipped

def spike_dataset_feature_computation(dataset_root: Path,
                                      good_rois: set,
                                      model_name="Global_EXC_15Hz_smoothing100ms_high_noise",
                                      edge=32,
                                      new_model=False,
                                      n_jobs=4):
    # 1. find all suite2p folders
    suite2p_folders = [
        p for p in dataset_root.rglob('suite2p')
        if (p / 'plane0' / 'F.npy').exists()
    ]
    print(f"Found {len(suite2p_folders)} suite2p folders in {dataset_root}")

    all_rows        = []
    all_raw_feats   = []
    all_z_feats     = []
    total_skipped   = 0

    # 2. parallel map
    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        futures = {
            exe.submit(
                _process_one_suite2p,
                folder, good_rois, model_name, edge, new_model
            ): folder
            for folder in suite2p_folders
        }
        for fut in as_completed(futures):
            rows, rawf, zf, skipped = fut.result()
            all_rows      .extend(rows)
            all_raw_feats .extend(rawf)
            all_z_feats   .extend(zf)
            total_skipped += skipped

    print(f"Skipped {total_skipped} ROIs not labeled as good.")

    # 3. assemble DataFrame
    cols = [
        "spike_key", "suite2p_path", "roi_index", "spk_instance",
        "spike_prob_index", "raw_f_index", "neuron_features",
        "left_based_prominences", "feature_names",
        "raw_features","raw_z_features",
        "z_scored_raw_features", "z_scored_z_features"
    ]
    df = pd.DataFrame(all_rows, columns=cols)

    # 4. compute z-scores
    df['z_scored_raw_features'] = [list(row) for row in  (zscore_features(all_raw_feats))]
    df['z_scored_z_features']   =[list(row) for row in  (zscore_features(all_z_feats))]

    return df
def spike_dataset_feature_computation_old(dataset_root: Path, good_rois: set,
                                      model_name = "Global_EXC_15Hz_smoothing100ms_high_noise", 
                                      edge=32, new_model = False):
    """
    This function finds all suite2p folders in the dataset root directory and creates summary files for each ROI, computing spike proababilities if a new model is specified.
    It then computes spike features for each good ROI and returns a DataFrame containing the spike data.
    Args:
        dataset_root (Path): The root directory containing the dataset. 
        good_rois (set): A set of tuples containing the source file path and ROI index for good ROIs.
        model_name (str): The name of the Cascade model to use for spike probability computation.
        edge (int): The edge size for the Cascade model.
        new_model (bool): If True, recomputes spike probabilities using the specified Cascade model.
    Returns:
        pd.DataFrame: A DataFrame containing spike data, including spike keys, suite2p paths, ROI indices, spike attributes, and computed features.
    """
   
    suite2p_folders = [p for p in dataset_root.rglob('suite2p') if (p / 'plane0' / 'F.npy').exists()]
    feature_list = []
    z_features_list = []
    rows = []
    spk_instances= []
    print(f"Found {len(suite2p_folders)} suite2p folders in {dataset_root}")
    skipped = 0
    for suite2p_folder in suite2p_folders:
        summary = SummaryFiles(suite2p_folder.parent, CascadePredictor(model_name=model_name), new_model = new_model)
        summary.load_files()  # Load the summary files for this instance
        summary._create_spike_prob(new_model=new_model)  # Ensure spike probabilities are created

        # 2. For each GOOD ROI, find spikes and compute features
        for roi_idx in range(summary.raw_fluorescence.shape[0]):
            roi_key = (os.path.normpath(str(suite2p_folder / "plane0" / "F.npy")), roi_idx)
            if roi_key not in good_rois:
                skipped += 1

                continue  # Skip ROIs not labeled as good
            #Create instance of Neuron
            roi = Neuron(roi_idx, {}, summary, fs=summary.sampling_rate)

            #Get roi features
            roi._get_features()
            
            #Get roi spikes
            roi._find_spikes()

            #Get all spike features
            roi._compute_spike_features()
            
            #Get spike features z scored within the roi 
            roi._zscore_spike_features()

            # 3. Compute features for each spike
            for spike in roi.spikes:
                #Set the ROI index for the spike
                spike._set_roi_index(roi_idx)  

                #Get the raw features and feature names
                raw_features = list(spike.features.values())
                z_features = list(spike.z_features) #this is stored as an array whose order matches order of storage for regular features
                feature_names = list(spike.features.keys())

                #Add the raw features to the feature list for zscoring later
                feature_list.append(raw_features)
                z_features_list.append(z_features)

                #Create a unique spike key
                spike_key = f"{suite2p_folder.parts[-3].split('_')[1]}_{suite2p_folder.parts[-2][:3]}_{roi_idx}_{spike.idx_prob}"

                # Create a dictionary representation of the spike instance
                spike_dict = spike.__dict__
                for key, value in spike_dict.items():
                    spike_dict[key] = list(value) if isinstance(value,np.ndarray) else value

                # Append the spike data to the rows list
                rows.append([
                    spike_key,  # Unique identifier for the spike
                    str(suite2p_folder / "plane0"),  # suite2p path
                    roi_idx,      # ROI index
                    spike_dict,  # Spike instance attributes
                    spike.idx_prob,     # Spike index in spike_prob trace
                    spike.idx_raw,        # Spike index in raw fluorescence trac
                    roi.features,       #Neuron level information
                    list(roi.left_base_prominences),  # Left base prominences
                    feature_names,  # Feature names
                    raw_features,      # Raw features list
                    z_features,
                    None,      # Placeholder for z scored features
                    None,
    
                ])
                spk_instances.append(spike)
    print(f"Skipped {skipped} ROIs not labeled as good.")
    # Create a DataFrame from the collected rows
    features_df = pd.DataFrame(rows, columns=["spike_key", "suite2p_path", "roi_index", "spk_instance",
                                              "spike_prob_index", "raw_f_index", "neuron_features", "left_based_prominences",
                                              "feature_names", "raw_features","raw_z_features", "z_scored_raw_features", "z_scored_z_features"])
    #Add z scores
    features_zscore = zscore_features(feature_list) 
    features_df['z_scored_raw_features'] = features_zscore
    z_features_zscore = zscore_features(z_features_list)
    features_df['z_scored_z_features'] = z_features_zscore
    return features_df