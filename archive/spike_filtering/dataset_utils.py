from utils.io_utils import SummaryFiles
from Cascade.cascade2p.cascade_wrapper import CascadePredictor
import os 
from pathlib import Path
from data_classes.neuron import Neuron
import pandas as pd
from spike_filtering.feature_utils import zscore_features
from data_classes.spike import Spike
import numpy as np
from typing import Type, TypeVar
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

def features_only(reinst_df: pd.DataFrame):
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
    for __ , row in reinst_df.iterrows():
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
        neuron_instance._find_valleys()
        try:
            neuron_instance._preprocess_spikes_and_valleys() 
        except Exception as e:
            raise ValueError(f"Error in preprocessing spikes and valleys for spike {spike_key}: {e}")

        #Compute features for the spike instance
        spk_instance.compute_features(spk_instance.i, neuron_instance.raw_fluorescence,
                                    neuron_instance.spike_prob, neuron_instance.left_base_prominences, 
                                    neuron_instance.features["spike_prom_skew"], neuron_instance.valleys)
        spk_instances_new.append(spk_instance.__dict__)
        # Append the features and feature names to the lists
        features_by_roi[roi_id][spike_key] = list(spk_instance.features.values())
        feature_df_row = list(spk_instance.features.values())
        feature_df_row.insert(0, spike_key)
        features_new.append(feature_df_row)
        feature_cols = [f"feature_{i}" for i in range(len(feature_df_row)-1)]
        feature_cols.insert(0, "spike_key")
        feature_names_new.append(feature_cols)

    #Recreate features z scored within a neuron
    """for spike_features_dict in list(features_by_roi.values()):
        z_features = zscore_features(np.array(list(spike_features_dict.values())))
        for i, key in enumerate(spike_features_dict.keys()):
            row_idx = reinst_df.index[reinst_df["spike_key"] == key][0]
            reinst_df.at[row_idx, "raw_z_features"] = z_features[i]"""
    reinst_df.reset_index()

    # Add the new features and feature names to the DataFrame
    reinst_df["spk_instance"] = spk_instances_new
    #reinst_df["z_scored_z_features"] = zscore_features(np.array(reinst_df["raw_z_features"].tolist()))
    #reinst_df["z_scored_raw_features"] = zscore_features(np.array(features_new))
    #reinst_df["raw_features"] = features_new
    #reinst_df["feature_names"] = feature_names_new
    feature_df = pd.DataFrame(features_new, columns=feature_cols)
    return reinst_df, feature_df

def plot_feature_distributions(df: pd.DataFrame, cols=None, bins=40, figsize=(10,4),
                               qq=True, show_stats=True, max_plots=None):
    """
    Plot distribution and Q-Q plot for each numeric column in df (or provided cols).
    Returns a dict of stats: {col: {'n':..., 'mean':..., 'std':..., 'skew':..., 'kurtosis':..., 'shapiro_stat':..., 'shapiro_p':...}}
    - cols: list of column names to analyze (default: all numeric columns)
    - qq: whether to include Q-Q plot
    - show_stats: annotate the plots with basic stats
    - max_plots: limit number of features plotted (None => all)
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c in df.columns]
    if max_plots is not None:
        cols = cols[:max_plots]

    results = {}
    for col in cols:
        series = df[col].dropna().astype(float)
        n = len(series)
        if n == 0:
            results[col] = {'n': 0}
            continue

        mean = float(series.mean())
        std = float(series.std(ddof=1))
        skew_v = float(stats.skew(series))
        kurt_v = float(stats.kurtosis(series))  # fisher's (0 for normal)
        # Shapiro-Wilk (note: valid for n <= 5000)
        try:
            sh_stat, sh_p = stats.shapiro(series) if n <= 5000 else (np.nan, np.nan)
        except Exception:
            sh_stat, sh_p = (np.nan, np.nan)

        results[col] = {
            'n': n, 'mean': mean, 'std': std,
            'skew': skew_v, 'kurtosis': kurt_v,
            'shapiro_stat': sh_stat, 'shapiro_p': sh_p
        }

        # Plotting
        fig, axes = plt.subplots(1, 2 if qq else 1, figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Histogram + KDE
        sns.histplot(series, bins=bins, kde=True, stat='density', ax=axes[0], color='C0')
        axes[0].set_title(f"{col} (n={n})")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Density')

        if show_stats:
            stat_text = f"mean={mean:.3f}\nstd={std:.3f}\nskew={skew_v:.3f}\nkurt={kurt_v:.3f}"
            if not np.isnan(sh_p):
                stat_text += f"\nShapiro p={sh_p:.3g}"
            axes[0].text(0.95, 0.95, stat_text, transform=axes[0].transAxes,
                         ha='right', va='top', fontsize=9, bbox=dict(boxstyle="round", fc="w", alpha=0.8))

        # Q-Q plot
        if qq:
            qq_ax = axes[1]
            stats.probplot(series, dist="norm", plot=qq_ax)
            qq_ax.set_title(f"Q-Q plot: {col}")

        plt.tight_layout()
        plt.show()

    return results
def _process_one_suite2p(suite2p_folder, roi_labels_path, model_name, edge, new_model):
    good_rois = load_good_rois(roi_labels_path)
    summary = SummaryFiles(suite2p_folder.parent, CascadePredictor(model_name=model_name), new_model = new_model)
    summary.load_files()  # Load the summary files for this instance
    summary._create_spike_prob(new_model=new_model)  # Ensure spike probabilities are created
    feature_list = []
    z_features_list = []
    reinst_rows = []
    feature_rows = []
    skipped = 0
    feature_names = set()
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
        print("Computing features for ROI", roi_idx, "in", suite2p_folder)
        #Get roi spikes
        roi._find_spikes()
        print(f"  Found {len(roi.spikes)} spikes")
        #Get Valleys
        roi._find_valleys()
        print(f"  Found {len(roi.valleys)} valleys")
        #Preprocess spikes and valleys (normalize, couple, rank)
        roi._preprocess_spikes_and_valleys()
        print(f"  Processed {len(roi.spikes)} spikes and {len(roi.valleys)} valleys")
        #Get all spike features
        roi._compute_spike_features()
        print(f"  Computed features for {len(roi.spikes)} spikes")
        #Get spike features z scored within the roi 
        roi._zscore_spike_features()

        #Get spike features minmax scaled within the roi
        roi._minmax_spike_features()

        # 3. Compute features for each spike
        for spike in roi.spikes:
            #Set the ROI index for the spike
            spike._set_roi_index(roi_idx)  

            #Get the raw features and feature names
            raw_features = list(spike.features.values())
            z_features = list(spike.z_features) #this is stored as an array whose order matches order of storage for regular features
            minmax_features = list(spike.mm_features)
            feature_names_roi = list(spike.features.keys())
            if len(feature_names) == 0:
                feature_names = set(feature_names_roi)
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
            reinstantiation_values = [spike_key, str(suite2p_folder / "plane0"), roi_idx, spike_dict, spike.idx_prob, spike.idx_raw, roi.features, list(roi.left_base_prominences)]
            dataframe_values = minmax_features
            dataframe_values.insert(0, spike_key)
            # Append the spike data to the rows list
            reinst_rows.append(reinstantiation_values)
            feature_rows.append(dataframe_values)
            """rows.append([
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
                minmax_features,
                None,      # Placeholder for z scored features
                None,

            ])"""
    return feature_names, feature_rows, reinst_rows, feature_list, z_features_list, skipped

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

    all_reinst_rows        = []
    all_raw_feats   = []
    all_z_feats     = []
    total_skipped   = 0
    feature_rows   = []
    feat_names = set()
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
            feat_names, feat_rows, reinst_rows, rawf, zf, skipped = fut.result()
            feat_names = set(feat_names)
            all_reinst_rows      .extend(reinst_rows)
            feature_rows         .extend(feat_rows)
            all_raw_feats .extend(rawf)
            all_z_feats   .extend(zf)
            total_skipped += skipped

    print(f"Skipped {total_skipped} ROIs not labeled as good.")

    # 3. assemble DataFrame
    reinst_cols = [
        "spike_key", "suite2p_path", "roi_index", "spk_instance",
        "spike_prob_index", "raw_f_index", "neuron_features",
        "left_based_prominences"]
    feat_cols = list(feat_names)
    feat_cols.insert(0, "spike_key")

    reinst_df = pd.DataFrame(all_reinst_rows, columns=reinst_cols)
    feat_df = pd.DataFrame(feature_rows, columns=feat_cols)


    # 4. compute z-scores
   
    return reinst_df, feat_df
