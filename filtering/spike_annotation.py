from .spike_labeler_gui import SpikeLabelerApp
import pandas as pd
import numpy as np
from data_classes.spike import Spike
from pathlib import Path
from .dataset_utils import reinstantiate_objects
def create_spike_instances(features_df: pd.DataFrame):
    spk_instances = []
    for i, row in features_df.iterrows():
        suite2p_path = row['suite2p_path']
        roi_idx = int(row['roi_index'])
        spike_prob_idx = int(row['spike_prob_index'])
        raw_f_idx = int(row['raw_f_index'])
        raw_f = np.load(suite2p_path + '/F.npy')[roi_idx]
        spike_prob = np.load(suite2p_path + '/cascade_spike_prob.npy')[roi_idx]
        features = row["raw_features"]
        # Create Spike instance
        spike = Spike(spike_prob_idx, spike_prob[spike_prob_idx], raw_f_idx, raw_f[raw_f_idx])
        spike._set_roi_index(roi_idx)
        spike.features = features
        spk_instances.append(spike)
    features_df['spk_instance'] = spk_instances
    return features_df

def label_spikes(sampled: pd.DataFrame):
    labels = []
    for i, row in sampled.iterrows():
        spike_dict = row["spk_instance"]
        spike : Spike = reinstantiate_objects(Spike, spike_dict)
        suite2p_path = Path(row['suite2p_path'])
        raw_f = np.load(suite2p_path / 'F.npy')[spike.roi_index]
        spike_prob = np.load(suite2p_path / 'cascade_spike_prob.npy')[spike.roi_index]

        def callback(i):
            labels.extend(i)
        SpikeLabelerApp(raw_f, spike_prob, [spike.idx_prob], [spike.idx_raw],
                        callback, neuron_id=spike.roi_index)
    return labels[:len(sampled)]

def merge_annotations(features_df: pd.DataFrame, sampled: pd.DataFrame):
    merge_key = "spike_key"
    features_df.set_index(merge_key, inplace=True)
    sampled.set_index(merge_key, inplace=True)  
    features_df.update(sampled[['label']])
    features_df.reset_index(inplace=True)
    return features_df

def main_annotate(features_df: pd.DataFrame, n_annotations : int):

    features_df["spk_instance"] = features_df["spk_instance"].apply(lambda x: Spike.__new__(Spike).__dict__.update(x) if isinstance(x, dict) else x)
    sampled = features_df.sample(n=min(n_annotations, len(features_df)), random_state=42).reset_index(drop=True)
    labels = label_spikes(sampled)
    sampled['label'] = labels
    return sampled


    