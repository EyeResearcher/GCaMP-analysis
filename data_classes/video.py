from pathlib import Path
from data_classes.roi import ROI
from data_classes.neuron import Neuron
from data_classes.neuron_group import NeuronGroup
import pandas as pd  
import os 
import tifffile as tiff
from analysis.grouping import main_grouping
from analysis.correlation import compute_sttc_matrix
from utils.io_utils import SummaryFiles, save_video_metrics
from Cascade.cascade2p.cascade_wrapper import CascadePredictor

class Video:

    NEURON_SUMMARY_COLUMNS = [
        'num_spikes', 'spike_indices', 'fluorescence_values',
        'spike_prob_values','average_spike_amplitude', 'spike_frequency', 'tau_constants', 'rise_constants',
        'auc', 'auc_per_spike'
    ]

    def __init__(self, folder: Path, roi_model, cascade_model: CascadePredictor, fs=30):
        self.path = folder
        self.video_id = self.path.name
        self.timepoint = folder.parts[-2]
        self.experiment = folder.parts [-3]
        self.region, self.treatment = self._parse_region_and_treatment()
        self.fs = fs
        self.dimensions = self._get_tiff_dimensions()
        self.summary_files = SummaryFiles(self.path, cascade_model)
        self.summary_files._create_spike_prob(new_model=False)  # Ensure spike probabilities are created
        # Instantiate and filter ROIs
        n_rois = self.summary_files.raw_fluorescence.shape[0]
        self.rois = [ROI(i, self.summary_files, roi_model) for i in range(n_rois)]
        self.good_rois = []

        # Placeholders
        self.filtered_summary = {}
        self.neurons = []
        self.spike_train_list = []
        self.sttc_matrix = None
        self.neuron_groups = []
        self.group_distances = {}
        self.avg_group_sttc = []
        self.summary_df = pd.DataFrame()
        self.good_indices = []

    def _parse_region_and_treatment(self):
        parts = self.video_id.split('_', 1)
        region = parts[0]
        treatment = parts[1] if len(parts) > 1 else 'Baseline'
        return region, treatment

    def _process_neurons(self):
        # 1) Filter ROIs
        for roi in self.rois:
            roi._filter_roi()
            if roi.is_good_cell:
                self.good_rois.append((roi.row_index, roi.features))
        self.good_indices = [self.good_rois[i][0] for i in range(len(self.good_rois))]
        # 2) Filter summary files
        keys = ['raw_fluorescence', 'Fneu', 'spike_prob', 'spks', 'iscell', 'stat', 'ops']
        for key in keys:
            arr = getattr(self.summary_files, key)
            try:
                self.filtered_summary[key] = arr[self.good_indices]
            except Exception:
                self.filtered_summary[key] = arr

        # 3) Instantiate Neurons for good ROIs
        self.neurons = [Neuron(idx, features, self.summary_files, fs=self.fs)
                        for idx, features in self.good_rois]
    
        # 4) Compute per-neuron stats
        for neuron in self.neurons:
            neuron._compute_all_spike_stats()
            self.spike_train_list.append(neuron.binary_spike_train)

    def _process_whole_video(self):
        # 5) Compute STTC matrix
        self.sttc_matrix = compute_sttc_matrix(
            self.spike_train_list,
            self.filtered_summary["ops"]
        )

        # 6) Make neuron groups (indices refer to positions in filtered neuron list)
        idx_groups, self.group_distances, self.avg_group_sttc = \
            main_grouping(
                self.sttc_matrix,
                self.filtered_summary['stat'])
        # assign filtered_index to each Neuron instance
        for filt_idx, neuron in enumerate(self.neurons):
            neuron.filtered_index = filt_idx
        # map index groups (relative positions) to actual Neuron objects
        self.neuron_groups = [
            [self.neurons[i] for i in group] for group in idx_groups
        ]
    def _build_summary_dataframe(self):
        # 7) Build summary DataFrame
        records = []
        for neuron in self.neurons:
            records.append({
                'num_spikes':               neuron.num_spikes,
                'spike_indices':            neuron.f_peak_indices,
                'fluorescence_values':      neuron.f_peak_values,
                'spike_prob_values':        neuron.spike_prob_peak_values,
                'average_spike_amplitude':  neuron.f_average_amplitude,
                'spike_frequency':          neuron.spike_frequency,
                'tau_constants':            neuron.tau_stats,
                'rise_constants':           neuron.rise_constants_stats,
                'auc':                      neuron.area_under_curve,
                'auc_per_spike':            neuron.area_per_spike,
            })
        self.summary_df = pd.DataFrame.from_records(
            records,
            columns=self.NEURON_SUMMARY_COLUMNS,
            index=self.good_indices
        )
        return self.summary_df
    
    def _get_tiff_dimensions(self):
        for file in os.listdir(self.path):
            if file.lower().endswith(('.tif', '.tiff')):
                img = tiff.imread(str(self.path / file))
                if img.ndim == 3:
                    _, y, x = img.shape
                elif img.ndim == 2:
                    y, x = img.shape
                else:
                    raise ValueError(f"Unexpected TIFF shape: {img.shape}")
                return x, y
        raise FileNotFoundError("No .tif or .tiff file found in folder.")
    def video_main(self):
        self._parse_region_and_treatment()
        self._process_neurons()
        self._process_whole_video()
        self._build_summary_dataframe()
        save_video_metrics(self.path, self.filtered_summary, self.summary_df, self.experiment, self.timepoint, self.sttc_matrix)
