"""Video class for managing individual recording sessions."""
from __future__ import annotations
from logging import config
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
import pandas as pd
from data_classes.neuron_group import NeuronGroup
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_classes.neuron import Neuron
from data_classes.roi import ROI
from pipeline.neuron_grouping import group_neurons_by_dtw, group_neurons_by_sttc, compare_groupings
from pipeline.preprocessing import load_suite2p_data
from roi_classifier.prepare_data import normalize_minmax
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import json
if TYPE_CHECKING:
    from .timepoint import Timepoint
from joblib import Parallel, delayed

def get_savgol_params(fs, sensor_type='gcamp8s'):
    """Get Savitzky-Golay parameters based on sampling frequency and sensor."""
    if sensor_type == 'gcamp8s':
        # Target ~500-800ms window for GCaMP8s (slower kinetics)
        window_frames = int(0.6 * fs)  # 600ms window
    elif sensor_type == 'gcamp6f':
        # Faster sensor, shorter window
        window_frames = int(0.3 * fs)  # 300ms window
    else:
        # Default/GCaMP6s
        window_frames = int(0.4 * fs)  # 400ms window
    
    # Ensure odd number
    window_length = 2 * (window_frames // 2) + 1
    # For GCaMP8s, use larger minimum
    window_length = max(9, window_length)  # Minimum 9 for slow sensors
    
    polyorder = 3  # Cubic fit better for slow, smooth transients
    
    return window_length, polyorder

class Video:
    """Represents a single video recording session."""
    
    def __init__(self, path: Path, suite2p_path: Path, timepoint: Optional[Timepoint] = None):
        """
        Initialize Video.
        
        Parameters:
            suite2p_path: Directory containing Suite2p output
            timepoint: Parent timepoint object
        """
        self.path = Path(path)
        self.video_id = path.name
        self.timepoint = timepoint
        self.suite2p_path = Path(suite2p_path)
        self.suite2p_data = load_suite2p_data(suite2p_path)
        # === Processed data ===
        self.norm_f = np.ndarray([])
        self.norm_sm_f = np.ndarray([])
        self.norm_sg_f = np.ndarray([])
        self.sm_sp = np.ndarray([])
        # === Metadata ===
        self.n_rois, self.n_frames = self.suite2p_data['F'].shape
        self._parse_metadata()
        
        # Will be populated by pipeline
        self.neurons : List[Neuron] = []
        # === Grouping results ===
        self.sttc_matrix = np.ndarray([])
        self.dtw_matrix = np.ndarray([])
        self.sttc_groups : List[NeuronGroup] = []
        self.dtw_groups : List[NeuronGroup] = []
        self.grouping_stats = pd.DataFrame()
        self.summary_df = None
    
    def process_fluorescence_traces(self) -> None:
        self.norm_f = normalize_minmax(self.suite2p_data['F'], self.suite2p_path / 'F_minmax.npy')
        # Smooth each ROI along the time axis (axis=1). Each row is an ROI.
        self.norm_sm_f = gaussian_filter1d(self.norm_f, sigma=4.0, axis=1 )

        window_length, polyorder = get_savgol_params(self.suite2p_data['fs'], sensor_type='gcamp8s')
        self.norm_sg_f = savgol_filter(self.norm_f, window_length=window_length, polyorder=polyorder, axis=1)
        cascade_prob = np.load(self.suite2p_path / 'cascade_spike_prob.npy')
        # Smooth spike-probabilities per-ROI along time as well
        self.sm_sp = gaussian_filter1d(cascade_prob, sigma=4.0, axis=1 )
        return cascade_prob, self.norm_sm_f, self.norm_sg_f, self.sm_sp

    def filter_rois(self, all_rois: List[ROI], roi_classifier : RandomForestClassifier) -> tuple[List[ROI], List[ROI], np.ndarray]:
        """Extract features and filter ROIs using the classifier."""
        all_feats = Parallel(n_jobs=-1)(
            delayed(roi.extract_features)(self.norm_sm_f[i, :], self.sm_sp[i, :])
            for i, roi in enumerate(all_rois)
        )

        print(f"Extracted features for {len(all_rois)} ROIs.")
        feats_df = pd.DataFrame(all_feats)

        if roi_classifier is None:
            preds = np.ones(len(all_rois), dtype=bool)
        else:
            # Determine expected order of features. Prefer model attribute, fallback to config file
            expected = None
            if hasattr(roi_classifier, 'feature_names_in_'):
                try:
                    expected = list(roi_classifier.feature_names_in_)
                except Exception:
                    expected = None
            if expected is None:
                cfg_path = Path('roi_classifier/models/roi_classifier_config.json')
                if cfg_path.exists():
                    try:
                        cfg = json.load(open(cfg_path))
                        expected = cfg.get('feature_names')
                    except Exception:
                        expected = None

            # Ensure DataFrame contains expected columns (fill missing with NaN)
            if expected:
                for col in expected:
                    if col not in feats_df.columns:
                        feats_df[col] = np.nan
                X = feats_df[expected].values
            else:
                X = feats_df.values

            preds = roi_classifier.predict(X)
        print(f"Total passed: {np.sum(preds)} / {len(all_rois)}")
        good_roi_mask = np.asarray(preds).astype(bool)


        for roi, pred, i in zip(all_rois, good_roi_mask, range(len(all_rois))):
            if roi.is_good is False:
                # preserve an explicitly marked-bad ROI
                good_roi_mask[i] = False
                continue
            roi.is_good = bool(pred)
            good_roi_mask[i] = roi.is_good

        good_roi_mask = np.asarray(good_roi_mask, dtype=bool)

        good_rois = [roi for roi in all_rois if roi.is_good]
        bad_rois = [roi for roi in all_rois if not roi.is_good]
        return good_rois, bad_rois, good_roi_mask
    def get_all_spike_features(self, spk_model : RandomForestClassifier) -> pd.DataFrame:

        spike_features_list = Parallel(n_jobs=-1)(
                    delayed(neuron.get_spike_features)(self.norm_sm_f[neuron.index, :])
                    for neuron in self.neurons)
        spike_features_flat = []
        for neuron, res in zip(self.neurons, spike_features_list):

            if res is None:
                feats_list = []
                peaks = np.array([], dtype=int)
            else:
                try:
                    feats_list, peaks = res
                except Exception:
                    feats_list = res
                    peaks = np.array([], dtype=int)

            neuron.spk_features = [] if feats_list is None else list(feats_list)
            neuron.peaks = np.asarray(peaks)
            neuron.n_peaks_raw = int(len(neuron.peaks))
            for feat_dict in (feats_list or []):
                spike_features_flat.append(feat_dict)

        spk_feats_df = pd.DataFrame(spike_features_flat)
        if spk_model is None or spk_feats_df.empty:
            return spk_feats_df, np.array([], dtype=bool)

        expected = None
        cfg_path = Path('spike_classifier/models/spike_classifier_config.json')
        if cfg_path.exists():
            try:
                cfg = json.load(open(cfg_path))
                if cfg.get('use_top_features') and cfg.get('selected_features'):
                    expected = cfg.get('selected_features')
                else:
                    expected = cfg.get('feature_names')
            except Exception:
                expected = None

        if expected is None:
            expected = getattr(spk_model, 'feature_names_in_', None)

        if expected:
            for col in expected:
                if col not in spk_feats_df.columns:
                    spk_feats_df[col] = np.nan
            Xdf : pd.DataFrame= spk_feats_df[expected].copy()    
            X = Xdf.values
        else:
            X = spk_feats_df.values

        spike_mask = spk_model.predict(X)

        # Diagnostic: per-neuron extracted / passed counts
        prev = 0
        per_neuron_summary = []
        for neuron in self.neurons:
            n_sp = len(neuron.spk_features) if getattr(neuron, 'spk_features', None) is not None else 0
            passed = 0
            if n_sp > 0:
                seg = spike_mask[prev: prev + n_sp]
                try:
                    passed = int(np.asarray(seg).sum())
                except Exception:
                    passed = int(sum(1 for v in seg if bool(v)))
            per_neuron_summary.append((neuron.index, n_sp, passed))
            prev += n_sp
        return spk_feats_df, spike_mask
    
    def filter_all_spikes(self, spike_mask: np.ndarray) -> list[Neuron]:
        prev_idx = 0
        for neuron in self.neurons:
            n_spikes_extracted = len(neuron.spk_features)
            spike_preds = spike_mask[prev_idx: prev_idx + n_spikes_extracted]
            prev_idx += n_spikes_extracted
            neuron.peaks_filtered = neuron.filter_spikes(spike_preds)
        neurons_with_spikes = [n for n in self.neurons if len(n.peaks_filtered) > 0]
        self.neurons = neurons_with_spikes
        for i, n in enumerate(neurons_with_spikes):
            n.filtered_index = i
        return neurons_with_spikes
    
    def get_spike_statistics(self) -> None:
        # Run instantiation in parallel and assign returned Spike objects back
        inst_spikes = Parallel(n_jobs=-1)(
            delayed(n.instantiate_spikes)(self.norm_sm_f[n.index, :], self.norm_sg_f[n.index, :]) for n in self.neurons
        )

        # Joblib workers don't mutate main-process objects; assign spikes and stats back
        for neuron, result in zip(self.neurons, inst_spikes):
            if result is None:
                neuron.spikes = []
                neuron.all_spk_stats = []
            else:
                try:
                    spikes, all_stats = result
                    neuron.spikes = [] if spikes is None else list(spikes)
                    neuron.all_spk_stats = [] if all_stats is None else list(all_stats)
                except (TypeError, ValueError):
                    # Backward compat: if result is just spikes list
                    neuron.spikes = [] if result is None else list(result)
                    neuron.all_spk_stats = []

        per_neuron_spike_summaries = {
            n.index: n.summarize_spike_statistics(self.suite2p_data['F'][n.index]) for n in self.neurons
        }
        return per_neuron_spike_summaries
    
    def get_group_summary(self, time_window, distance_threshold) -> pd.DataFrame:
        """
        Docstring for get_group_summary
        
        Args:
            config: dict specifying grouping parameters
        Returns:
            grouping_stats: Dict summarizing grouping results
             'n_sttc_groups': len(sttc_groups),
            'n_dtw_groups': len(dtw_groups),
            'agreement': agreement,
            'combined_stats': combined_stats
    }
        """

        self.sttc_groups, self.sttc_matrix = group_neurons_by_sttc(
            self.neurons, self.n_frames, time_window = time_window, distance_threshold=distance_threshold
        )
        #self.dtw_groups, self.dtw_matrix = [], np.ndarray([]) #group_neurons_by_dtw(
            #self.neurons, **dtw_config
        #)
        grouping_stats = compare_groupings(self.sttc_groups, self.dtw_groups, self.sttc_matrix, self.dtw_matrix, self.neurons)
        return grouping_stats, self.sttc_groups
    def _parse_metadata(self):
        """
        Parse metadata from directory structure.
        Expected: ex337/treatment/timepoint/video/suite2p/plane0/
        """
        # Get parent directories
        parts = self.path.parts
        
        # video is current folder name
        self.video_id = self.path.name
        
        # Go up: video -> timepoint -> treatment -> experiment
        if len(parts) >= 4:
            # parts[-1] = plane0, parts[-2] = suite2p, parts[-3] = video
            # parts[-4] = timepoint, parts[-5] = treatment, parts[-6] = experiment
            try:
                # If path includes suite2p/plane0, adjust indices
                if 'suite2p' in parts:
                    suite2p_idx = parts.index('suite2p')
                    # suite2p-1 = video, suite2p-2 = timepoint, suite2p-3 = treatment
                    if suite2p_idx >= 3:
                        self.timepoint_name = parts[suite2p_idx - 2]
                        self.treatment = parts[suite2p_idx - 3]
                        if suite2p_idx >= 4:
                            self.experiment_name = parts[suite2p_idx - 4]
                        else:
                            self.experiment_name = 'unknown'
                    else:
                        self.timepoint_name = 'unknown'
                        self.treatment = 'unknown'
                        self.experiment_name = 'unknown'
                else:
                    # Direct video path without suite2p
                    self.timepoint_name = parts[-2] if len(parts) >= 2 else 'unknown'
                    self.treatment = parts[-3] if len(parts) >= 3 else 'unknown'
                    self.experiment_name = parts[-4] if len(parts) >= 4 else 'unknown'
            except (IndexError, ValueError):
                self.timepoint_name = 'unknown'
                self.treatment = 'unknown'
                self.experiment_name = 'unknown'
        else:
            self.timepoint_name = 'unknown'
            self.treatment = 'unknown'
            self.experiment_name = 'unknown'
        
    def get_group_summary_table(self) -> pd.DataFrame:
        """Summarize grouping results into a table of rows for reporting."""
        rows = []
        
        # STTC groups
        for i, group in enumerate(self.sttc_groups):
            for neuron in group:
                rows.append({
                    'neuron_id': neuron.row_index,
                    'group_method': 'sttc',
                    'group_id': i,
                    'group_size': len(group)
                })
        
        # DTW groups
        for i, group in enumerate(self.dtw_groups):
            for neuron in group:
                rows.append({
                    'neuron_id': neuron.row_index,
                    'group_method': 'dtw',
                    'group_id': i,
                    'group_size': len(group)
                })
        
        return pd.DataFrame(rows)
    
    def __repr__(self):
        return f"Video(id={self.video_id}, neurons={len(self.neurons)})"