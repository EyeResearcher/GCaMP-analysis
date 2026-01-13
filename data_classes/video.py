"""Video class for managing individual recording sessions."""
from __future__ import annotations
from dataclasses import dataclass
from logging import config
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING
import pandas as pd
from data_classes.neuron_group import NeuronGroup
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_classes.neuron import Neuron
from data_classes.roi import ROI
from pipeline.neuron_grouping import group_neurons_by_dtw, group_neurons_by_sttc, compare_groupings
from utils.io_utils import load_suite2p_data
from roi_classifier.prepare_data import normalize_minmax
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from utils.visualization import visualize_neuron_groups
import matplotlib.pyplot as plt
import json
from matplotlib.figure import Figure
if TYPE_CHECKING:
    from .timepoint import Timepoint
from joblib import Parallel, delayed

def get_savgol_params(fs, sensor_type='gcamp8s'):
    """Get Savitzky-Golay parameters based on sampling frequency and sensor.
    Args:
        fs: Sampling frequency in Hz
        sensor_type: Type of calcium sensor ('gcamp6f', 'gcamp6s', 'gcamp8s', etc.):
            GCAMP8s assumed by default, 600ms window.
            GCAMP6f uses shorter window due to faster kinetics, 300ms.
            GCAMP6s uses intermediate window, 400ms.

    Returns:
        window_length: Length of the filter window (odd integer)"""
    if sensor_type == 'gcamp8s':
        window_frames = int(0.6 * fs) 
    elif sensor_type == 'gcamp6f':
        window_frames = int(0.3 * fs)
    else:
        window_frames = int(0.4 * fs)  

    window_length = 2 * (window_frames // 2) + 1
    window_length = max(9, window_length) 
    
    polyorder = 3
    
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
        self.bad_rois : List[ROI] = []
        self.neurons : List[Neuron] = []
        self.n_good_rois = 0
        self.n_bad_rois = 0
        # === Grouping results ===
        
        self.sttc_fig = Figure()
        self.dtw_fig = Figure()
        self.sttc_matrix = np.ndarray([])
        self.dtw_matrix = np.ndarray([])
        self.sttc_groups : List[NeuronGroup] = []
        self.dtw_groups : List[NeuronGroup] = []
        self.agreement = 0.0
        self.grouping_stats = pd.DataFrame()
        self.summary_df = pd.DataFrame()
        self.bad_rois_features = pd.DataFrame()

    
    def process_fluorescence_traces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Normalize and smooth fluorescence traces.
            Args:
                self: Video object
            Returns:
                cascade_prob: Loaded cascade spike probability traces
                norm_sm_f: Smoothed min-max normalized fluorescence traces
                norm_sg_f: Savitzky-Golay filtered min-max normalized fluorescence traces
                sm_sp: Smoothed cascade spike probability traces
        """
        self.norm_f = normalize_minmax(self.suite2p_data['F'], self.suite2p_path / 'F_minmax.npy')
        self.norm_sm_f = gaussian_filter1d(self.norm_f, sigma=4.0, axis=1 )

        window_length, polyorder = get_savgol_params(self.suite2p_data['fs'], sensor_type='gcamp8s')
        self.norm_sg_f = savgol_filter(self.norm_f, window_length=window_length, polyorder=polyorder, axis=1)

        return self.norm_sm_f, self.norm_sg_f
    def _create_roi_objects(self) -> List[ROI]:
        """Create ROI objects from Suite2p data."""
        rois = []
        for i in range(self.n_rois):
            roi = ROI(index=i, 
                      f_trace=self.suite2p_data['F'][i, :],
                      stats=self.suite2p_data['stat'][i] if 'stat' in self.suite2p_data else None,
                      fneu = self.suite2p_data['Fneu'][i] if 'Fneu' in self.suite2p_data else None)
            rois.append(roi)
        return rois
    
    def _create_neuron_objects(self, good_rois: List[ROI]) -> List[Neuron]:
        """Create Neuron objects from good ROIs."""

        for i, roi in enumerate(good_rois):
            neuron = Neuron(roi_instance=roi,
                            filtered_index=i,
                            fs = self.suite2p_data['fs'])
            self.neurons.append(neuron)


    def filter_rois(self, all_rois: List[ROI],
                    roi_classifier : RandomForestClassifier) -> tuple[List[ROI], np.ndarray]:
        """Extract features and filter ROIs using the classifier."""


        all_feats = Parallel(n_jobs=-1)(
            delayed(roi.extract_features)(self.norm_sm_f[i, :])
            for i, roi in enumerate(all_rois)
        )
        for roi, feats in zip(all_rois, all_feats):
            roi.features = feats
        feats_df = pd.DataFrame(all_feats)

        if roi_classifier is None:
            raise RuntimeError("ROI classifier model is not provided.")
        else:
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
            if expected:
                for col in expected:
                    if col not in feats_df.columns:
                        feats_df[col] = np.nan
                X = feats_df[expected].values
            else:
                X = feats_df.values

            preds = roi_classifier.predict(X)
            
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
        self.bad_rois = [roi for roi in all_rois if not roi.is_good]
        self.n_good_rois = len(good_rois)
        self.n_bad_rois = len(self.bad_rois)
        return good_rois, good_roi_mask
    
    def get_bad_rois_features_df(self) -> pd.DataFrame:
        """
        Create a DataFrame of features from bad ROIs.
        
        Returns:
            DataFrame indexed by ROI index with feature columns.
            Returns empty DataFrame if no bad ROIs exist.
        """
        if not self.bad_rois:
            return pd.DataFrame()
        
        features_list = []
        indices = []
        for roi in self.bad_rois:
            features_list.append(roi.features)
            indices.append(roi.index)
        
        if not features_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(features_list, index=indices)
        df.index.name = 'roi_index'
        self.bad_rois_features = df
        return df
    def _extract_spike_features_parallel(self) -> list[dict]:
        """
        Extract spike features from all neurons in parallel.
        
        Returns:
            List of feature dictionaries for all spikes across all neurons.
        """
        spike_features_list = Parallel(n_jobs=-1)(
            delayed(neuron.get_spike_features)(self.norm_sm_f[neuron.index, :])
            for neuron in self.neurons
        )
        
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
        return spk_feats_df
    
    def _prepare_spike_features_for_model(
        self, 
        spk_feats_df: pd.DataFrame, 
        spk_model: RandomForestClassifier
    ) -> np.ndarray:
        """
        Prepare spike features matrix for model prediction.
        
        Loads config to get expected feature names and ensures all required
        columns exist in the dataframe.
        
        Args:
            spk_feats_df: DataFrame with spike features
            spk_model: Trained spike classifier model
            
        Returns:
            Feature matrix X ready for prediction
        """
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
            X = spk_feats_df[expected].copy().values
        else:
            X = spk_feats_df.values
        
        return X
    
    def _get_mask(self, x: np.ndarray, spike_model: RandomForestClassifier) -> np.ndarray:
        """
        Predict spike labels using the spike classifier model.
        
        Args:
            x: Feature matrix for spikes
            spike_model: Trained spike classifier model
            
        Returns:
            Boolean mask indicating predicted good spikes
        """
        if spike_model is None:
            raise RuntimeError("Spike classifier model is not provided.")
        if x.shape[0] == 0:
            raise Warning(f"No spike features available for prediction for video {self.path}")
        spike_mask = spike_model.predict(x)
        return spike_mask
    
    
    def filter_all_spikes(self, spike_feats_df, spike_model) -> tuple[list[Neuron], np.ndarray]:
        """
        This method prepares spike features and filters spikes for all neurons
        using the provided spike classifier model. 
        Args:
            spike_feats_df: DataFrame containing spike features for all neurons
            spike_model: Trained spike classifier model
        Returns:
            Tuple containing:
                - List of Neuron objects with filtered spikes
                - Numpy array mask indicating predicted good spikes
        """
        x = self._prepare_spike_features_for_model(spike_feats_df, spike_model)
        spike_mask = self._get_mask(x, spike_model)

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
        return neurons_with_spikes, spike_mask
    
    def get_spike_statistics(self) -> dict[int, dict]:
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
        self.summary_df = pd.DataFrame.from_dict(per_neuron_spike_summaries, orient='index')
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
        grouping_summary = compare_groupings(self.sttc_groups, self.dtw_groups, self.sttc_matrix, self.dtw_matrix, self.neurons)
        self.sttc_groups = grouping_summary['sttc_groups']
        self.dtw_groups = grouping_summary['dtw_groups']
        self.n_sttc_groups = len(self.sttc_groups)
        self.n_dtw_groups = len(self.dtw_groups)
        self.agreement = grouping_summary['agreement']
        self.grouping_stats = pd.DataFrame([grouping_summary['combined_stats']])

        return grouping_summary, self.sttc_groups
    
    def _visualize_neuron_groups(self, config_label: Optional[str] = None) -> tuple:
        """
        Visualize neuron groups and save the figure.
        
        Args:
            config_label: Optional label for the configuration (e.g., "tw0.033_dt0.3")
        
        Returns:
            fig: Matplotlib figure object
            output_path: Path to saved image
        """
        groups = []
        if "sttc" in config_label.lower():
            groups = self.sttc_groups
        elif "dtw" in config_label.lower():
            groups = self.dtw_groups

        if len(groups) == 0:
            return None, None
        
        img_size = self.suite2p_data.get('ops', {}).get('Ly', 1024), self.suite2p_data.get('ops', {}).get('Lx', 1024)
        fig, output_path = visualize_neuron_groups(
            neuron_groups=groups,
            stat=self.suite2p_data['stat'] if 'stat' in self.suite2p_data else np.array([]),
            img_size=img_size,
            video_path=self.path,
            config_label=config_label
        )
        if 'sttc' in config_label.lower():
            self.sttc_fig = fig
        elif 'dtw' in config_label.lower():
            self.dtw_fig = fig
        return fig, output_path
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
     
    
    def __repr__(self):
        return f"Video(id={self.video_id}, neurons={len(self.neurons)})"
    

@dataclass(frozen=True)
class VideoStatistics:
    """Pure in-memory container for per-video outputs."""
    video_name: str

    per_neuron_spike_summaries: pd.DataFrame
    grouping_stats: pd.DataFrame
    bad_rois_features: pd.DataFrame

    sttc_matrix: np.ndarray
    dtw_matrix: np.ndarray

    sttc_fig: Optional[Figure] = None 
    dtw_fig: Optional[Figure] = None

    @classmethod
    def from_video(cls, video : "Video") -> "VideoStatistics":
        """Convenience constructor; keeps Video dependency out of __init__."""
        return cls(
            video_name=video.path.name,
            per_neuron_spike_summaries=video.summary_df,
            grouping_stats=video.grouping_stats,
            bad_rois_features=video.bad_rois_features,
            sttc_matrix=video.sttc_matrix,
            dtw_matrix=video.dtw_matrix,
            sttc_fig=getattr(video, "sttc_fig", None),
            dtw_fig=getattr(video, "dtw_fig", None))
    
@dataclass
class VideoStatisticsWriter:
    """
    Responsible for writing VideoStatistics to disk.

    Directory scheme:
      <output_root>/<video_name>/metrics/...
    or if you pass output_dir directly, it writes into that folder.
    """
    save_fig_dpi: int = 300
    save_fig_bbox_inches: str = "tight"

    def metrics_dir(self, output_root: Path, video_name: str) -> Path:
        # You can change this scheme later without touching VideoStatistics
        return output_root / video_name / "metrics"

    def write(self, stats: "VideoStatistics", output_root: Path) -> dict[str, str]:
        out_dir = self.metrics_dir(output_root, stats.video_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        base = stats.video_name
        manifest: dict[str, str] = {}

        # Tables
        excel_path = out_dir / f"{base}_metrics.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            if not stats.per_neuron_spike_summaries.empty:
                stats.per_neuron_spike_summaries.to_excel(
                    writer, sheet_name='spike_summary', index=False
                )
            
            if not stats.grouping_stats.empty:
                stats.grouping_stats.to_excel(
                    writer, sheet_name='grouping_stats', index=False
                )
            
            if not stats.bad_rois_features.empty:
                stats.bad_rois_features.to_excel(
                    writer, sheet_name='bad_rois_features', index=True
                )
        
        manifest["metrics_excel"] = str(excel_path)
        # Matrices
        sttc_npy = out_dir / f"{base}_sttc_matrix.npy"
        np.save(sttc_npy, stats.sttc_matrix)
        manifest["sttc_matrix_npy"] = str(sttc_npy)

        dtw_npy = out_dir / f"{base}_dtw_matrix.npy"
        np.save(dtw_npy, stats.dtw_matrix)
        manifest["dtw_matrix_npy"] = str(dtw_npy)

        # Figures (optional)
        if stats.sttc_fig is not None:
            sttc_png = out_dir / f"{base}_sttc_groups.png"
            stats.sttc_fig.savefig(sttc_png, dpi=self.save_fig_dpi, bbox_inches=self.save_fig_bbox_inches)
            manifest["sttc_fig_png"] = str(sttc_png)

        if stats.dtw_fig is not None:
            dtw_png = out_dir / f"{base}_dtw_groups.png"
            stats.dtw_fig.savefig(dtw_png, dpi=self.save_fig_dpi, bbox_inches=self.save_fig_bbox_inches)
            manifest["dtw_fig_png"] = str(dtw_png)

        

        return manifest