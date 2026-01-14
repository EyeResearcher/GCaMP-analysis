from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import TYPE_CHECKING

from pipeline.reports import SpikeReport
if TYPE_CHECKING:
    from data_classes.video import Video
@dataclass
class SpikeService:
    n_jobs: int = -1
    spike_config_path: Optional[Path] = Path("spike_classifier/models/spike_classifier_config.json")

    def extract_spike_features(self, video: "Video") -> pd.DataFrame:
        spike_features_list = Parallel(n_jobs=self.n_jobs)(
            delayed(neuron.get_spike_features)(video.norm_sm_f[neuron.index, :])
            for neuron in video.neurons
        )

        spike_features_flat = []
        for neuron, res in zip(video.neurons, spike_features_list):
            if res is None:
                feats_list, peaks = [], np.array([], dtype=int)
            else:
                try:
                    feats_list, peaks = res
                except Exception:
                    feats_list, peaks = res, np.array([], dtype=int)

            neuron.spk_features = list(feats_list or [])
            neuron.peaks = np.asarray(peaks)
            neuron.n_peaks_raw = int(len(neuron.peaks))

            for feat in (feats_list or []):
                spike_features_flat.append(feat)

        return pd.DataFrame(spike_features_flat)

    def _prepare_matrix(self, spk_feats_df: pd.DataFrame, model: Any) -> np.ndarray:
        expected = None

        if self.spike_config_path and self.spike_config_path.exists():
            try:
                cfg : dict = json.load(open(self.spike_config_path))
                if cfg.get("use_top_features") and cfg.get("selected_features"):
                    expected = cfg.get("selected_features")
                else:
                    expected = cfg.get("feature_names")
            except Exception:
                expected = None

        if expected is None:
            expected = getattr(model, "feature_names_in_", None)

        if expected:
            for col in expected:
                if col not in spk_feats_df.columns:
                    spk_feats_df[col] = np.nan
            return spk_feats_df[list(expected)].copy().values

        return spk_feats_df.values

    def filter_spikes(self, video: "Video", spk_feats_df: pd.DataFrame, spike_model: Any) -> np.ndarray:
        if spike_model is None:
            raise RuntimeError("Spike classifier model is not provided.")

        X = self._prepare_matrix(spk_feats_df, spike_model)
        if X.shape[0] == 0:
            return np.asarray([], dtype=bool)

        spike_mask = spike_model.predict(X).astype(bool)

        prev_idx = 0
        kept_neurons = []
        for neuron in video.neurons:
            n_spikes = len(neuron.spk_features)
            spike_preds = spike_mask[prev_idx: prev_idx + n_spikes]
            prev_idx += n_spikes
            neuron.peaks_filtered = neuron.filter_spikes(spike_preds)
            if len(neuron.peaks_filtered) > 0:
                kept_neurons.append(neuron)

        video.neurons = kept_neurons
        for i, n in enumerate(video.neurons):
            n.filtered_index = i

        return spike_mask

    def compute_spike_statistics(self, video: "Video") -> pd.DataFrame:
        inst = Parallel(n_jobs=self.n_jobs)(
            delayed(n.instantiate_spikes)(
                video.norm_sm_f[n.index, :],
                video.norm_sg_f[n.index, :],
            )
            for n in video.neurons
        )

        for neuron, result in zip(video.neurons, inst):
            if result is None:
                neuron.spikes = []
                neuron.all_spk_stats = []
            else:
                try:
                    spikes, all_stats = result
                    neuron.spikes = [] if spikes is None else list(spikes)
                    neuron.all_spk_stats = [] if all_stats is None else list(all_stats)
                except (TypeError, ValueError):
                    neuron.spikes = [] if result is None else list(result)
                    neuron.all_spk_stats = []

        per_neuron = {n.index: n.summarize_spike_statistics(video.suite2p_data["F"][n.index]) for n in video.neurons}
        video.summary_df = pd.DataFrame.from_dict(per_neuron, orient="index")
        return video.summary_df
    
    def _aggregate_summary_means(self, summary_df: pd.DataFrame) -> dict[str, float]:
        if summary_df.empty:
            return {}

        numeric_cols = summary_df.select_dtypes(include=["number"]).columns

        means = {}
        for col in numeric_cols:
            value = summary_df[col].mean()
            if pd.notna(value):
                means[f"mean_{col}"] = float(value)

        return means
    def run(self, video: "Video", spike_model: Any) -> SpikeReport:
        """
        Populates on video:
          - neurons updated after filtering
          - summary_df
        Returns spike/neuron counts for narration.
        """
        n_neurons_in = len(video.neurons)

        # Extract raw peak features and store on neurons
        spk_feats_df = self.extract_spike_features(video)

        # Count raw spikes before filtering (use what your Neuron objects already track)
        n_spikes_raw = sum(getattr(n, "n_peaks_raw", 0) for n in video.neurons)

        # Filter spikes + drop neurons with no spikes
        self.filter_spikes(video, spk_feats_df, spike_model)

        # Count kept spikes after filtering
        n_spikes_kept = sum(len(getattr(n, "peaks_filtered", [])) for n in video.neurons)

        # Compute per-neuron spike stats df (and attach spike objects)
        summary_df = self.compute_spike_statistics(video)

        # Aggregate means of numeric columns
        mean_metrics = self._aggregate_summary_means(summary_df)
        return SpikeReport(
            n_neurons_in=n_neurons_in,
            n_neurons_out=len(video.neurons),
            n_spikes_raw=int(n_spikes_raw),
            n_spikes_kept=int(n_spikes_kept),
            mean_metrics=mean_metrics
        )

