from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pipeline.reports import SpikeReport
from utils.inference import prepare_features

from spike_processing.detector import SpikeDetector
from spike_processing.filter import SpikeFilter
from spike_processing.factory import SpikeFactory
from spike_processing.kinetics import SpikeKinetics
from spike_processing.summary import NeuronSpikeSummary

if TYPE_CHECKING:
    from data_classes.video import Video


@dataclass
class SpikeService:
    n_jobs: int = -1

    detector: SpikeDetector = field(default_factory=SpikeDetector)
    spike_filter: SpikeFilter = field(default_factory=SpikeFilter)
    factory: SpikeFactory = field(default_factory=SpikeFactory)
    summarizer: NeuronSpikeSummary = field(default_factory=NeuronSpikeSummary)

    def extract_spike_features(self, video: "Video", dist: int = 20) -> pd.DataFrame:
        """
        Populates on each neuron:
          - spk_features (list[dict])
          - peaks (np.ndarray)
          - n_peaks_raw (int)
        Returns flattened feature dataframe for inference.
        """
        f = video.norm_sm_f
        all_n = video.neurons
        results = Parallel(
                    n_jobs=self.n_jobs)(delayed(
                    self.detector.get_feats)(f[n.index, :],
                                             int(n.index),
                                            dist=dist)
                                                      for n in all_n)

        spike_features_flat: list[dict] = []
        for neuron, (feats_list, peaks) in zip(video.neurons, results):
            neuron.spk_features = list(feats_list or [])
            neuron.peaks = np.asarray(peaks, dtype=int)
            neuron.n_peaks_raw = int(len(neuron.peaks))
            spike_features_flat.extend(neuron.spk_features)

        return pd.DataFrame(spike_features_flat)

    def filter_spikes(
        self,
        video: "Video",
        spk_feats_df: pd.DataFrame,
        spike_model: Any,
        model_config: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Populates on each neuron:
          - peaks_filtered (list[int])
        Filters out neurons with no kept peaks and reindexes filtered_index.
        Returns spike_mask (bool array aligned with spk_feats_df rows).
        """
        if spike_model is None:
            raise RuntimeError("Spike classifier model is not provided.")

        if spk_feats_df.empty:
            video.neurons = []
            return np.asarray([], dtype=bool)

        # Use shared helper from roi_service
        transform = model_config.get("transform") if model_config else None
        X = prepare_features(spk_feats_df, spike_model, transform)

        if X.shape[0] == 0:
            video.neurons = []
            return np.asarray([], dtype=bool)

        spike_mask = spike_model.predict(X).astype(bool)

        prev_idx = 0
        kept_neurons = []
        for neuron in video.neurons:
            n_spikes = len(getattr(neuron, "spk_features", []))
            spike_preds = spike_mask[prev_idx : prev_idx + n_spikes]
            prev_idx += n_spikes

            neuron.peaks_filtered = self.spike_filter.apply(neuron.peaks, spike_preds)

            if len(neuron.peaks_filtered) > 0:
                kept_neurons.append(neuron)

        video.neurons = kept_neurons
        for i, n in enumerate(video.neurons):
            n.filtered_index = i

        return spike_mask

    def compute_spike_statistics(self, video: "Video") -> pd.DataFrame:
        """
        Populates on each neuron:
          - spikes (list[Spike])
          - all_spk_stats (list[dict])
          - summary_stats (dict)
        Populates on video:
          - summary_df (pd.DataFrame)
        """
        kinetics = SpikeKinetics(fs=float(getattr(video, "fs", 30.0)))

        inst = Parallel(n_jobs=self.n_jobs)(
            delayed(self.factory.instantiate_spikes)(
                sm_norm_f=video.norm_sm_f[n.index, :],
                sg_norm_f=video.norm_sg_f[n.index, :],
                peaks_filtered=getattr(n, "peaks_filtered", []),
            )
            for n in video.neurons
        )

        for neuron, spikes in zip(video.neurons, inst):
            neuron.spikes = list(spikes or [])
            neuron.all_spk_stats = []

            for sp in neuron.spikes:
                if sp.f_small_window_sg is None:
                    continue
                sp.stats = kinetics.compute(sp.f_small_window_sg)
                neuron.all_spk_stats.append(sp.stats)

        per_neuron = {
            n.index: self.summarizer.summarize(
                n,
                f_trace_raw=video.suite2p_data["F"][n.index],
            )
            for n in video.neurons
        }
        video.summary_df = pd.DataFrame.from_dict(per_neuron, orient="index")
        return video.summary_df

    def _aggregate_summary_means(self, summary_df: pd.DataFrame) -> dict[str, float]:
        """Aggregate mean metrics from summary DataFrame."""
        if summary_df.empty:
            return {}

        numeric_cols = summary_df.select_dtypes(include=["number"]).columns

        means: dict[str, float] = {}
        for col in numeric_cols:
            value = summary_df[col].mean()
            if pd.notna(value):
                means[f"mean_{col}"] = float(value)

        return means

    def run(self, video: "Video", spike_model: Any, model_config: Optional[dict] = None) -> SpikeReport:
        """
        Populates on video:
          - neurons updated after filtering
          - summary_df
        Returns spike/neuron counts for narration.
        """
        n_neurons_in = len(video.neurons)

        spk_feats_df = self.extract_spike_features(video)
        n_spikes_raw = int(sum(getattr(n, "n_peaks_raw", 0) for n in video.neurons))

        self.filter_spikes(video, spk_feats_df, spike_model, model_config=model_config)
        n_spikes_kept = int(sum(len(getattr(n, "peaks_filtered", [])) for n in video.neurons))

        summary_df = self.compute_spike_statistics(video)
        mean_metrics = self._aggregate_summary_means(summary_df)

        return SpikeReport(
            n_neurons_in=n_neurons_in,
            n_neurons_out=len(video.neurons),
            n_spikes_raw=n_spikes_raw,
            n_spikes_kept=n_spikes_kept,
            mean_metrics=mean_metrics,
        )