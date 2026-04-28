"""Spike filtering pipeline: classification filtering and orchestration.

Mirrors ``roi_processing.filtering`` — bundles the SpikeFilter step class
alongside the orchestrating SpikeService.

Step classes remain as separate dataclass units so they can be tested or swapped
independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from gcamp_analysis.reports import SpikeReport
from utils.inference import prepare_features

from .features import describe_spikes
from .kinetics import SpikeKinetics

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.video import Video


# =====================================================================
#  SPIKE FILTER
# =====================================================================


@dataclass
class SpikeFilter:
    """Convert per-peak predictions (bool/0-1) into a list of retained peak indices."""

    def apply(self, peaks: np.ndarray, predictions: np.ndarray) -> List[int]:
        peaks_arr = np.asarray(peaks, dtype=int).reshape(-1)
        preds_arr = np.asarray(predictions).reshape(-1)

        if peaks_arr.size == 0:
            return []

        if preds_arr.dtype != bool:
            preds_arr = preds_arr.astype(float) > 0.5
        else:
            preds_arr = preds_arr.astype(bool)

        # Length safety
        if preds_arr.size < peaks_arr.size:
            pad = np.zeros(peaks_arr.size - preds_arr.size, dtype=bool)
            preds_arr = np.concatenate([preds_arr, pad])
        elif preds_arr.size > peaks_arr.size:
            preds_arr = preds_arr[: peaks_arr.size]

        return [int(p) for p, keep in zip(peaks_arr.tolist(), preds_arr.tolist()) if bool(keep)]


# =====================================================================
#  SPIKE SERVICE (orchestrator)
# =====================================================================


@dataclass
class SpikeService:
    """Orchestrates the full spike sub-pipeline across a video's neurons."""

    n_jobs: int = -1

    spike_filter: SpikeFilter = field(default_factory=SpikeFilter)

    def extract_spike_features(self, video: "Video") -> pd.DataFrame:
        """
        Populates on each neuron:
          - spk_features (list[dict])
          - peaks (np.ndarray)
          - n_peaks_raw (int)
        Returns flattened feature dataframe for inference.

        In concatenated mode, spike detection is run separately on each
        parsed section so that section boundaries cannot produce
        artefacts. Section-local peak indices are shifted back into the
        full-trace coordinate system.
        """
        fs = float(video.fs)

        if video.is_concatenated and video.concat_sections:
            spike_features_flat: list[dict] = []
            section_results: dict[str, list[tuple]] = {}
            for section in video.concat_sections:
                smoothed = video.section_traces[section.section_key]["norm_sm_f"]
                section_results[section.section_key] = Parallel(n_jobs=self.n_jobs)(
                    delayed(describe_spikes)(smoothed[n.index, :], int(n.index), fs=fs)
                    for n in video.neurons
                )

            for neuron_idx, neuron in enumerate(video.neurons):
                merged_feats: list[dict] = []
                merged_peaks: list[np.ndarray] = []

                for section in video.concat_sections:
                    feats_list, _keys, peaks = section_results[section.section_key][neuron_idx]
                    feats = [dict(feature_dict) for feature_dict in (feats_list or [])]
                    shifted_peaks = np.asarray(peaks, dtype=int) + section.start_frame

                    for feature_dict in feats:
                        feature_dict["_section_key"] = section.section_key
                        feature_dict["_section_kind"] = section.section_kind

                    merged_feats.extend(feats)
                    if shifted_peaks.size:
                        merged_peaks.append(shifted_peaks)

                neuron.spk_features = merged_feats
                neuron.peaks = (
                    np.concatenate(merged_peaks) if merged_peaks else np.asarray([], dtype=int)
                )
                neuron.n_peaks_raw = int(len(neuron.peaks))
                spike_features_flat.extend(merged_feats)

            return pd.DataFrame(spike_features_flat)
        else:
            # --- Original single-video path ---
            f = video.norm_sm_f
            all_n = video.neurons
            results = Parallel(
                        n_jobs=self.n_jobs)(delayed(
                        describe_spikes)(f[n.index, :],
                                                 int(n.index),
                                                fs=fs)
                                                          for n in all_n)

            spike_features_flat: list[dict] = []
            for neuron, (feats_list, _keys, peaks) in zip(video.neurons, results):
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

        # Drop internal bookkeeping columns before preparing features
        inference_df = spk_feats_df.drop(
            columns=["_section_key", "_section_kind", "_segment"],
            errors="ignore",
        )

        transform = model_config.get("transform") if model_config else None
        X = prepare_features(inference_df, spike_model, transform)

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

        In concatenated mode, spike statistics are summarized per
        parsed section key alongside the whole-trace aggregates.
        """
        kinetics = SpikeKinetics(fs=float(video.fs))

        is_concat = video.is_concatenated and bool(video.concat_sections)

        for neuron in video.neurons:
            neuron.instantiate_spikes(
                sm_norm_f=video.norm_sm_f[neuron.index, :],
                sg_norm_f=video.norm_sg_f[neuron.index, :],
            )
            neuron.all_spk_stats = []

            for sp in neuron.spikes:
                if sp.f_small_window_sg is None:
                    continue
                sp.stats = kinetics.compute(sp.f_small_window_sg)
                if is_concat:
                    for section in video.concat_sections:
                        if section.start_frame <= sp.sm_f_idx < section.end_frame:
                            sp.section_key = section.section_key
                            sp.section_kind = section.section_kind
                            sp.stats["_section_key"] = section.section_key
                            sp.stats["_section_kind"] = section.section_kind
                            break
                neuron.all_spk_stats.append(sp.stats)

        if is_concat:
            per_neuron = {
                n.index: n.summarize_spikes_by_section(
                    f_trace_raw=video.suite2p_data["F"][n.index],
                    sections=video.concat_sections,
                )
                for n in video.neurons
            }
        else:
            per_neuron = {
                n.index: n.summarize_spikes(
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
        """Full pipeline: extract → filter → statistics. Returns a SpikeReport."""
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
