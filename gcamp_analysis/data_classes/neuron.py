"""Neuron class for filtered ROIs."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.signal import peak_prominences
from .roi import ROI
from .spike import Spike

@dataclass
class Neuron:
    """
    Validated neuron built from an ROI.

    IMPORTANT:
    - We keep *all* ROI info accessible via delegation to `roi`.
    - We do NOT copy roi.__dict__ into neuron.__dict__.
    """

    roi: ROI
    filtered_index: int
    fs: float = 15.0

    # pipeline-populated
    spikes: List[Spike] = field(default_factory=list)
    spk_features: List[Dict[str, Any]] = field(default_factory=list)

    peaks: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))
    n_peaks_raw: int = 0
    peaks_filtered: List[int] = field(default_factory=list)

    all_spk_stats: List[Dict[str, Any]] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    # Optional debugging payload
    raw_stats: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str):
        """
        Delegate missing attributes to the ROI.
        This preserves the convenience of your old __dict__.update approach,
        while keeping Neuron typed + safer.
        """
        return getattr(self.roi, name)

    def instantiate_spikes(
        self,
        sm_norm_f: np.ndarray,
        sg_norm_f: np.ndarray,
    ) -> None:
        """Create Spike objects from ``peaks_filtered`` and attach to ``self.spikes``.

        Computes prominences, neighbour indices, and small windows needed
        for downstream kinetics.  Modifies *self* in place.
        """
        # Lazy import to avoid circular dependency (data_classes ↔ spike_processing)
        from gcamp_analysis.spike_processing.kinetics import _create_small_window

        if not self.peaks_filtered:
            self.spikes = []
            return

        sm = np.asarray(sm_norm_f, dtype=float).reshape(-1)
        sg = np.asarray(sg_norm_f, dtype=float).reshape(-1)
        peaks = np.asarray(self.peaks_filtered, dtype=int).reshape(-1)

        if sm.size < 3 or sg.size < 3 or peaks.size == 0:
            self.spikes = []
            return

        prominences, left_bases, right_bases = peak_prominences(sm, peaks)

        spikes: List[Spike] = []
        for i, peak_idx in enumerate(peaks.tolist()):
            sp = Spike(sm_f_idx=int(peak_idx), position_idx=int(i))

            sp.left_base = int(left_bases[i])
            sp.right_base = int(right_bases[i])
            sp.prominence = float(prominences[i])

            sp.prev_position_idx = int(peaks[i - 1]) if i > 0 else 0
            sp.next_position_idx = (
                int(peaks[i + 1]) if i < (peaks.size - 1) else int(sm.size - 1)
            )

            small_window, _, _ = _create_small_window(sg, peaks, i)
            sp.f_small_window_sg = small_window

            if 0 <= sp.sm_f_idx < sg.size:
                sp.f_value = float(sg[sp.sm_f_idx])

            spikes.append(sp)

        self.spikes = spikes

    def summarize_spikes(self, f_trace_raw: np.ndarray) -> Dict[str, Any]:
        """Aggregate per-spike stats into a per-neuron summary dict.

        Populates ``self.summary_stats`` and returns it.
        """
        if not self.all_spk_stats:
            self.summary_stats = {}
            return {}

        stats_df = pd.DataFrame(self.all_spk_stats)

        # Spike frequency (Hz)
        f_trace = getattr(self, "f_trace", None)
        n_frames = len(f_trace) if f_trace is not None else 0
        fs = float(self.fs)
        spike_frequency = float(len(self.spikes) / (n_frames / fs)) if n_frames > 0 else 0.0

        summary = OrderedDict()
        summary["neuron_idx"] = int(getattr(self, "index", -1))
        summary["filtered_index"] = int(self.filtered_index)
        summary["spike_frequency"] = spike_frequency
        summary["number_of_spikes"] = int(len(self.spikes))

        summary["spike_indices"] = list(self.peaks_filtered)
        summary["spike_values_normalized"] = [
            float(sp.f_value) for sp in self.spikes if sp.f_value is not None
        ]

        raw = np.asarray(f_trace_raw, dtype=float).reshape(-1)
        summary["spike_values_raw"] = [
            float(raw[sp.sm_f_idx])
            for sp in self.spikes
            if sp.sm_f_idx is not None and 0 <= int(sp.sm_f_idx) < raw.size
        ]

        for col in stats_df.columns:
            x = pd.to_numeric(stats_df[col], errors="coerce")
            summary[f"mean_{col}"] = float(x.mean())
            summary[f"var_{col}"] = float(x.var())

        self.summary_stats = dict(summary)
        return self.summary_stats

    def summarize_spikes_segmented(
        self,
        f_trace_raw: np.ndarray,
        split_frame: int,
    ) -> Dict[str, Any]:
        """Aggregate per-spike stats into baseline / treatment summaries.

        Produces the normal whole-trace summary (``spike_frequency``, …)
        *plus* ``baseline_*`` and ``treatment_*`` prefixed versions of
        every aggregated stat.  Also adds ``baseline_active`` and
        ``treatment_active`` booleans indicating whether the ROI was
        classified as active in each segment.

        Parameters
        ----------
        f_trace_raw : np.ndarray
            Full raw fluorescence trace (both segments concatenated).
        split_frame : int
            Frame index where baseline ends and treatment begins.
        """
        # First, compute the whole-trace summary via the existing method
        whole = self.summarize_spikes(f_trace_raw)
        if not whole:
            return {}

        fs = float(self.fs)
        raw = np.asarray(f_trace_raw, dtype=float).reshape(-1)

        # Segment metadata from ROI
        active_segs = getattr(self.roi, "active_segments", {})
        whole["baseline_active"] = active_segs.get("baseline", True)
        whole["treatment_active"] = active_segs.get("treatment", True)

        # Split spike stats by segment tag
        bl_stats = [s for s in self.all_spk_stats if s.get("_segment") == "baseline"]
        tx_stats = [s for s in self.all_spk_stats if s.get("_segment") == "treatment"]

        bl_spikes = [sp for sp in self.spikes if sp.sm_f_idx < split_frame]
        tx_spikes = [sp for sp in self.spikes if sp.sm_f_idx >= split_frame]

        n_bl_frames = split_frame
        n_tx_frames = len(raw) - split_frame

        # --- Baseline summary ---
        bl_freq = float(len(bl_spikes) / (n_bl_frames / fs)) if n_bl_frames > 0 else 0.0
        whole["baseline_spike_frequency"] = bl_freq
        whole["baseline_number_of_spikes"] = len(bl_spikes)

        if bl_stats:
            bl_df = pd.DataFrame(bl_stats).drop(columns=["_segment"], errors="ignore")
            for col in bl_df.columns:
                x = pd.to_numeric(bl_df[col], errors="coerce")
                whole[f"baseline_mean_{col}"] = float(x.mean())
                whole[f"baseline_var_{col}"] = float(x.var())

        # --- Treatment summary ---
        tx_freq = float(len(tx_spikes) / (n_tx_frames / fs)) if n_tx_frames > 0 else 0.0
        whole["treatment_spike_frequency"] = tx_freq
        whole["treatment_number_of_spikes"] = len(tx_spikes)

        if tx_stats:
            tx_df = pd.DataFrame(tx_stats).drop(columns=["_segment"], errors="ignore")
            for col in tx_df.columns:
                x = pd.to_numeric(tx_df[col], errors="coerce")
                whole[f"treatment_mean_{col}"] = float(x.mean())
                whole[f"treatment_var_{col}"] = float(x.var())

        self.summary_stats = whole
        return self.summary_stats

    def __repr__(self) -> str:
        idx = getattr(self.roi, "index", None)
        return f"Neuron(index={idx}, filtered_index={self.filtered_index}, spikes={len(self.spikes)})"
