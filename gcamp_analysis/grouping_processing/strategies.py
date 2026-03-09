"""Grouping strategies: Protocol, concrete implementations, and registry.

To add a new strategy:
  1. (Optional) Add a similarity function in ``similarity.py``
  2. Write a class with ``name: str`` and ``compute(video, config) -> GroupingResult``
  3. Register it in ``STRATEGY_REGISTRY`` at the bottom of this file
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Protocol

import numpy as np
from scipy.signal import savgol_filter

from gcamp_analysis.data_classes.neuron_group import NeuronGroup
from gcamp_analysis.grouping_processing.similarity import (
    compute_correlation_matrix,
    compute_dtw_matrix,
    compute_sttc_matrix,
)
from gcamp_analysis.grouping_processing.clustering import (
    cluster_hierarchical,
    cluster_threshold_graph,
)

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.video import Video


# ── Result container + Protocol ──────────────────────────────────────


@dataclass(frozen=True)
class GroupingResult:
    """Output of a single grouping strategy."""

    groups: list[NeuronGroup]
    matrix: np.ndarray | None
    config_label: str


class GroupingStrategy(Protocol):
    """Interface every grouping strategy must satisfy."""

    name: str

    def compute(self, video: "Video", config: Dict[str, Any]) -> GroupingResult: ...


# ── Concrete strategies ──────────────────────────────────────────────


@dataclass
class CorrelationStrategy:
    """Trace-correlation grouping with threshold-graph clustering."""

    name: str = "corr"

    def compute(self, video: "Video", config: Dict[str, Any]) -> GroupingResult:
        trace_key = str(config.get("trace", "norm_sm_f"))
        method = str(config.get("method", "pearson"))
        if trace_key == "F":
            traces_full = np.asarray(video.suite2p_data["F"], float)
        else:
            traces_full = getattr(video, trace_key, None)
            if traces_full is None:
                raise ValueError(f"Trace key '{trace_key}' not found on video object.")
    
        if video.is_concatenated and video.split_frame is not None:
            sf = video.split_frame
            traces = traces_full[[n.index for n in video.neurons], :sf]
        else:
            traces = traces_full[[n.index for n in video.neurons], :]

        C = compute_correlation_matrix(
            traces,
            method=method,
            remove_global=bool(config.get("remove_global", True)),
            use_diff=bool(config.get("use_diff", True)),
            diff_order=int(config.get("diff_order", 1)),
            zscore_each=bool(config.get("zscore_each", True)),
            clip_negatives=bool(config.get("clip_negatives", True)),
        )

        groups = cluster_threshold_graph(
            video.neurons,
            1.0 - C,
            threshold=float(config.get("distance_threshold", 0.6)),
            min_group_size=int(config.get("min_group_size", 2)),
            method="corr",
        )

        label_parts = [f"corr_{trace_key}_{method}"]
        if config.get("remove_global", True):
            label_parts.append("rmglobal")
        if config.get("use_diff", True):
            label_parts.append(f"diff{config.get('diff_order', 1)}")
        if config.get("zscore_each", True):
            label_parts.append("z")

        return GroupingResult(groups=groups, matrix=C, config_label="_".join(label_parts))


@dataclass
class STTCStrategy:
    """STTC-based grouping with hierarchical clustering."""

    name: str = "sttc"

    def compute(self, video: "Video", config: Dict[str, Any]) -> GroupingResult:
        tw = float(config.get("time_window", 0.033))
        dt = float(config.get("distance_threshold", 0.3))
        link = str(config.get("linkage_method", "average"))
        min_group = int(config.get("min_group_size", 2))

        if video.is_concatenated and video.split_frame is not None:
            # Only use baseline spikes (those before split_frame)
            n_frames = video.split_frame
        else:
            n_frames = video.n_frames

        sttc = compute_sttc_matrix(
            video.neurons, n_frames, time_window=tw, fs=float(video.fs),
            max_frame=video.split_frame if (video.is_concatenated and video.split_frame is not None) else None,
        )

        groups = cluster_hierarchical(
            video.neurons, 1.0 - sttc,
            threshold=dt, linkage_method=link, min_group_size=min_group,
            method="sttc", group_id_prefix="sttc", t_win=tw, sttc_thresh=1.0 - dt,
        )

        return GroupingResult(groups=groups, matrix=sttc, config_label=f"sttc_tw{tw}_dt{dt}")


@dataclass
class DTWStrategy:
    """GPU-accelerated SoftDTW grouping with hierarchical clustering."""

    name: str = "dtw"

    def compute(self, video: "Video", config: Dict[str, Any]) -> GroupingResult:
        down = int(config.get("downsample_factor", 3))
        gpu = bool(config.get("use_gpu", True))
        link = str(config.get("linkage_method", "average"))
        pctl = int(config.get("distance_percentile", 30))
        min_group = int(config.get("min_group_size", 2))

        # In concatenated mode, only use baseline segment for DTW
        if video.is_concatenated and video.split_frame is not None:
            dtw = compute_dtw_matrix(
                video.neurons, downsample_factor=down, use_gpu=gpu,
                max_frame=video.split_frame,
            )
        else:
            dtw = compute_dtw_matrix(video.neurons, downsample_factor=down, use_gpu=gpu)
        if dtw is None:
            return GroupingResult(groups=[], matrix=None, config_label="dtw_skipped")

        dtw = np.asarray(dtw, dtype=float)
        nonzero = dtw[dtw > 0]
        if nonzero.size == 0:
            return GroupingResult(groups=[], matrix=dtw, config_label="dtw_empty")

        thresh = float(np.percentile(nonzero, pctl))
        groups = cluster_hierarchical(
            video.neurons, dtw,
            threshold=thresh, linkage_method=link, min_group_size=min_group,
            method="dtw", group_id_prefix="dtw", dtw_thresh=thresh,
        )

        return GroupingResult(groups=groups, matrix=dtw, config_label="dtw")


# ── Registry ─────────────────────────────────────────────────────────

STRATEGY_REGISTRY: Dict[str, type] = {
    "corr": CorrelationStrategy,
    "sttc": STTCStrategy,
    "dtw": DTWStrategy,
}
