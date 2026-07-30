"""Leaf-level data contracts produced by experiment processing.

``VideoRunRecord`` is the boundary between processing one ``Video`` and
aggregating an experiment tree. It should contain processed values only, not
tree traversal or parent-level aggregation behavior. The record is immutable
so leaf processing produces a stable result that can be converted into a
``NodeSummary``.

Adding an aggregated statistic
------------------------------
Add a field here when the value originates from one processed video and is not
already represented as a key inside an existing ``StatSummary``. Then:

1. Populate the field when ``ExperimentProcessor._process_one_video`` creates
   the record.
2. Add the corresponding field to ``NodeSummary`` in ``summary_utils.py``.
3. Map it in ``summary_from_video_record``.
4. Define how multiple child values combine in ``aggregate_node_summaries``.

Do not add a separate dataclass field for every kinetics metric. Metrics stored
as keys in ``StatSummary`` are intentionally schema-flexible and propagate
without changing ``VideoRunRecord``. They are also flattened automatically for
sibling exports by ``comparison_utils.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from gcamp_analysis.experiments.summary_utils import StatSummary


@dataclass(frozen=True)
class VideoRunRecord:
    """Immutable summary produced after a single video is processed.

    Attributes
    ----------
    video_dir : Path
        Root directory of the video.
    metrics_dir : Path
        Directory where per-video metrics are saved.
    n_rois_total : int
        Total ROIs detected by Suite2p.
    n_rois_good : int
        ROIs that passed the classifier.
    n_neurons : int
        Neurons retained after spike detection.
    n_spikes_kept : int
        Total spikes across all neurons after filtering.
    n_groups_per_strategy : dict[str, int]
        Number of neuron groups per grouping strategy.
    kin_unweighted : StatSummary
        Kinetics averaged equally across neurons.
    kin_weighted_spikes : StatSummary
        Kinetics weighted by per-neuron spike count.
    freq_unweighted : StatSummary
        Spike frequency averaged equally across neurons.
    """

    video_dir: Path
    metrics_dir: Path

    n_rois_total: int
    n_rois_good: int
    n_neurons: int
    n_spikes_kept: int
    n_neurons_grouped: int = 0
    n_neurons_ungrouped: int = 0
    n_groups_per_strategy: dict[str, int] = field(default_factory=dict)

    group_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    kin_unweighted: StatSummary = field(default_factory=StatSummary)
    kin_weighted_spikes: StatSummary = field(default_factory=StatSummary)
    freq_unweighted: StatSummary = field(default_factory=StatSummary)

    kin_grouped: StatSummary = field(default_factory=StatSummary)
    kin_ungrouped: StatSummary = field(default_factory=StatSummary)
    freq_grouped: StatSummary = field(default_factory=StatSummary)
    freq_ungrouped: StatSummary = field(default_factory=StatSummary)

    light_evoked_details: dict[str, pd.DataFrame] = field(default_factory=dict)
