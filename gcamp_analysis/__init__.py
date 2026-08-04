"""GCaMP fluorescence analysis library.

``gcamp_analysis`` is the analysis stage of the project. It loads one Suite2p
video, transforms and classifies its fluorescence traces, detects and
classifies candidate calcium events, groups neurons by functional similarity,
and aggregates results across an experiment directory tree.

Subpackages
-----------
- ``data_classes``: in-memory ``Video``, ``ROI``, ``Neuron``, ``Spike``,
  ``NeuronGroup`` objects.
- ``roi_processing``: trace preprocessing and ROI classification.
- ``spike_processing``: candidate-event detection, filtering, and kinetics.
- ``grouping_processing``: similarity matrices, clustering, and group summaries.
- ``experiments``: hierarchical aggregation and sibling comparisons.
- ``reporting``: filesystem writers for workbooks, arrays, CSVs, and figures.
- ``waves``: detection of propagating calcium episodes.
- ``longitudinal``: cross-day registration and functional-group tracking.

See ``gcamp_analysis/README.md`` for the full data flow and terminology.
"""
