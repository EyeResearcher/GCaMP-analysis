# Functional grouping

This subpackage builds pairwise matrices, clusters analysis neurons, and summarizes groups.

## Operational definition of a group

A neuron group is a cluster of at least `min_group_size` final analysis neurons returned by one enabled strategy. Membership depends on the strategy, configuration, accepted events, and traces used. Groups are **functional similarity clusters**; they are not direct evidence of synapses, direction of signaling, or anatomical connections.

For the main `combined` strategy:

1. Use all final analysis neurons.
2. Compute maximum lagged Pearson trace correlation within `±max_lag` frames; negative results are clipped to zero.
3. Compute the spike-time tiling coefficient (STTC) using accepted event trains and the configured `dt` window.
4. Clip both matrices to `[0, 1]` and multiply them element by element. A pair scores highly only when both trace and event-timing similarity are high.
5. Convert similarity to distance with `1 - similarity`, run hierarchical clustering, and cut the tree at `cluster_param` using the configured linkage/criterion.
6. Drop clusters smaller than `min_group_size` (default 2).

Thus two neurons can be individually active but remain ungrouped if they do not fall into the same retained cluster.

## Strategy inputs and outputs

| Strategy | Input | Matrix meaning | Group rule |
|---|---|---|---|
| `combined` | Savitzky-Golay z-scored traces plus accepted event times | Product of max-lag correlation and STTC; larger means more similar | Hierarchical clustering on `1 - similarity` |
| `dtw` | Raw fluorescence traces, optionally downsampled | Soft-DTW distance; smaller means more similar | Connected components below a percentile-derived distance threshold |
| `light-evoked` | Smoothed normalized traces plus a light-pulse schedule | Encoded/aligned response matrix rather than generic connectivity | Cells grouped by ON/OFF response pattern and response count |

Only strategies registered in `STRATEGY_REGISTRY` can be run by `GroupingService`. At present these are `combined`, `dtw`, and `light-evoked`.

## Per-group `grouping_stats` columns

| Column | Definition |
|---|---|
| `group_id` | Strategy-local cluster identifier. Do not use it to match a group across videos/days. |
| `method` | Strategy that created the row. |
| `number_neurons` | Group size. |
| `neuron_indices` | Stable Suite2p ROI row indices. |
| `filtered_idxs` | Compact positions among retained neurons at grouping time. |
| `spike_rate` | Arithmetic mean of member neurons' `spike_frequency`. |
| `number_of_spikes` | Arithmetic mean of member neurons' accepted event counts; despite the singular name, this is not the group total. |
| `mean_<matrix>` | Mean off-diagonal matrix value among group members. Interpretation follows that matrix: higher is closer for similarity matrices, lower is closer for DTW distance. |
| `mean_<kinetic>` | Mean of the member neurons' already-computed mean kinetic values. |

Experiment aggregation separately computes mean/median group size, mean within-group strategy-matrix value, and mean **total** accepted spikes per group. The exported `mean_group_corr_<strategy>` name is historical: for a distance strategy such as DTW it is a within-group matrix value, not a Pearson correlation.

## Light-evoked detail

For each pulse, the closest accepted event from each neuron inside the response window is labeled `light-evoked`; unmatched events are `spontaneous`. Detail sheets include amplitude, kinetics, and latency. If a neuron has at least two evoked events, a separate stereotypy sheet reports mean, sample standard deviation, and coefficient of variation for amplitude and kinetics.

## Figures and statistics

- Group overlays are maps of categorical membership; no test is run.
- Heatmaps display the strategy matrix; no test is run.
Top-group cross-day TIFF tracking is implemented separately in `gcamp_analysis.longitudinal`, because it requires registration and persistent cell identities beyond a single video. Leave-one-spike-out analysis is not currently implemented.
