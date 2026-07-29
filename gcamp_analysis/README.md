# `gcamp_analysis` package

This package is the analysis stage of the project. It loads one Suite2p video, transforms and classifies its fluorescence traces, detects and classifies candidate calcium events, groups neurons by functional similarity, and aggregates results through an experiment directory tree.

## Data flow

```text
Suite2p arrays + trained ROI model + trained spike model + pipeline config
    -> trace preprocessing
    -> ROI classification
    -> candidate-spike detection and classification
    -> per-neuron activity and transient-kinetics summaries
    -> functional grouping
    -> per-video files and experiment-level comparisons
```

The subpackage READMEs describe each boundary in detail:

- [`concatenation/README.md`](concatenation/README.md): metadata for videos made from baseline, treatment, and recovery sections.
- [`data_classes/README.md`](data_classes/README.md): the in-memory `Video`, `ROI`, `Neuron`, `Spike`, and `NeuronGroup` objects.
- [`roi_processing/README.md`](roi_processing/README.md): trace transformations and ROI classification.
- [`spike_processing/README.md`](spike_processing/README.md): event detection, filtering, and kinetic measurements.
- [`grouping_processing/README.md`](grouping_processing/README.md): the definition of a group, similarity matrices, clustering, and section comparisons.
- [`experiments/README.md`](experiments/README.md): aggregation across videos and directory levels.
- [`reporting/README.md`](reporting/README.md): every saved workbook, array, CSV, and figure.
- [`longitudinal/README.md`](longitudinal/README.md): registration of same-region masks and largest-group membership tracking across days.

The two top-level orchestration modules have narrow interfaces: `video_runner.py` receives a configured `Video`, models, and services and runs traces → ROIs → spikes → grouping in that order; `reports.py` defines the small immutable count/summary records returned by those stages. Neither module introduces an additional scientific metric.

## Terms used in the outputs

| Term | Definition in this pipeline |
|---|---|
| Detected ROI | One row of Suite2p `F.npy`; it has not necessarily passed either classifier. |
| Active/good ROI | An ROI accepted by the trained ROI classifier (or by supplied manual labels). In concatenated mode it is retained if at least one section passes. |
| Analysis neuron | A good ROI that also has at least one candidate event accepted by the spike classifier. ROIs with no accepted events are removed before grouping and final spike summaries. |
| Spike/event | A local maximum of the smoothed, min-max-normalized trace that the spike classifier accepts. It is a detected calcium-fluorescence event, not a claim of an electrophysiologically resolved action potential. |
| Spike frequency | Accepted event count divided by analyzed duration in seconds, reported in events/s (numerically Hz). |
| Neuron group | At least `min_group_size` analysis neurons assigned to the same cluster by one enabled grouping strategy. See the grouping README for the strategy-specific rule. |
| Group size | Number of neurons in that group, not number of spikes, pixels, or ROIs initially detected by Suite2p. |
| Connectivity | Functional similarity according to the selected trace/event metric. It does **not** establish a synapse, directionality, or anatomical connectivity. |
| Grouped neuron | An analysis neuron belonging to at least one group from any enabled strategy. |
| Ungrouped neuron | An analysis neuron that belongs to no retained group from any enabled strategy. |

## Interpreting the email questions

### Outputs already produced

- **Visual maps of groupings:** `<video>_<strategy>_groups.png` overlays the retained groups on Suite2p ROI locations.
- **Group size and activity:** the `grouping_stats` workbook sheet contains `number_neurons`, the mean member `spike_rate`, and mean member `number_of_spikes`. The hierarchy-level comparison workbook also contains group counts and mean/median group size.
- **Baseline versus later sections:** concatenated mode compares baseline groups with each treatment/recovery section and reports similarity change, active/inactive membership, re-clustered subgroups, and spatial dispersion.
- **Individual points behind comparisons:** per-neuron rows are in `spike_summary`; per-group rows are in `grouping_stats` and section-comparison CSVs. Experiment summary sheets are aggregated rows, so these detail files should be retained when plotting individual observations.
- **Day alignment:** the analysis does not infer days from a plot position. Experiment comparisons use immediate directory children and their folder names. Parallel, consistently named directory levels are therefore required; day labels should be treated as categorical labels unless downstream plotting explicitly parses and sorts them.

### Optional longitudinal outputs

The standard per-video pipeline still treats days independently. The optional `gcamp_analysis.longitudinal` command now registers same-region Suite2p masks, selects the largest groups on an anchor day, tracks their members across days, and writes colored TIFF overlays plus cell/group history tables. This supports `1-1` → `1-1_DayN` while keeping `1-2` and other regions separate. See the [longitudinal README](longitudinal/README.md).

### Not currently produced by the package

- A single scalar dedicated to day-4-to-day-5 or day-6-to-day-7 change. Longitudinal cell/group rows now provide the source values, from which a specifically defined adjacent-day change can be calculated.
- A leave-one-spike-out sensitivity analysis (“what firing data look like while excluding one spike”).
- Inferential tests attached to the saved production plots. The package computes descriptive means, variances, counts, correlations/similarities, and within/between variance components. A plotted separation is not automatically a p-value or significance result.

## Statistics associated with saved figures

| Figure | Observation represented | Calculation | Inferential test |
|---|---|---|---|
| `*_groups.png` | One colored spatial shape per retained neuron group | Cluster membership and Suite2p ROI pixels/centroids | None |
| `*_heatmap.png` | One cell per neuron pair | Strategy matrix (combined functional similarity, DTW distance, or light-response matrix) | None |
| `*_delta_corr_vs_dispersion.png` | One point per baseline group | x = baseline mean pairwise centroid distance; y = section mean combined similarity minus baseline mean combined similarity; marker area = group size | None |
| `*_centroid_distances.png` | One point per active neuron from a baseline group that forms a section subgroup | Distance from baseline-group centroid versus distance from section-subgroup centroid | None |

The reporting layer currently uses blue, black, and gray for the section-comparison scatter plots. Group overlays use categorical colors so simultaneous groups remain distinguishable; matrix heatmaps use a continuous colormap.

## Units and missing values

- Frame indices are zero-based Python indices; section `end_frame` values are exclusive.
- Sampling rate `fs` is in frames/s (Hz).
- Spatial distances are in Suite2p image pixels.
- Latencies are in milliseconds; decay tau and half-maximum width are in seconds.
- `NaN` means the quantity could not be estimated, commonly because a transient was too short, lacked a threshold crossing, or had insufficient finite values. It should not be silently interpreted as zero.
