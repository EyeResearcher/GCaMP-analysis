# Saved reports and figures

This subpackage is the filesystem boundary. It snapshots completed `Video` results and writes them; scientific calculations belong to the processing and experiment subpackages.

## Per-video output directory

Files are written under `<video>/metrics/`.

### `<video>_metrics.xlsx`

| Sheet | Row meaning | Main definitions |
|---|---|---|
| `spike_summary` | One row per final analysis neuron | Accepted event frequency/count, event frames/values, and per-neuron kinetic means/variances |
| `grouping_stats` | One row per retained group per strategy | Group membership/size, mean member activity, and within-group matrix values |
| `bad_rois_features` | One row per ROI rejected by the ROI classifier | ROI classifier input features; index is the original Suite2p ROI row |
| light-evoked group sheets | One row per accepted event in that group | Evoked/spontaneous label, pulse latency, amplitude, and kinetics |
| `*_stereotypy` sheets | One row per neuron with at least two evoked events | Mean, sample SD, and coefficient of variation |

Excel sheet names are capped at 31 characters and made unique, so very long group/section labels can be shortened.

### NumPy matrices

- `<video>_<strategy>_matrix.npy`: square matrix for the enabled strategy. Row order is the active neuron order supplied to grouping; use saved group neuron indices and the code's grouping result when exact mapping is required.

For combined similarity, larger off-diagonal values indicate greater similarity. For DTW, smaller values indicate smaller distance.

### Figures

| Filename | Definition | Statistical test |
|---|---|---|
| `<video>_<strategy>_groups.png` | Categorical spatial overlay of retained group membership | None |
| `<video>_<strategy>_heatmap.png` | Strategy matrix heatmap | None |

Spatial group overlays deliberately use multiple categorical colors to distinguish groups.

## Experiment-level outputs

Every applicable tree node receives:

- `metrics/sibling_comparisons.xlsx` with `summary` and `legend` sheets;

The comparison workbook is aggregated. Individual neuron and group data points remain in the per-video workbook/CSV files. To overlay raw points on a day-level graph, assemble those detail rows with explicit source video and day labels rather than treating hierarchy means as individual observations.

## Missing and zero values

- `NaN` means not estimable and should generally remain missing in plots/statistics.
- A zero group count is a real count.
- Empty analyses still create an Excel workbook with an `empty` sheet so the pipeline has a consistent artifact.

