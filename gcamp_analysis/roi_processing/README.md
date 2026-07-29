# ROI and trace processing

This subpackage converts raw fluorescence into standardized trace representations and uses the trained ROI classifier to decide which Suite2p ROIs are active enough for spike analysis.

## Inputs

- `Video.suite2p_data["F"]`, shaped `(n_rois, n_frames)`.
- Sampling rate from the `Video` object.
- `traces.smooth_sigma` and sensor choice from the pipeline configuration.
- A trained ROI classifier and its optional feature-transform configuration, unless manual ROI labels are supplied.
- Optional Suite2p `stat` and `Fneu` values, retained on each ROI.

## Trace outputs

`TraceService` populates these in-memory arrays, all aligned to the original `F` rows and frames:

| Field | Definition |
|---|---|
| `norm_f` | Per-ROI min-max scaling of raw fluorescence to `[0, 1]`. |
| `norm_sm_f` | Gaussian-smoothed `norm_f`; used for ROI and candidate-spike features. |
| `norm_sg_f` | Savitzky-Golay-smoothed `norm_f`; used for transient windows/kinetics. |
| `z_f` | Per-ROI z-score of raw fluorescence. |
| `savgol_z_f` | Savitzky-Golay-smoothed z-score; used by combined functional grouping. |

It also saves `suite2p/plane0/F_minmax.npy`, which is `norm_f` and has the same shape as `F.npy`. In concatenated mode the arrays are additionally sliced into `Video.section_traces` without changing their ROI order.

## ROI classifier features

Each ROI becomes one feature row containing derivative skew/asymmetry, peak-prominence summaries, trace range, variance-of-variance, autocorrelation decay, an SNR estimate, peak density, median prominence, and prominence-weighted peak density. These features describe trace shape and activity; they are not the final reported spike kinetics.

## Classification outputs

- `ROI.is_good`: classifier/manual decision for the ROI.
- `Video.neurons`: `Neuron` objects created from good ROIs.
- `Video.bad_rois_features`: feature rows for rejected ROIs, written to `bad_rois_features` in the video workbook.
- `ROIReport`: total, good, and bad counts plus `good / total` pass rate.

In concatenated mode, every section is classified separately. `ROI.active_segments[section_key]` records each decision, and union logic retains an ROI if **any** section passes. Therefore:

- “active in a section” means that section passed the ROI classifier;
- “good ROI for the concatenated video” means at least one section passed;
- a good ROI can still be absent from final neuron outputs if the spike classifier accepts no event for it.

The classifier output is model-dependent and should not be described as a universal biological threshold.

