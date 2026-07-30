# Spike/event processing

This subpackage detects candidate peaks in each good ROI, computes classifier features, applies the trained spike classifier, and summarizes accepted calcium-fluorescence events.

“Spike” is the code's compact name for an accepted fluorescence event. It should not be interpreted as electrophysiologically resolving an individual action potential.

## Inputs

- Good `Neuron` objects from ROI processing.
- `norm_sm_f` for candidate detection and classifier features.
- `norm_sg_f` for transient-kinetics windows.
- Raw `F` for reporting raw peak fluorescence values.
- Sampling rate `fs`.
- A trained spike classifier and optional feature-transform configuration.

## Candidate detection and classification

Candidates are local maxima found with SciPy `find_peaks`. The minimum separation is scaled from the implementation's 20-frame reference at 15 Hz and is never below three frames. Every candidate receives four classifier features:

| Feature | Meaning |
|---|---|
| `spike_prom` | Candidate prominence divided by the full smoothed-trace range. |
| `dominance_score` | Candidate prominence divided by the largest prominence in its local peak cluster. |
| `mini_prom` | Rise from the preceding local valley to the window maximum, divided by trace range. |
| `distance` | Length in frames of the valley-to-valley transient window. |

The trained model predicts which candidates to keep. A neuron with no accepted candidate is removed from `Video.neurons`; remaining neurons receive new compact `filtered_index` values.

## Per-event kinetic outputs

Kinetics are calculated on the Savitzky-Golay-smoothed, min-max-normalized valley-to-valley window:

| Field | Definition | Unit |
|---|---|---|
| `rise_slope_hz` | Linear slope from window start to peak after normalizing the transient from baseline 0 to peak 1. | normalized fluorescence/s |
| `decay_tau_seconds` | Time from peak to the first interpolated crossing at `1/e` of amplitude; no exponential fit is required. | s |
| `half_max_width_seconds` | Interpolated full width between left and right half-amplitude crossings. | s |

If the required samples or crossings are unavailable, the metric is `NaN`, not zero.

## Per-neuron outputs

`spike_summary` has one row per final analysis neuron. Important columns are:

- `spike_frequency = accepted events / (analyzed frames / fs)`, in events/s;
- `number_of_spikes`, an integer event count;
- lists of accepted frame indices and peak values;
- the mean and sample variance of each kinetic metric across that neuron's events.

## What this module does not calculate

- no leave-one-event-out or leave-one-spike-out sensitivity curve;
- no firing-rate confidence interval or hypothesis test;
- no deconvolved action-potential count per fluorescence transient.

To answer “what does firing look like while excluding one spike,” a separate sensitivity analysis must repeatedly recompute each target summary after removing one accepted event and clearly specify whether removal is per neuron, per group, or per recording.

