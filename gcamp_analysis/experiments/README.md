# Experiment hierarchy and aggregation

This subpackage runs the video pipeline over an experiment directory tree, summarizes each video, combines child summaries into parent summaries, and builds one comparison row per immediate child directory.

## Inputs

- An experiment root containing video directories with `suite2p/plane0/F.npy`.
- Parallel, meaningful directory levels (for example treatment → day → video).
- A configured `VideoPipelineRunner` with trained models and grouping settings.

The tree is filesystem-driven. Every comparison is among **immediate siblings**. Folder names are exported as the `child` labels; the package does not infer that `Day_10` is later than `Day_2`, nor does it repair missing or misnamed days.

## Video-level values collected

Each processed video contributes:

- detected/good ROI counts, final neuron count, and accepted event count;
- group counts by strategy;
- mean/median group size, mean within-group strategy-matrix value, and mean total spikes per group;
- unweighted and spike-count-weighted kinetic summaries;
- unweighted spike-frequency summaries;
- separate summaries for neurons that belong to any enabled group and neurons that belong to none;
- optional light-evoked detail tables.

## Aggregation rules

For each statistic, `StatSummary` stores:

- `mean`;
- `var_within`, the weighted average of child total variances;
- `var_between`, the weighted variance of child means;
- `var = var_within + var_between`.

At a video leaf, kinetics are summarized across neurons both unweighted and weighted by accepted event count. Spike frequency is unweighted across neurons. When video children are aggregated, ordinary summaries are weighted by neuron count, grouped/ungrouped summaries use their corresponding neuron counts, and group-level metrics are weighted by group count. At higher directory levels, child summaries use video count where applicable.

These are descriptive aggregation and variance-decomposition calculations, not ANOVA, t-tests, mixed-effects models, or multiple-comparison-corrected significance tests.

## `sibling_comparisons.xlsx`

Every directory whose processed children can be compared receives `metrics/sibling_comparisons.xlsx`:

- `summary`: one row per immediate child;
- `legend`: a generated definition for each exported column.

Structural columns include `child`, `n_videos`, `n_neurons`, `n_groups_<strategy>`, group-size fields, and grouped fractions. Statistical columns follow:

```text
<statistic>_<mean|var|within|between>_<unweighted|weighted|grouped|ungrouped>
```

Current kinetics bases originate from per-neuron `rise_slope_hz`, `decay_tau_seconds`, and `half_max_width_seconds`; frequency uses `spike_frequency`.

`frac_grouped` uses the final analysis-neuron denominator, not all Suite2p ROIs. A neuron is grouped if it appears in at least one group under any enabled strategy.

## Day-to-day questions

The workbook supplies day-level values when days are sibling directories, including group count/size, within-group functional similarity, spikes per group, kinetics, and frequency. It does not itself compute a dedicated adjacent-day delta. The optional `gcamp_analysis.longitudinal` module preserves anchor-cell identity between same-region videos and tracks overlap of the largest anchor-day groups. Downstream day plots should still join by explicit day label and overlay recording-level points to expose missing or uncertain matches.
