# `utils`

Shared, dependency-light helpers used across the ROI classifier, spike
classifier, and the `gcamp_analysis` pipeline. Nothing here owns pipeline
state; these are pure functions and small IO helpers.

| Module | Responsibility |
|---|---|
| `io_utils.py` | Load Suite2p arrays (`load_suite2p_data`), pipeline/config YAML (`load_config`), and trained models (`load_model`). |
| `inference.py` | Align a feature DataFrame to a trained model's expected feature names before prediction (`prepare_features`, `get_model_feature_names`). |
| `label_utils.py` | Parse/format ROI and spike label keys and normalize label dictionaries (`parse_spike_key`, `make_spike_key`, `get_label_value`, `create_label_dict`). |
| `stats_utils.py` | Descriptive effect sizes and resampling tests (`compute_cohen_d`, `compute_hedges_g`, `perform_permutation_test`, `compute_bootstrap_ci`, `compare_distributions`). |
| `visualization.py` | Reusable plotting: trace-with-spikes overlays, neuron-group spatial maps, and matrix heatmaps consumed by the reporting layer. |

## Conventions

- Functions validate only at the boundary (shape/length/finite checks) and
  return `NaN` rather than raising when a quantity is not estimable.
- Statistical helpers are descriptive; they do not apply multiple-comparison
  correction unless the caller requests it.
- `load_suite2p_data` memory-maps large arrays (`F`, `Fneu`, `spks`) so callers
  should not assume writeable arrays.
