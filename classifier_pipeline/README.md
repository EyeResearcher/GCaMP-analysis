# `classifier_pipeline`

The shared model-training engine used by both the ROI classifier and the spike
classifier. It builds a labeled dataset, optimizes candidate models with
hyperparameter search, and persists the tuned model plus its evaluation
outputs. It is data-source agnostic: the calling module supplies a loader.

## Modules

| Module | Responsibility |
|---|---|
| `datasets.py` | `ClassifierDataset` / `DataSplit`: assemble features/labels and train-test splits. |
| `optimize.py` | `ModelOptimizer` / `OptimizationResults`: cross-validated hyperparameter search over configured model families. |
| `run_pipeline.py` | `PipelineRunner`: orchestrates dataset build → optimization → selection from a hyperparameter config. |
| `train_classifier.py` | Generic `train_classifier` entry point parameterized by a data-loader callable; shared by ROI and spike training. |
| `io_utils.py` | Load configs and labeled data; save optimization outputs (model, metrics JSON). |
| `utils.py` / `verbose_utils.py` | Model factory helpers and human-readable progress/summary printing. |
| `annotation.py` | Shared annotation helpers backing the labeling GUIs. |

## Usage

`train_classifier` is invoked by the `roi_classifier` and `spike_classifier`
packages, which pass their own data loaders. See those packages (and the root
`README.md`) for the end-to-end labeling → training workflow. Configuration
(model families and search spaces) is read from the pipeline config YAML.

## Design notes

- Scientific feature computation lives in the calling modules; this package
  only selects and fits models over already-computed features.
- Outputs are written as a joblib model plus a results JSON so downstream
  inference (`utils.inference.prepare_features`) can align features by name.
