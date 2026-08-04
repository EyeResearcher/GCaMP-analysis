# `roi_classifier`

First stage of the workflow. Trains a lightweight classifier that separates
active, experimentally meaningful neurons from inactive cells and false-positive
Suite2p detections. See the root `README.md` for the guided, screenshot-driven
walkthrough; this file is the module-level reference.

## Modules

| Module | Responsibility |
|---|---|
| `prepare_data.py` | Recursively find `suite2p/plane0/F.npy` under a dataset root and build a per-ROI feature dictionary (`.npy`). |
| `annotate_data.py` | Interactive GUI to label ROIs Good/Bad; writes labels back into the feature `.npy` in place. |
| `train_classifier.py` | Train and persist the ROI model via `classifier_pipeline.train_classifier`. |

## Command-line usage

```bash
python -m roi_classifier.prepare_data --dataset_root /path/to/videos
python -m roi_classifier.annotate_data --data_path data/all_roi_features.npy -n 100
python -m roi_classifier.train_classifier            # see --help for options
```

## Notes

- Features describe trace shape and activity (derivative skew, prominence
  statistics, SNR, peak density); they are not the final reported spike
  kinetics. The extractors live in `gcamp_analysis.roi_processing.features`.
- Labels use `1` (Good), `0` (Bad), `-1` (Unlabeled) with a `source` field.
- Aim for ~100–200 balanced labels before training.
