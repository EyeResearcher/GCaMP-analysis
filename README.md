# Post-Suite2p GCaMP fluorescence analysis

This repository starts after Suite2p. It assumes your recordings already
contain Suite2p outputs such as `suite2p/plane0/F.npy` and `iscell.npy`, then
uses two supervised classifiers to separate:

- active, experimentally relevant ROIs from inactive or false-positive
  detections
- candidate calcium transients from noise or artifacts

Recording analysis reports event kinetics and groups neurons with similar
activity. Separate experiment analysis compares completed recording results.

This repository does **not** run Suite2p, launch Suite2p batches, or perform
raw TIFF preprocessing. Generate and quality-check those inputs elsewhere
before using anything here.

The analysis is not intended to infer precise action-potential timing or prove
synaptic connectivity. Its outputs remain dependent on acquisition conditions,
upstream preprocessing, manual labels, and the applicability of the selected
classifiers to the new data.

## Current project status (3 September 2026)

- The ROI classifier, spike classifier, main analysis pipeline, experiment
  comparisons, and optional longitudinal tracking are implemented.
- The pipeline discovers recordings by finding directories that already contain
  `suite2p/plane0/` outputs.
- Hugging Face model-bundle loading is implemented and validates a pinned
  revision, manifest, checksums, feature order, transforms, and scikit-learn
  version before inference.
- The ROI and spike classifier artifacts are published in the public
  [`mmzinn12/gcamp-analysis-models`](https://huggingface.co/mmzinn12/gcamp-analysis-models)
  repository. It contains matched 15 Hz in-vitro and 3 Hz in-vivo model pairs.
  That repository still needs a root `manifest.json` before this checkout can
  use the automatic Hub loader end to end. Until then, download a matched pair
  and configure explicit local paths.
- `gcamp_analysis.waves` remains an experimental downstream research module. It
  is not integrated into the recording entry point and should not be treated as a validated
  production-stage biological wave claim.

## What this repo expects as input

At minimum, each recording directory used by the main pipeline must already
contain:

- `suite2p/plane0/F.npy`
- `suite2p/plane0/iscell.npy`

The pipeline also uses these files when present:

- `suite2p/plane0/Fneu.npy`
- `suite2p/plane0/spks.npy`
- `suite2p/plane0/stat.npy`
- `suite2p/plane0/ops.npy`

`ops.npy` is used only for three fields: `fs` (frame rate), `Ly`, and `Lx`
(image dimensions). In this suite2p version `ops.npy` is saved by default and
contains a merged dict of the db, pipeline settings, registration outputs, and
detection outputs; the loader discards all other fields including large
registration arrays such as `regPC`. When `ops.npy` is absent the sampling
rate falls back to 15 Hz. If `Fneu.npy` is missing, the loader substitutes
zeros. The wave and longitudinal workflows depend on specific `stat.npy`
fields: `ypix` and `xpix` (pixel coordinates used in longitudinal registration
and ROI visualization) and `med` (centroid coordinates used in wave
propagation analysis).

`iscell.npy` is required to confirm the recording has a complete suite2p
output including ROI detection and initial classification. The pipeline does
**not** apply suite2p's own cell classification. Instead it loads every row of
`F.npy` and applies its own trained ROI classifier to all detected ROIs
regardless of the suite2p label.

## Required run order

| Order | Stage | When it is required | Main command or entry point |
| --- | --- | --- | --- |
| 1 | Install the environment | Always | `conda env create -f environment.yml`, then `conda activate gcamp` |
| 2 | Generate Suite2p outputs elsewhere | Always before using this repo on a recording | External to this repository |
| 3A | Train an ROI classifier | Required when no suitable validated ROI model is available | `prepare_data` -> `annotate_data` -> `train_classifier` |
| 3B | Train a spike classifier | After 3A, when no suitable validated spike model is available | `prepare_data` -> `annotate_spikes` -> `train_classifier` |
| 4 | Configure the model pair and analysis | Always | Create `config/pipeline_config.yaml` using either explicit local paths or one pinned Hugging Face bundle |
| 5 | Validate the main analysis | Recommended before writing results | `python -m gcamp_analysis analyze /path/to/recordings --config config/pipeline_config.yaml --dry-run` |
| 6 | Analyze recordings independently | Always for primary results | `python -m gcamp_analysis analyze /path/to/recordings --config config/pipeline_config.yaml` |
| 7 | Compare experiments and optionally track cells | After recording analysis | `python -m experiment_analysis run --config config/experiment.yaml` |
| 8 | Run wave analysis | Optional and experimental; only after stage 6 | `python -m gcamp_analysis.waves ...`, followed by the relevant scripts under `wave_scripts/` |

Stages 3A and 3B may be skipped only when a previously trained model pair has
been shown to generalize to the acquisition and biological conditions being
analyzed. The ROI and spike models plus their JSON sidecars form one inference
bundle; do not mix unrelated iterations.

## Installation

### Prerequisites

- Python 3.10+
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Precomputed Suite2p outputs for every recording you want to analyze

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/EyeResearcher/GCaMP-analysis.git

# 2. Enter the repository
cd GCaMP-analysis

# 3. Create the environment
conda env create -f environment.yml

# 4. Activate it
conda activate gcamp

# 5. Fallback if conda dependency resolution is incomplete
pip install -r requirements.txt

# 6. Verify core dependencies
python -c "import numpy, scipy, pandas, sklearn, matplotlib, joblib, yaml, huggingface_hub; print('All dependencies OK')"
```

## Classifier-training workflow

Train the classifiers only when you do not already have a validated model pair
for your acquisition context. Both training workflows assume recordings have
already been processed by Suite2p.

Both workflows are thin wrappers around the shared `classifier_pipeline`
package, which handles dataset assembly, cross-validated hyperparameter
search, and model persistence. Scientific feature computation lives in the
calling module; `classifier_pipeline` only selects and fits models over
already-computed features.

### ROI classifier

The ROI workflow has three modules: `prepare_data`, `annotate_data`, and
`train_classifier`.

#### 1. `prepare_data`

This module recursively searches a dataset root for existing
`suite2p/plane0/F.npy` files and builds a per-ROI feature dictionary.

**Usage:**

```bash
# Process every recording under a dataset directory
python -m roi_classifier.prepare_data --dataset_root /path/to/videos

# Show all arguments
python -m roi_classifier.prepare_data --help
```

**Expected directory structure:**

Your dataset can be nested however you like, as long as the recordings already
contain Suite2p outputs somewhere below the dataset root.

```text
dataset_root/
├── experiment_1/
│   ├── day_1/
│   │   └── video_001/
│   │       └── suite2p/
│   │           └── plane0/
│   │               └── F.npy
│   └── day_2/
│       └── video_002/
│           └── suite2p/
│               └── plane0/
│                   └── F.npy
├── experiment_2/
│   └── suite2p/
│       └── plane0/
│           └── F.npy
└── standalone_video/
    └── suite2p/
        └── plane0/
            └── F.npy
```

ROI training discovers recordings from `suite2p/plane0/F.npy`. The full main
pipeline also requires `iscell.npy`; `Fneu.npy`, `spks.npy`, `stat.npy`, and
`ops.npy` are used when present.

#### 2. `annotate_data`

This module provides an interactive GUI for manually labeling ROIs as good
(active neurons) or bad (inactive/noise). Those labels train the ROI
classifier.

**Usage:**

```bash
# Annotate up to 100 randomly selected ROIs
python -m roi_classifier.annotate_data --data_path data/all_roi_features.npy -n 100

# Annotate only unlabeled ROIs
python -m roi_classifier.annotate_data --data_path data/all_roi_features.npy --unlabeled_only

# Show all arguments
python -m roi_classifier.annotate_data --help
```

**GUI controls:**

| Control | Action |
| --- | --- |
| `1` or **Active** button | Label ROI as good |
| `0` or **Inactive** button | Label ROI as bad |
| `Space` or `→` or **Skip** button | Skip ROI without labeling |
| `←` or **Previous** button | Go back to previous ROI |
| `Q` or `Esc` or **Save & Quit** button | Save progress and exit |

**Tips:**

- Start with `--unlabeled_only` to label new ROIs.
- Use `--labeled_only` to review and correct existing labels.
- Progress is auto-saved at regular intervals and on exit.
- Aim for at least 100 to 200 labeled ROIs with both classes represented
  before training.

#### 3. `train_classifier`

This module trains an ROI classifier from the manually labeled ROI features and
optimizes model family, transforms, and hyperparameters.

**Usage:**

```bash
# Train with default settings
python -m roi_classifier.train_classifier

# Custom model name
python -m roi_classifier.train_classifier --name my_roi_model

# Custom output directory
python -m roi_classifier.train_classifier --output_dir desired/output/dir

# Quiet mode
python -m roi_classifier.train_classifier --no-verbose
```

**Tips:**

- Aim for roughly balanced good/bad labels.
- Review the confusion matrix for systematic errors.
- If accuracy is poor, label more ROIs and retrain.

### Spike classifier

The spike workflow mirrors the ROI workflow but operates on candidate calcium
events inside ROIs that passed the ROI stage.

#### 1. `prepare_data`

Detects candidate spikes in good ROIs and extracts spike-level features.

**Usage:**

```bash
# Process spikes from labeled ROI data
python -m spike_classifier.prepare_data

# Limit the number of ROIs processed
python -m spike_classifier.prepare_data --max_rois 50

# Save to a different output file
python -m spike_classifier.prepare_data -o training_data/spike_filtering/spike_features.npy
```

#### 2. `annotate_spikes`

GUI for labeling candidate spikes as good (real transient) or bad
(noise/artifact).

**Usage:**

```bash
# Annotate spikes across all ROIs
python -m spike_classifier.annotate_spikes

# Limit to the first 20 ROIs
python -m spike_classifier.annotate_spikes --max_rois 20

# Only annotate unlabeled spikes
python -m spike_classifier.annotate_spikes --unlabeled_only

# Review labeled spikes
python -m spike_classifier.annotate_spikes --labeled_only
```

#### 3. `train_classifier`

Trains the spike classifier using the same optimization pipeline as the ROI
classifier.

**Usage:**

```bash
# Train with default settings
python -m spike_classifier.train_classifier

# Custom name and output location
python -m spike_classifier.train_classifier --output_dir spike_classifier/models --name my_spike_model

# Include auto-labeled spikes
python -m spike_classifier.train_classifier --no-manual_only
```

### Using trained classifiers

After both workflows, you should have one matched ROI/spike model pair:

```text
models/
├── roi_classifier.joblib
├── roi_classifier_results.json
├── spike_classifier.joblib
└── spike_classifier_results.json
```

These files are consumed by the main analysis pipeline.

## Analysis pipeline

The main pipeline applies the trained ROI and spike classifiers to recordings
that already contain Suite2p outputs, then extracts per-neuron activity,
grouping, and comparison results.

> **Prerequisites:**
> - A compatible ROI model plus its results JSON sidecar
> - A compatible spike model plus its results JSON sidecar
> - A user-created `config/pipeline_config.yaml`
> - An experiment directory containing video folders with existing
>   `suite2p/plane0/` outputs

### Analysis-model sources

The analysis accepts either explicit local paths or a single versioned Hugging
Face repository containing a matched ROI/spike pair. All Hugging Face
references in this README refer only to these scikit-learn analysis
classifiers.

The public model repository is
[`mmzinn12/gcamp-analysis-models`](https://huggingface.co/mmzinn12/gcamp-analysis-models).
Its current model inventory is:

| Model pair | ROI artifacts | Spike artifacts | Intended acquisition context |
| --- | --- | --- | --- |
| `15hz_invitro_base` | `roi/15hz_invitro_base/roi_classifier.joblib` and `roi_classifier_results.json` | `spike/15hz_invitro_base/spike_classifier.joblib` and `spike_classifier_results.json` | 15 Hz in-vitro recordings |
| `3hz_invivo_base` | `roi/3hz_invivo_base/invivo_roi_classifier.joblib` and `invivo_roi_classifier_results.json` | `spike/3hz_invivo_base/invivo_spike_classifier.joblib` and `invivo_spike_classifier_results.json` | 3 Hz in-vivo recordings |

Select the ROI and spike models from the same row. Each results JSON is a
required inference sidecar, not an optional training report. Model choice must
match the acquisition context and should be checked on representative labeled
data before batch analysis.

For automatic loading, the repository must also contain a root manifest that
selects one matched pair. The expected layout is:

```text
manifest.json
roi/
├── 15hz_invitro_base/
│   ├── roi_classifier.joblib
│   └── roi_classifier_results.json
└── 3hz_invivo_base/
    ├── invivo_roi_classifier.joblib
    └── invivo_roi_classifier_results.json
spike/
├── 15hz_invitro_base/
│   ├── spike_classifier.joblib
│   └── spike_classifier_results.json
└── 3hz_invivo_base/
    ├── invivo_spike_classifier.joblib
    └── invivo_spike_classifier_results.json
```

Point the pipeline at a pinned release tag or full commit, never the mutable
`main` branch:

```yaml
models:
  source: huggingface
  repo_id: mmzinn12/gcamp-analysis-models
  revision: <release-tag-or-full-commit>
```

The repository currently has no release tags. Its verified head on the earlier
status date was commit `c566b58e7dd1f63934f65566ac525cf12db914f5`, but that
revision still lacked `manifest.json` and therefore could not be loaded
automatically by this application.

For local or air-gapped use, configure all four paths explicitly:

```yaml
models:
  source: local
  roi_model_path: /path/to/roi/model.joblib
  roi_config_path: /path/to/roi/results.json
  spike_model_path: /path/to/spike/model.joblib
  spike_config_path: /path/to/spike/results.json
```

### Execution

Run the stages separately:

```bash
python -m gcamp_analysis analyze /path/to/recordings --config config/pipeline_config.yaml
python -m experiment_analysis run --config config/experiment.yaml
```

The recording command accepts `--sensor`, `--dry-run`, and `--quiet`.
A dry run computes recording results without writing reports or bundles.
Recording analysis never runs experiment comparisons. The old combined
`main.py` command has been removed.

See [recording usage](gcamp_analysis/RECORDING_ENTRY_POINT.md) and
[experiment configuration](experiment_analysis/README.md). Copy
`config/experiment_config.example.yaml` and adapt `config/recordings.example.csv`
to reference completed bundles and the actual experimental metadata.

### Acquisition format and suite2p settings

This pipeline starts from suite2p outputs; it does not read raw microscope
files. The microscope operator must run suite2p on the raw recordings before
using anything here. Suite2p supports ScanImage multi-page TIFFs, Olympus OIR
files (via the `movie` reader), HDF5, NWB, and other formats.

The following suite2p settings directly affect how the outputs are used:

| Suite2p setting | Why it matters here |
| --- | --- |
| `fs` | Saved to `ops.npy`; read as the recording frame rate for smoothing, deconvolution, and kinetics. Set it to the actual frame rate of the acquisition (e.g. 15 for 15 Hz). |
| `nplanes` | Use `1` for single-plane recordings. Multi-plane outputs are not tested with this pipeline. |
| `nchannels` | Use `1` for single-channel (GCaMP-only) recordings. |
| `save_ops_orig` | Defaults to `True`; must remain `True` for `ops.npy` to be written. |

See `config/exp_structure_config.yaml` for the acquisition settings used in
the current project.

### Organizing recordings for the analysis tree

Recording analysis discovers Suite2p recordings independently of experimental
folder order. One directory represents one recording. The following layouts
remain convenient, but comparisons are defined by explicit metadata and
experiment configuration rather than inferred automatically from folders.

```text
experiment_root/
├── Treatment_A/
│   ├── Week_1/
│   │   ├── video_001/          ← leaf: contains suite2p/plane0/
│   │   └── video_002/
│   └── Week_2/
│       ├── video_003/
│       └── video_004/
└── Treatment_B/
    ├── Week_1/
    │   └── video_005/
    └── Week_2/
        └── video_006/
```

Configure metadata levels `[treatment, timepoint]` to reproduce these comparisons.

**Project-specific example (CA1/DG × saline/muscimol × Week1–3):**

```text
experiment_root/
├── CA1/
│   ├── saline/
│   │   ├── Week1/
│   │   │   ├── 1-1/            ← region 1-1, Day 1 (implicit)
│   │   │   │   └── suite2p/plane0/
│   │   │   └── 1-2/
│   │   │       └── suite2p/plane0/
│   │   ├── Week2/
│   │   │   ├── 1-1_Day8/       ← same region, Day 8
│   │   │   └── 1-2_Day8/
│   │   └── Week3/
│   │       ├── 1-1_Day15/
│   │       └── 1-2_Day15/
│   └── muscimol/
│       ├── Week1/  …
│       ├── Week2/  …
│       └── Week3/  …
└── DG/
    ├── saline/    …
    └── muscimol/  …
```

Metadata levels `[region, treatment, timepoint]` reproduce these comparisons:

| Comparison | Siblings |
| --- | --- |
| Root level | `CA1` vs `DG` |
| Region level | `saline` vs `muscimol` (within each region) |
| Treatment level | `Week1` vs `Week2` vs `Week3` (within each treatment/region) |

**Video folder naming for longitudinal tracking:**

The video folder name encodes the region identity and, optionally, the day:

| Convention | Meaning |
| --- | --- |
| `1-1` | Region `1-1`, Day 1 (no suffix = Day 1) |
| `1-1_Day8` | Region `1-1`, Day 8 |
| `1-1_Day15` | Region `1-1`, Day 15 |
| `1-2` | Region `1-2` — tracked separately from `1-1` |

These names are optional labeling conventions. Experiment analysis requires
explicit timepoints and unique series IDs identifying the animal and field of
view. Anatomical labels such as CA1 do not uniquely identify a recording series.
`config/exp_structure_config.yaml` remains acquisition/layout documentation;
`config/experiment_config.example.yaml` demonstrates executable configuration.

### Output

**Per-video outputs** are saved in `<video>/metrics/`:

- `<video>_metrics.xlsx`
- `<video>_corr_matrix.npy`
- `<video>_dtw_matrix.npy`
- `<video>_corr_groups.png`
- `<video>_corr_heatmap.png`
- `<video>_dtw_groups.png` if enabled
- `<video>_dtw_heatmap.png` if enabled
- Portable bundle: `<video>/analysis_results/manifest.json` and referenced generation files

Additionally, `F_minmax.npy` is written to each recording's
`suite2p/plane0/` directory.

**Experiment-level outputs** are written to the configured `output_dir` and
include named comparison workbooks/CSVs, resolved recording assignments,
validation warnings, and optional tracking outputs.

Both notebooks call the same APIs as the two commands. Recording analysis does
not start comparisons; experiment analysis does not rerun recordings.

## Longitudinal tracking

Enable `alignment.enabled: true` in a longitudinal comparison to register and
track cells/groups. Without alignment, temporal comparisons summarize whole
recordings. Alignment requires bundle ROI coordinates, reference images, and
snap images plus explicit series/timepoint metadata.

## Wave-analysis status

`gcamp_analysis.waves` explores spatially propagating retinal calcium
activity. It is deliberately downstream of the main analysis rather than a
fourth production stage.

Run the main pipeline first because the ROI-based detector reads each
recording's Suite2p `F.npy`, `stat.npy`, and `ops.npy` files plus the generated
`metrics/*_metrics.xlsx` workbook. Raw TIFFs and reliable spatial/temporal
metadata are also needed for movie corroboration and interpretable propagation
speeds.

### What is implemented

- ROI-event population null testing followed by planar-versus-radial
  propagation fits and activation-time permutation tests
- An independent block-averaged raw-movie propagation analysis
- A neighbor-graph cross-correlation method for local lag reconstruction
- A WaveMiner-compatible x/y/t flood-fill reimplementation and supporting
  corroboration, stability, plotting, and summary scripts
- Focused automated tests for the wave-method code paths, primarily unit and
  synthetic-data checks

### What remains incomplete

- Wave analysis is not called by the recording entry point and is not part of the standard
  notebook workflow.
- The public CLI exposes only the primary ROI-based analysis and only a subset
  of its configuration.
- Thresholds and defaults have not been calibrated and externally validated
  across sensors, frame rates, preparations, treatments, or laboratories.
- Missing or incorrect pixel size, frame rate, TIFF alignment, or accepted
  event indices can invalidate speed estimates or cross-modal corroboration.
- The included WaveMiner-style implementation follows the published
  description but is not the official WaveMiner source.
- There is not yet an end-to-end biological validation set, locked parameter
  set, sensitivity analysis, or PI-approved acceptance criterion establishing
  which outputs support a retinal-wave conclusion.

Current primary invocation:

```bash
# Run only after the main pipeline has produced per-recording metrics
python -m gcamp_analysis.waves /path/to/dataset --output-dir wave_results --days 10

# Combine resulting outputs after one or more runs
python wave_scripts/summarize_wave_analysis.py wave_results
```

The present outputs can support exploratory statements about calcium activity
that is statistically consistent with spatial propagation. They do not by
themselves establish action-potential propagation, synaptic connectivity, or a
validated retinal-wave phenotype. See
[`gcamp_analysis/waves/README.md`](gcamp_analysis/waves/README.md) and
[`wave_scripts/README.md`](wave_scripts/README.md) for method and script-level
details.
