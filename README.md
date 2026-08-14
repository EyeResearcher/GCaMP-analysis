# Post-Suite2p GCaMP fluorescence analysis

This repository is a lab-personalizable workflow for population-level analysis
of GCaMP fluorescence. It uses two supervised classifiers: one separates active,
experimentally relevant ROIs from inactive or false-positive Suite2p detections,
and the other separates candidate calcium transients from noise or artifacts.
The main pipeline then reports event kinetics, groups neurons with similar
activity, aggregates recordings, and compares sibling experimental conditions.

The analysis is not intended to infer precise action-potential timing or prove
synaptic connectivity. Its inputs are Suite2p-style arrays, and its outputs
remain dependent on acquisition conditions, preprocessing, manual labels, and
the applicability of the selected classifiers to the new data.

## Current project status (14 August 2026)

- The ROI classifier, spike classifier, main analysis, experiment comparisons,
  and optional longitudinal tracking are implemented.
- Hugging Face model-bundle loading is implemented and validates a pinned
  revision, manifest, checksums, feature order, transforms, and scikit-learn
  version before inference.
- The ROI and spike classifier artifacts are published in the public
  [`mmzinn12/gcamp-analysis-models`](https://huggingface.co/mmzinn12/gcamp-analysis-models)
  repository. It contains matched 15 Hz in-vitro and 3 Hz in-vivo model pairs.
  The repository does not yet contain the root `manifest.json` required by this
  application's automatic Hub loader, and this checkout does not contain a
  portable `config/pipeline_config.yaml`. Until those integration pieces are
  added, download a matched pair and use explicit local paths.
- `gcamp_analysis.waves` is an **incomplete, experimental research module**.
  Its algorithms and focused synthetic/unit tests are present, but it is not
  integrated into the main pipeline and is not yet validated for a final
  biological wave claim. See [Wave-analysis status](#wave-analysis-status) below.

## Required run order

| Order | Stage | When it is required | Main command or entry point |
| --- | --- | --- | --- |
| 1 | Install the environment | Always | `conda env create -f environment.yml`, then `conda activate gcamp` |
| 2 | Produce Suite2p outputs | Skip only if every recording already has `suite2p/plane0/` arrays | Suite2p directly, or the machine-specific `preprocessing/batch_s2p.py` helper |
| 3A | Train an ROI classifier | Required when no suitable validated ROI model is available | `prepare_data` -> `annotate_data` -> `train_classifier` |
| 3B | Train a spike classifier | After 3A, when no suitable validated spike model is available | `prepare_data` -> `annotate_spikes` -> `train_classifier` |
| 4 | Configure the model pair and analysis | Always | Create `config/pipeline_config.yaml` using either explicit local paths or one pinned Hugging Face bundle |
| 5 | Validate the main analysis | Recommended before writing results | `python main.py /path/to/experiment_root --dry-run` |
| 6 | Run the main analysis and comparisons | Always for primary results | `python main.py /path/to/experiment_root` |
| 7 | Run longitudinal tracking | Optional; only after stage 6 has produced per-recording metrics | `python -m gcamp_analysis.longitudinal ...` |
| 8 | Run wave analysis | Optional and experimental; only after stage 6 | `python -m gcamp_analysis.waves ...`, followed by the relevant scripts |

Stages 3A and 3B may be skipped only when a previously trained model pair has
been shown to generalize to the acquisition and biological conditions being
analyzed. The two models and their JSON sidecars form one inference bundle;
do not mix unrelated iterations.
## Installation 
### Prerequisites
- Python 3.10+ 
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
    - During installation, its advisable to check "Add to PATH" so conda is accessible 
- [Suite2p](https://github.com/MouseLand/suite2p) output data
### Setup
First, open a command prompt or an anaconda prompt on your computer. Then,
```bash
# 1. Clone the repository (or download the zip directly from GitHub and open the main folder)
git clone https://github.com/EyeResearcher/GCaMP-analysis.git

# 2. Navigate to the main folder
cd GCaMP-analysis

# 3. Create a fully equipped conda environment 
conda env create -f environment.yml


# 4. Activate that environment so all the tools are available to the program
conda activate gcamp

# Optionally, if conda dependency installation is not working, run this after activation
pip install -r requirements.txt

# 5. Verify dependency installation 
python -c "import numpy, scipy, pandas, sklearn, matplotlib, joblib, yaml, huggingface_hub; print('All dependencies OK')"
```
## Classifier-training workflow

This section expands stages 3A and 3B from the run-order table. Training is a
sequential workflow when suitable pretrained analysis classifiers are not
available.

### ROI Classifier 
The first stage comprises 3 modules: `prepare_data`, `annotate_data`, and `train_classifier`. A guided workflow is provided in jupyter notebooks, but each module's functionality can also be executed in a command terminal. 

#### 1. `prepare_data`
This module recursively searches a specified directory for Suite2p outputs and writes a dictionary containing information for every ROI found.

**Usage:**
```bash
# Process raw arrays of fluorescence data from a dataset directory
python -m roi_classifier.prepare_data --dataset_root /path/to/videos

# Get description of the module's purpose and its arguments 
python -m roi_classifier.prepare_data --help
```

**Expected Directory Structure:**

The program recursively searches `dataset_root` for any `suite2p/plane0/F.npy` files. Your data can be organized in any nested structure as long as Suite2p outputs exist somewhere within:

```
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
`ops.npy` are used when present. The downstream wave workflow requires
`F.npy`, `stat.npy`, and `ops.npy` plus the main pipeline's metrics workbook.

#### 2. `annotate_data`

This module provides an interactive GUI for manually labeling ROIs as "Good" (active neurons) or "Bad" (inactive/noise). Labels are used to train the ROI classifier.

**Usage:**
```bash
# Annotate up to 100 randomly selected ROIs
python -m roi_classifier.annotate_data --data_path data/all_roi_features.npy -n 100

# Only annotate unlabeled ROIs
python -m roi_classifier.annotate_data --data_path data/all_roi_features.npy --unlabeled_only

# Get descriptions of all arguments
python -m roi_classifier.annotate_data --help
```
**GUI Controls:**

| Control | Action |
|---------|--------|
| `1` or **Active** button | Label ROI as Good (active neuron) |
| `0` or **Inactive** button | Label ROI as Bad (inactive/noise) |
| `Space` or `→` or **Skip** button | Skip ROI without labeling |
| `←` or **Previous** button | Go back to previous ROI |
| `Q` or `Esc` or **Save & Quit** button | Save progress and exit |

**Tips:**
- Start with `--unlabeled_only` to label new ROIs
- Use `--labeled_only` to review and correct existing labels
- Progress is auto-saved at regular intervals and on exit
- Aim for at least 100-200 labeled ROIs (balanced between Good/Bad) before training

**Output:**

Updates the `.npy` file in place with:
- `label.value`: `1` (Good), `0` (Bad), or `-1` (Unlabeled)
- `label.source`: `'manual'` for human-annotated labels

#### 3. `train_classifier`

This module trains an ROI classifier using the manually labeled data from `annotate_data`. It automatically optimizes model type, feature transforms, and hyperparameters.

**Usage:**
```bash
# Train with default settings
python -m roi_classifier.train_classifier 

# Specify model name
python -m roi_classifier.train_classifier --name my_roi_model

# Specify output directory (must remember where it is stored for pipeline anlaysis)
python -m roi_classifier.train_classifier --output_dir desired/output/dir

# Quiet mode
python -m roi_classifier.train_classifier --no-verbose
```

**Tips:**
- Ensure you have at least 500+ labeled ROIs before training
- Aim for balanced classes (roughly equal Good/Bad labels)
- Review the confusion matrix to identify systematic errors
- If accuracy is low, label more ROIs and retrain

### Spike Classifier

The second stage follows the same three-step workflow as the ROI classifier, but operates on candidate fluorescence events (spikes) within good ROIs. It trains a classifier to distinguish real calcium transients from noise artifacts.

> **Prerequisite:** Complete the ROI Classifier workflow first. Only ROIs labeled as "Good" will be processed for spike detection.

#### 1. `prepare_data`

Detects candidate spikes in all good ROIs and extracts spike-level features.

**Usage:**
```bash
# Process spikes from labeled ROI data
python -m spike_classifier.prepare_data 
# Limit number of ROIs processed
python -m spike_classifier.prepare_data --max_rois 50

# Save to a different output file
python -m spike_classifier.prepare_data -o training_data/spike_filtering/spike_features.npy
```

**Output:**

Updates the `.npy` file, adding a `spikes` dictionary to each good ROI:
```python
roi_data['spikes'] = {
    spike_idx: {
        'features': {...},      # Spike features (prominence, width, etc.)
        'windows': {...},       # Window indices for visualization
        'label': {'value': -1, 'source': 'unlabeled'}
    },
    ...
}
```

---

#### 2. `annotate_spikes`

ROI-centric GUI for labeling candidate spikes as "Good" (real transient) or "Bad" (noise/artifact).

**Usage:**
```bash
# Annotate spikes across all ROIs
python -m spike_classifier.annotate_spikes 

# Limit to first 20 ROIs
python -m spike_classifier.annotate_spikes --max_rois 20

# Only annotate unlabeled spikes
python -m spike_classifier.annotate_spikes --unlabeled_only

# Review previously labeled spikes
python -m spike_classifier.annotate_spikes --labeled_only
```
**Tips:**
- Use `X` (label remaining as bad) to quickly dismiss obvious noise ROIs
- The spike listbox lets you jump to specific spikes within an ROI
- Aim for 200-500 labeled spikes before training

---

#### 3. `train_classifier`

Trains a spike classifier using the same optimization pipeline as the ROI classifier.

**Usage:**
```bash
# Train with default settings
python -m spike_classifier.train_classifier 

# Custom name and output
python -m spike_classifier.train_classifier --output_dir spike_classifier/models --name my_spike_model

# Include auto-labeled spikes
python -m spike_classifier.train_classifier --no-manual_only
```

**Output:**

- `<name>.joblib`: Trained spike classifier
- `<name>_results.json`: Training metrics and configuration

---

### Using the Trained Classifiers

After completing both classifier workflows, you'll have:
```
models/
├── roi_classifier.joblib
└── roi_classifier_results.json
├── spike_classifier.joblib
└── spike_classifier_results.json

```

These models are used by the Analysis Pipeline (next section) to automatically filter ROIs and spikes during batch processing.

---

### Analysis Pipeline

The final stage applies the trained classifiers to process entire experiment directories, extracting spike statistics and comparing experimental conditions.

> **Prerequisites:**
> - A compatible ROI model plus its results JSON sidecar
> - A compatible spike model plus its results JSON sidecar
> - A user-created `config/pipeline_config.yaml` (it is not currently shipped)

### Analysis-model sources

The analysis accepts either explicit local paths or a single versioned Hugging
Face repository containing the compatible ROI/spike pair. All Hugging Face
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
selects one matched pair. The complete expected layout is:

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

Iteration folder names are arbitrary stable identifiers. The root
`manifest.json` selects the active ROI and spike iterations by storing their
exact nested paths; the model and results sidecar for an iteration must be in
the same folder. This allows the repository to retain older iterations without
the application guessing which one to load.

Point the pipeline at a pinned release tag or full commit (never the mutable
`main` branch):

```yaml
models:
  source: huggingface
  repo_id: mmzinn12/gcamp-analysis-models
  revision: <release-tag-or-full-commit>
```

The repository currently has no release tags. Its verified head on the status
date is commit `c566b58e7dd1f63934f65566ac525cf12db914f5`, but that revision does
not yet contain `manifest.json` and therefore cannot be loaded automatically by
the current application. After a manifest and release tag are published, pin
that immutable tag or its full commit hash rather than `main`. The application
will then download the allow-listed files once through `huggingface_hub`,
validate them, and reuse the standard local cache. No Hugging Face CLI is
required for normal use of this public repository.

For local or air-gapped use, configure all four paths explicitly:

```yaml
models:
  source: local
  roi_model_path: /path/to/roi/model.joblib
  roi_config_path: /path/to/roi/results.json
  spike_model_path: /path/to/spike/model.joblib
  spike_config_path: /path/to/spike/results.json
```

#### Execution

**Usage:**
```bash
# Run pipeline on an experiment directory
python main.py /path/to/experiment_root

# Optionally specify the sensor type 
python main.py /path/to/experiment_root --sensor <your_sensor>

# Validate the full analysis without writing output files
python main.py /path/to/experiment_root --dry-run

# Display program purpose and argument definitions
python main.py --help
```

`--dry-run` loads the configured models and Suite2p data and performs trace
processing, ROI and spike classification, grouping, experiment aggregation,
and sibling comparisons. It prints the normal progress and summary output but
does not create or modify metrics workbooks, NumPy matrices, figures, or output
directories.
#### Directory Structure & Sibling Comparisons

The pipeline automatically compares sibling directories at each level of your experiment hierarchy. **Parallel directory structures are required** for meaningful comparisons.

**Recommended Structure:**
Note: this specific nomenclature is based on our lab's sets of experiments. The important idea is that each item, i.e. child, immediately wihtin the same folder, i.e. parent, represent the same type of group and can/should be meaningfully compared. 
```
experiment_root/
├── Treatment_A/
│   ├── Week_1/
│   │   ├── video_001/
│   │   │   └── suite2p/plane0/F.npy
│   │   └── video_002/
│   │       └── suite2p/plane0/F.npy
│   └── Week_2/
│       ├── video_003/
│       │   └── suite2p/plane0/F.npy
│       └── video_004/
│           └── suite2p/plane0/F.npy
└── Treatment_B/
    ├── Week_1/
    │   └── video_005/
    │       └── suite2p/plane0/F.npy
    └── Week_2/
        └── video_006/
            └── suite2p/plane0/F.npy
```

This structure enables comparisons:
- `Treatment_A` vs `Treatment_B` (at root level)
- `Week_1` vs `Week_2` (within each treatment)

> **Note:** Each parent folder's immediate children are compared as siblings. Name folders descriptively so comparisons are interpretable.

---

#### Output

**Per-Video Outputs** (saved in `<video>/metrics/`):
- `<video>_metrics.xlsx`: Multi-sheet workbook containing:
  - `spike_summary`: Per-neuron spike statistics (kinetics, frequency, etc.)
  - `grouping_stats`: Neuron group membership and similarity metrics
  - `bad_rois_features`: Features of ROIs classified as inactive/noise
- `<video>_corr_matrix.npy`: Pairwise neuron correlation matrix
- `<video>_dtw_matrix.npy`: Pairwise DTW distance matrix
- `<video>_corr_groups.png`: Spatial overlay of correlation-based neuron groups
- `<video>_corr_heatmap.png`: Correlation matrix heatmap
- `<video>_dtw_groups.png`: Spatial overlay of DTW-based neuron groups (if enabled)
- `<video>_dtw_heatmap.png`: DTW distance heatmap (if enabled)
- `<video>_analysis_summary.json`: Versioned, comparison-ready summary used
  without rerunning video analysis

Additionally, `F_minmax.npy` (min-max normalized traces) is saved into each video's `suite2p/plane0/` directory.

**Experiment-Level Outputs** (saved in `<experiment_root>/metrics/`):
- `sibling_comparisons.xlsx`: Multi-sheet Excel file with statistical comparisons at each hierarchy level

For notebooks, per-video computation and comparison are separate stages:

1. `notebooks/pipeline.ipynb` analyzes every discovered video independently.
2. `notebooks/comparative_analysis.ipynb` validates and loads the persisted
   summaries, then runs longitudinal, treatment, or generic hierarchy
   comparisons.

Longitudinal comparison with `align=False` uses whole-video descriptive
statistics across days and makes no cell-identity claim. Setting `align=True`
adds image registration, ROI matching, and cell/group tracking. Treatment
comparisons require an experiment-appropriate replicate unit such as well,
region, or animal. Explicit metadata tables are supported alongside the
existing folder-name inference.

---

### Optional Longitudinal Group Tracking

After the per-video pipeline has created each recording's Suite2p masks and
metrics workbook, repeated recordings of one region can be registered and
tracked across days:

```bash
python -m gcamp_analysis.longitudinal /path/to/experiment_root --region 1-1
```

The base recording name is treated as the region identity: `1-1`,
`1-1_Day2`, and `1-1_Day10` are matched to one another, while `1-2` is kept
separate. Treatments are also processed separately. The command selects the
largest 10% of groups on the latest day by default, then writes cell-match
quality tables, group-membership histories, and a registered multi-page TIFF
overlay. See [`gcamp_analysis/longitudinal/README.md`](gcamp_analysis/longitudinal/README.md)
for options, output definitions, and quality-control requirements.

---

## Wave-analysis status

The `gcamp_analysis.waves` package explores spatially propagating retinal
calcium activity. It is deliberately downstream of the main analysis rather
than a fourth production stage. Run the main pipeline first because the
ROI-based detector reads each recording's Suite2p `F.npy`, `stat.npy`, and
`ops.npy` files plus the generated `metrics/*_metrics.xlsx` workbook. Raw TIFFs
and reliable spatial/temporal metadata are needed for movie corroboration and
interpretable propagation speeds.

### What is implemented

- ROI-event population null testing followed by planar-versus-radial
  propagation fits and activation-time permutation tests.
- An independent block-averaged raw-movie propagation analysis.
- A neighbor-graph cross-correlation method for local lag reconstruction.
- A WaveMiner-compatible x/y/t flood-fill reimplementation and supporting
  corroboration, stability, plotting, and summary scripts.
- Focused automated tests for all four methods. The wave test subset currently
  passes, but these are primarily unit and synthetic-data checks.

### What remains incomplete

- Wave analysis is not called by `main.py`, is not part of the normal notebook
  workflow, and its alternative methods still require separate scripts under
  `scripts/`.
- The public CLI exposes only the primary ROI-based analysis and only a subset
  of its configuration. Dataset discovery also assumes the specific
  `<dataset>/<treatment>/<recording>/suite2p/plane0/` depth.
- Thresholds and defaults have not been calibrated and externally validated
  across sensors, frame rates, preparations, treatments, or laboratories.
- Missing or incorrect pixel size, frame rate, TIFF alignment, or accepted
  event indices can invalidate speed estimates or cross-modal corroboration.
- The official WaveMiner source cited by Yeager et al. (2025) was not publicly
  available when this module was written. The included implementation follows
  the published description but is not the official code and should not be
  described as an exact reproduction.
- There is not yet an end-to-end biological validation set, locked parameter
  set, sensitivity analysis, or PI-approved acceptance criterion establishing
  which outputs support a retinal-wave conclusion.

Current primary invocation:

```bash
# Run only after the main pipeline has produced per-recording metrics
python -m gcamp_analysis.waves /path/to/dataset \
  --output-dir wave_results --days 10

# Combine the resulting per-day/per-recording outputs
python scripts/summarize_wave_analysis.py wave_results
```

The present outputs can support exploratory statements about calcium activity
that is statistically consistent with spatial propagation. They do not by
themselves establish action-potential propagation, synaptic connectivity, or a
validated retinal-wave phenotype. See
[`gcamp_analysis/waves/README.md`](gcamp_analysis/waves/README.md) and
[`scripts/README.md`](scripts/README.md) for method and script-level details.
