# GCaMP Analysis Pipeline

A comprehensive Python pipeline for analyzing GCaMP calcium imaging data from Suite2p output. This pipeline includes ROI filtering, spike detection using Cascade, spike filtering, and neuron grouping using both STTC (Spike Time Tiling Coefficient) and DTW (Dynamic Time Warping) methods.

## Features

- **ROI Filtering**: Logistic regression classifier using 2 features (derivative_skew, spike_prom_mean)
- **Cascade Integration**: Spike inference using pre-trained Cascade models
- **Spike Filtering**: Logistic regression classifier using 8 features
- **Neuron Grouping**: Two complementary methods:
  - STTC: Spike Time Tiling Coefficient for temporal correlation
  - DTW: Dynamic Time Warping for trace similarity
- **Bad ROI Tracking**: Explicit tracking of filtered ROIs with reasons
- **Filtered Suite2p Output**: Saves filtered data with roi_mapping.csv
- **Comprehensive Excel Reports**: 5 sheets including Bad_ROIs sheet
- **Statistical Analysis**: Treatment comparisons, temporal analysis, group analysis

## Directory Structure

```
GCaMP-analysis/
├── config/
│   ├── pipeline_config.yaml      # Main pipeline configuration
│   ├── analysis_config.yaml      # Analysis settings
│   ├── experiment_config.yaml    # Experiment metadata
│   └── outputs/                  # Pipeline outputs
├── data_classes/
│   ├── experiment.py             # Experiment container
│   ├── timepoint.py              # Timepoint container
│   ├── video.py                  # Video/session container
│   ├── roi.py                    # ROI data class
│   ├── neuron.py                 # Neuron data class
│   ├── spike.py                  # Spike data class
│   ├── valley.py                 # Valley data class
│   └── neuron_group.py           # Neuron group data class
├── pipeline/
│   ├── main.py                   # Main pipeline orchestrator
│   ├── preprocessing.py          # Suite2p loading, Cascade
│   ├── roi_processing.py         # ROI feature extraction & filtering
│   ├── spike_detection.py        # Spike detection from Cascade
│   ├── spike_filtering.py        # Spike feature extraction & filtering
│   ├── neuron_grouping.py        # STTC & DTW grouping
│   └── io_handlers.py            # Excel reports, filtered Suite2p output
├── utils/
│   ├── io_utils.py               # File I/O utilities
│   ├── cascade_utils.py          # Cascade wrapper
│   ├── visualization.py          # Plotting functions
│   └── stats_utils.py            # Statistical utilities
├── analysis/
│   ├── group_analysis.py         # Group stability, coherence
│   ├── treatment_comparison.py   # Treatment effect analysis
│   └── temporal_analysis.py      # Temporal patterns, bursts, synchrony
├── roi_classifier/
│   ├── train.py                  # Train ROI classifier
│   ├── feature_extraction.py    # ROI features
│   └── models/                   # Saved models
├── spike_classifier/
│   ├── train.py                  # Train spike classifier
│   ├── prepare_training_data.py # Prepare training data
│   └── models/                   # Saved models
├── Cascade/                      # Cascade2p submodule
├── Pretrained_models/            # Pre-trained Cascade models
├── example_usage.py              # Example scripts
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Suite2p output files
- Cascade2p pre-trained models

### Install Dependencies

```bash
# Create conda environment
conda create -n gcamp-analysis python=3.9
conda activate gcamp-analysis

# Install required packages
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install openpyxl pyyaml joblib
pip install dtaidistance  # For DTW

# Install Cascade2p
cd Cascade
pip install -e .
cd ..
```

## Quick Start

### 1. Train Classifiers (First Time Only)

```python
# Train ROI classifier
python roi_classifier/train.py --training_data path/to/labeled_rois.csv

# Train spike classifier
python spike_classifier/train.py --training_data path/to/labeled_spikes.csv
```

### 2. Configure Pipeline

Edit `config/pipeline_config.yaml`:

```yaml
cascade:
  model_name: "Global_EXC_30Hz_smoothing100ms_high_noise"
  
roi_filtering:
  model_path: "roi_classifier/models/roi_classifier.pkl"
  threshold: 0.5
  
spike_filtering:
  model_path: "spike_classifier/models/spike_classifier.pkl"
  threshold: 0.5
```

### 3. Run Pipeline

```python
from pathlib import Path
import yaml
from pipeline.main import run_pipeline

# Load configuration
with open("config/pipeline_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Run pipeline
results = run_pipeline(
    suite2p_path=Path("path/to/suite2p/plane0"),
    output_dir=Path("config/outputs/my_analysis"),
    config=config
)

print(f"Good neurons: {results['good_neurons']}")
print(f"Bad ROIs: {len(results['bad_roi_indices'])}")
print(f"Total spikes: {results['total_spikes']}")
```

See `example_usage.py` for more detailed examples.

## Pipeline Steps

The pipeline performs the following steps in sequence:

1. **Load Suite2p Data**: Load F.npy, Fneu.npy, spks.npy, stat.npy, ops.npy, iscell.npy
2. **Compute Cascade Probabilities**: Run Cascade spike inference on fluorescence traces
3. **Extract ROI Features**: Compute 2 features for each ROI
4. **Filter ROIs**: Use trained classifier to identify good vs bad ROIs
5. **Detect Spikes**: Find spike peaks in Cascade probability traces
6. **Extract Spike Features**: Compute 8 features for each spike
7. **Filter Spikes**: Use trained classifier to identify good vs bad spikes
8. **Group Neurons (STTC)**: Find temporally correlated neurons
9. **Group Neurons (DTW)**: Find neurons with similar trace shapes
10. **Save Outputs**: Generate Excel reports and filtered Suite2p data

## Outputs

### Excel Report (5 sheets)

1. **Summary**: Overall statistics (neuron counts, spike rates, group counts)
2. **Neurons**: Per-neuron metrics (spike rate, features, group assignments)
3. **Groups_STTC**: STTC-based neuron groups with coherence metrics
4. **Groups_DTW**: DTW-based neuron groups with coherence metrics
5. **Bad_ROIs**: Filtered ROIs with reasons for exclusion

### Filtered Suite2p Data

```
filtered_suite2p/
└── plane0/
    ├── F.npy              # Filtered fluorescence (good neurons only)
    ├── Fneu.npy           # Filtered neuropil
    ├── spks.npy           # Filtered spikes
    ├── stat.npy           # Filtered ROI stats
    ├── iscell.npy         # Filtered cell classification
    └── roi_mapping.csv    # Maps new indices to original indices
```

## ROI Classification Features

1. **derivative_skew**: Skewness of the derivative of Cascade probability
2. **spike_prom_mean**: Mean prominence of detected spikes

## Spike Classification Features

1. **prob_height**: Height of Cascade probability at spike peak
2. **f_value**: Raw fluorescence value at spike peak
3. **prominence**: Prominence of spike peak
4. **width**: Width of spike at half-prominence
5. **derivative_at_peak**: Derivative at spike peak
6. **derivative_skew**: Skewness of derivative in spike window
7. **valley_depth**: Depth of pre-spike valley
8. **valley_area**: Area under pre-spike valley

## Neuron Grouping Methods

### STTC (Spike Time Tiling Coefficient)

- Measures temporal correlation between spike trains
- Parameters: dt (temporal window), correlation_threshold
- Good for detecting functionally coupled neurons

### DTW (Dynamic Time Warping)

- Measures similarity between fluorescence trace shapes
- Parameters: window_size, distance_threshold
- Good for detecting neurons with similar activity patterns

## Analysis Modules

### Group Analysis

```python
from analysis import analyze_group_stability, compute_group_coherence

# Analyze group stability across sessions
stability = analyze_group_stability([session1_groups, session2_groups])

# Compute coherence within a group
coherence = compute_group_coherence(group, method='cross_correlation')
```

### Treatment Comparison

```python
from analysis import compare_treatments

# Compare control vs treatment
comparison = compare_treatments(
    control_videos,
    treatment_videos,
    metric='spike_rate'
)
```

### Temporal Analysis

```python
from analysis import analyze_temporal_patterns, compute_burst_statistics

# Analyze temporal patterns
patterns = analyze_temporal_patterns(video, bin_size=1.0)

# Detect bursts
bursts = compute_burst_statistics(neurons, min_spikes=3)
```

## Configuration Options

See `config/pipeline_config.yaml` for detailed configuration options including:

- Cascade model selection
- ROI filtering thresholds
- Spike detection parameters
- Grouping method parameters
- Output options

## Training Classifiers

### ROI Classifier Training

1. Manually label ROIs (good/bad) using `roi_classifier/gui_annotator.py`
2. Extract features from labeled data
3. Train logistic regression model
4. Save model to `roi_classifier/models/`

### Spike Classifier Training

1. Manually label spikes (good/bad) using spike labeling GUI
2. Extract spike features
3. Train logistic regression model
4. Save model to `spike_classifier/models/`

## Troubleshooting

### Common Issues

1. **Missing Cascade models**: Download from Cascade2p repository
2. **Memory errors**: Process videos one at a time or reduce batch size
3. **Import errors**: Ensure all dependencies are installed
4. **Path errors**: Use absolute paths or ensure working directory is correct

### Debugging

Enable verbose logging in `pipeline_config.yaml`:

```yaml
processing:
  verbose: true
```

## Citation

If you use this pipeline, please cite:

- **Suite2p**: Pachitariu et al. (2017)
- **Cascade**: Rupprecht et al. (2021)
- **Your analysis pipeline**: [Your citation here]

## License

[Your license here]

## Contact

[Your contact information]
