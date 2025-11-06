# Quick Start Guide - GCaMP Analysis Pipeline

This guide will help you get started with the GCaMP analysis pipeline in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- Suite2p output files from your imaging session
- Labeled training data for ROI and spike classifiers (optional for first run)

## Step 1: Installation (2 minutes)

```bash
# Navigate to the project directory
cd GCaMP-analysis

# Install dependencies
pip install -r requirements.txt

# Install Cascade2p
cd Cascade
pip install -e .
cd ..
```

## Step 2: First-Time Setup (Optional - 10 minutes)

If you have labeled training data, train the classifiers:

```bash
# Train ROI classifier
python roi_classifier/train.py --training_data path/to/roi_labels.csv

# Train spike classifier  
python spike_classifier/train.py --training_data path/to/spike_labels.csv
```

**Note**: If you don't have training data yet, you can use the GUI annotators:
- ROI labeling: `python roi_classifier/gui_annotator.py`
- Spike labeling: See `spike_classifier/prepare_training_data.py`

## Step 3: Configure Pipeline (1 minute)

Edit `config/pipeline_config.yaml` to set your model paths:

```yaml
roi_filtering:
  model_path: "roi_classifier/models/roi_classifier.pkl"
  threshold: 0.5

spike_filtering:
  model_path: "spike_classifier/models/spike_classifier.pkl"
  threshold: 0.5

cascade:
  model_name: "Global_EXC_30Hz_smoothing100ms_high_noise"
```

## Step 4: Run Pipeline (1 minute)

Create a simple Python script or use `example_usage.py`:

```python
from pathlib import Path
import yaml
from pipeline.main import run_pipeline

# Load configuration
with open("config/pipeline_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Define your Suite2p data path
suite2p_path = Path("path/to/your/suite2p/plane0")
output_dir = Path("config/outputs/my_first_run")

# Run the pipeline
results = run_pipeline(
    suite2p_path=suite2p_path,
    output_dir=output_dir,
    config=config
)

# Print results
print(f"✓ Total ROIs: {results['total_rois']}")
print(f"✓ Good neurons: {results['good_neurons']}")
print(f"✓ Filtered out: {len(results['bad_roi_indices'])}")
print(f"✓ Total spikes: {results['total_spikes']}")
print(f"✓ STTC groups: {len(results['sttc_groups'])}")
print(f"✓ DTW groups: {len(results['dtw_groups'])}")
```

Run it:

```bash
python your_script.py
```

## Step 5: Check Outputs (1 minute)

Your results are saved in `config/outputs/my_first_run/`:

### Excel Report (`summary_report.xlsx`)
- **Summary** sheet: Overall statistics
- **Neurons** sheet: Per-neuron metrics and features
- **Groups_STTC** sheet: STTC-based neuron groups
- **Groups_DTW** sheet: DTW-based neuron groups
- **Bad_ROIs** sheet: Filtered ROIs with reasons

### Filtered Suite2p Data (`filtered_suite2p/plane0/`)
- `F.npy` - Fluorescence (good neurons only)
- `Fneu.npy` - Neuropil fluorescence
- `spks.npy` - Suite2p spikes
- `stat.npy` - ROI statistics
- `iscell.npy` - Cell classification
- `roi_mapping.csv` - **Maps filtered → original indices**

## What's Happening Under the Hood?

The pipeline performs these steps:

1. ✓ Loads Suite2p data
2. ✓ Computes Cascade spike probabilities
3. ✓ Extracts ROI features (2 features)
4. ✓ Filters ROIs (tracks bad ones)
5. ✓ Detects spikes from Cascade
6. ✓ Extracts spike features (8 features)
7. ✓ Filters spikes
8. ✓ Groups neurons by STTC (temporal correlation)
9. ✓ Groups neurons by DTW (trace similarity)
10. ✓ Generates outputs

## Common Issues

### "Model not found" Error
**Solution**: Train the classifiers first or update the model paths in `config/pipeline_config.yaml`

### "Cascade model not found" Error
**Solution**: Download pre-trained models from Cascade2p repository to `Pretrained_models/`

### Out of Memory Error
**Solution**: Process one video at a time or reduce batch size in config

### Import Errors
**Solution**: Make sure all dependencies are installed: `pip install -r requirements.txt`

## Next Steps

### Analyze Multiple Videos

See `example_usage.py` for multi-video analysis:

```python
python example_usage.py
# Choose option 2 for multi-video analysis
```

### Compare Treatments

```python
from analysis import compare_treatments

comparison = compare_treatments(
    control_videos,
    treatment_videos,
    metric='spike_rate'
)
print(comparison)
```

### Analyze Temporal Patterns

```python
from analysis import analyze_temporal_patterns

patterns = analyze_temporal_patterns(video, bin_size=1.0)
print(patterns)
```

### Analyze Group Stability

```python
from analysis import analyze_group_stability

stability = analyze_group_stability([session1_groups, session2_groups])
print(stability)
```

## Tips for Success

1. **Start with one video** to verify everything works
2. **Check the Bad_ROIs sheet** to understand what's being filtered
3. **Use roi_mapping.csv** to map between filtered and original indices
4. **Adjust thresholds** in `config/pipeline_config.yaml` based on your data
5. **Visualize results** using the functions in `utils/visualization.py`

## Getting Help

- Read the full documentation: `README.md`
- Check implementation details: `IMPLEMENTATION_SUMMARY.md`
- Review example code: `example_usage.py`
- Check configuration options: `config/*.yaml`

## File Checklist

Before running, make sure your Suite2p folder has:
- ✓ `F.npy` - Fluorescence traces
- ✓ `Fneu.npy` - Neuropil fluorescence
- ✓ `spks.npy` - Suite2p spike inference
- ✓ `stat.npy` - ROI statistics
- ✓ `ops.npy` - Suite2p options
- ✓ `iscell.npy` - Cell classification

## Summary

You should now be able to:
1. ✓ Install the pipeline
2. ✓ Configure it for your data
3. ✓ Run analysis on Suite2p outputs
4. ✓ Get filtered Suite2p data with roi_mapping.csv
5. ✓ Access comprehensive Excel reports with Bad_ROIs tracking
6. ✓ Analyze neuron groups using STTC and DTW methods

**Total time: ~5 minutes** (excluding training data collection)

Enjoy analyzing your GCaMP data! 🧠✨
