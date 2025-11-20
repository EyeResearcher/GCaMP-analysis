# ROI Classifier Normalization Strategy

## Overview

The ROI classifier has been refactored to normalize fluorescence and spike probability data **per-video** before computing features. This controls for videos with different brightness levels and ensures features are on consistent scales.

## Changes Made

### 1. Feature Extraction (`roi_classifier/feature_extraction.py`)

**New Normalization Functions:**
- `normalize_minmax(trace)` - Scales trace to [0, 1] range
- `normalize_deltaf_f(trace)` - Computes ΔF/F: (F_i - F_{i-1}) / F_i

**Updated `extract_roi_features()`:**
- Now accepts `normalization` parameter ('minmax' or 'deltaf')
- Normalizes F trace BEFORE computing derivative_skew
- Normalizes cascade_prob to [0, 1] BEFORE computing spike_prom_mean
- Features are now on consistent scales across videos

### 2. Training Script (`roi_classifier/train.py`)

**Enhanced Training:**
- Uses 85% of data for training (was 80%)
- Changed from MinMaxScaler to StandardScaler (features already normalized)
- Added `class_weight='balanced'` to handle class imbalance
- Includes 5-fold cross-validation
- Detailed performance metrics and feature coefficients
- Tracks normalization strategy used

### 3. Pipeline Integration (`pipeline/roi_processing.py`)

**Updated `extract_roi_features()`:**
- Accepts `normalization` parameter
- Applies same per-video normalization as training
- Ensures exact match between training and inference

### 4. Configuration (`config/pipeline_config.yaml`)

**New Section:**
```yaml
roi_filtering:
  normalization: "minmax"  # Options: "minmax" or "deltaf"
```

### 5. Main Pipeline (`pipeline/main.py`)

**Updated ROI Filtering:**
- Reads normalization strategy from config
- Passes to `extract_roi_features()`
- Logs which normalization is being used

## Two Normalization Strategies

### MinMax Normalization
**Formula:** `(F - F_min) / (F_max - F_min)`

**Advantages:**
- Simple and interpretable
- Maps all traces to [0, 1] range
- Preserves relative scale of signals
- Controls for overall brightness differences

**Use When:**
- Videos have different baseline brightness
- Want consistent feature ranges
- Simpler interpretation needed

### DeltaF/F Normalization
**Formula:** `(F_i - F_{i-1}) / F_i`

**Advantages:**
- Standard in calcium imaging
- Normalizes by current fluorescence
- Emphasizes changes relative to current state
- Handles drift better

**Use When:**
- Following traditional calcium imaging conventions
- Baseline drift is problematic
- Want to emphasize relative changes

## Comparing Strategies

### Run Comparison Script:
```bash
python compare_roi_normalization.py
```

This script will:
1. Extract features with both normalization strategies
2. Train separate models for each
3. Compare performance metrics
4. Provide recommendations

### Expected Output:
```
COMPARISON SUMMARY
==================================================================================
Normalization  Test Accuracy  Train Accuracy  CV Mean  CV Std  Good Precision  Good Recall  Good F1
MINMAX         0.XXX          0.XXX           0.XXX    0.XXX   0.XXX          0.XXX        0.XXX
DELTAF         0.XXX          0.XXX           0.XXX    0.XXX   0.XXX          0.XXX        0.XXX
```

## Training with More Data

The training script now uses:
- **85% training set** (increased from 80%)
- **15% test set** (decreased from 20%)
- **5-fold cross-validation** during training
- **Balanced class weights** to handle imbalance

This ensures the model learns from a substantial portion of the annotated data while still having a held-out test set for validation.

## Usage

### 1. Extract Features (from labels):
```bash
# MinMax normalization
python -c "from pathlib import Path; from roi_classifier.feature_extraction import prepare_roi_training_data; prepare_roi_training_data(Path('training_data/roi__filtering/roi_labels.csv'), Path('training_data/roi__filtering/roi_features_minmax.csv'), normalization='minmax')"

# DeltaF/F normalization
python -c "from pathlib import Path; from roi_classifier.feature_extraction import prepare_roi_training_data; prepare_roi_training_data(Path('training_data/roi__filtering/roi_labels.csv'), Path('training_data/roi__filtering/roi_features_deltaf.csv'), normalization='deltaf')"
```

### 2. Train Classifier:
```bash
# MinMax
python roi_classifier/train.py --features training_data/roi__filtering/roi_features_minmax.csv --output roi_classifier/models/roi_classifier_minmax.pkl --normalization minmax

# DeltaF/F
python roi_classifier/train.py --features training_data/roi__filtering/roi_features_deltaf.csv --output roi_classifier/models/roi_classifier_deltaf.pkl --normalization deltaf
```

### 3. Update Config:
```yaml
roi_filtering:
  normalization: "minmax"  # or "deltaf"
```

### 4. Copy Model:
```bash
# If using minmax
cp roi_classifier/models/roi_classifier_minmax.pkl roi_classifier/models/roi_classifier.pkl

# If using deltaf
cp roi_classifier/models/roi_classifier_deltaf.pkl roi_classifier/models/roi_classifier.pkl
```

### 5. Run Pipeline:
```bash
python pipeline/main.py --config config/pipeline_config.yaml --experiment /path/to/data
```

## Key Benefits

1. **Per-Video Normalization**: Features are computed on normalized data, controlling for video-to-video brightness differences

2. **Consistent Scales**: 
   - `spike_prom_mean` now always in [0, 1] range
   - `derivative_skew` computed on standardized data

3. **Better Generalization**: Model trained on normalized features should generalize better to new videos with different brightness levels

4. **Matched Training/Inference**: Exact same normalization applied during training and pipeline execution

5. **Flexibility**: Easy to switch between normalization strategies via config

## Feature Interpretation

### After Normalization:

**derivative_skew:**
- Computed on normalized F trace (smoothed with σ=4.0)
- Measures asymmetry of fluorescence changes
- Positive: More sharp increases (neuron-like)
- Negative: More sharp decreases (artifact-like)

**spike_prom_mean:**
- Computed on normalized CASCADE probability [0, 1]
- Mean prominence of detected peaks
- Higher values: Clear, prominent spikes
- Lower values: Weak or noisy signals

## Recommendations

1. **Start with MinMax**: Simpler and more intuitive
2. **Run Comparison**: Use `compare_roi_normalization.py` to test both
3. **Check Performance**: Compare accuracy, precision, recall
4. **Choose Best**: Select strategy with better performance
5. **Document Choice**: Note which normalization was used for reproducibility

## Files Modified

- `roi_classifier/feature_extraction.py` - Added normalization functions
- `roi_classifier/train.py` - Enhanced training with more data
- `pipeline/roi_processing.py` - Added per-video normalization
- `pipeline/main.py` - Read normalization from config
- `config/pipeline_config.yaml` - Added normalization setting
- `compare_roi_normalization.py` - NEW: Compare strategies

## Next Steps

1. Run comparison script to determine best normalization
2. Train final model with chosen strategy
3. Update config with chosen normalization
4. Test on full dataset
5. Evaluate ROI filtering quality
