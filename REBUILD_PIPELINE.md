# Rebuild Pipeline After File Corruption

The `all_roi_features.npy` file was corrupted. Follow these steps to rebuild everything:

## Step 1: Regenerate ROI Features from Suite2p Data
```powershell
python roi_classifier/prepare_data.py --dataset_root "C:\Users\mzinn1\Desktop\Datasets"
```
This will:
- Load F.npy files from each video
- Compute CASCADE spike probabilities
- Extract ROI features (derivative_skew, spike_prom_mean, spike_prom_skew)
- Save to `training_data/roi_filtering/all_roi_features.npy`
- **No JSON files created** (prevents corruption issues)

## Step 2: Apply ROI Classifier to Label ROIs
The trained classifier (`roi_classifier.joblib`) still exists. You need to apply it to the new ROI features:
```powershell
# TODO: Find the correct script that applies the classifier
# It should load the .joblib model and predict labels for all ROIs
```

## Step 3: Extract Spike Features from Good ROIs
```powershell
python spike_classifier/prepare_data.py
```
This will:
- Load ROIs labeled as "good" (label=1)
- Detect spikes in CASCADE probabilities (handles NaN padding)
- Extract spike features (prominence, isolation, distance, skewness)
- Save to `training_data/roi_filtering/all_roi_features.npy` (updated)
- Save spike keys to `training_data/roi_filtering/all_roi_features_spike_keys.csv`

## Step 4: Annotate Spikes
```powershell
python spike_classifier/annotate_spikes.py -n 50 --unlabeled_only
```
This will:
- Load spike data from .npy file
- Show GUI for manual spike annotation
- Save annotations to .npy and .csv (no JSON)
- Checkpoint every 30 annotations

## Changes Made to Prevent Future Corruption
1. **Removed all JSON I/O** from both `roi_classifier/prepare_data.py` and `spike_classifier/prepare_data.py`
2. **Removed JSON saving** from `spike_classifier/annotate_spikes.py`
3. **Added CASCADE NaN handling** in spike detection (32 NaN values at start/end)
4. Only `.npy` and `.csv` files are saved now

## Files That Should Exist After Rebuild
- ✅ `training_data/roi_filtering/all_roi_features.npy` - All ROI data with labels and spike features
- ✅ `training_data/roi_filtering/all_roi_features_spike_keys.csv` - Spike key→label mapping
- ✅ `training_data/roi_filtering/roi_classifier.joblib` - Trained ROI classifier (already exists)
- ❌ No `.json` files (removed to prevent corruption)
