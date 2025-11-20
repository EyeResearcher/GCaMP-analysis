# Excel Report Structure

## Video-Level Analysis (analysis.xlsx)

Located in each video's `metrics/` folder. Contains detailed per-neuron and per-group analysis.

### Sheet 1: Neuron_Summary
**Format:** One row per neuron

**Columns:**
- `original_roi_index` - ROI index from Suite2p
- `n_spikes` - Total number of spikes detected
- `spike_frequency_hz` - Spike rate in Hz
- `avg_amplitude` - Mean spike amplitude
- `var_amplitude` - Variance of spike amplitudes
- `avg_decay_constant` - Mean decay time constant
- `var_decay_constant` - Variance of decay constants
- `avg_rise_constant` - Mean rise time constant
- `var_rise_constant` - Variance of rise constants

### Sheet 2: Supplementary_Data
**Format:** One row per neuron with per-spike arrays

**Columns:**
- `original_roi_index` - ROI index from Suite2p
- `spike_indices` - Array of frame indices for each spike
- `spike_times` - Array of spike times
- `amplitudes` - Array of individual spike amplitudes
- `decay_constants` - Array of decay time constants per spike
- `rise_constants` - Array of rise time constants per spike
- `cascade_probabilities` - Array of CASCADE probability values

**Note:** Arrays are stored as strings for Excel compatibility. Parse with `eval()` or `ast.literal_eval()` in Python.

### Sheet 3: Group_Analysis
**Format:** One row per group (STTC or DTW)

**Columns:**
- `group_id` - Unique identifier (e.g., "STTC_0", "DTW_1")
- `group_type` - "STTC" or "DTW"
- `n_neurons` - Number of neurons in group
- `neuron_indices` - Array of neuron ROI indices in group
- `total_spikes` - Total spikes across all neurons in group
- `avg_amplitude` - Mean amplitude across all spikes in group
- `var_amplitude` - Variance of amplitudes in group
- `avg_decay_constant` - Mean decay constant in group
- `var_decay_constant` - Variance of decay constants in group
- `avg_rise_constant` - Mean rise constant in group
- `var_rise_constant` - Variance of rise constants in group
- `mean_sttc` - Mean STTC correlation within group (STTC groups only)
- `var_sttc` - Variance of STTC values within group
- `mean_dtw` - Mean DTW distance within group (DTW groups only)
- `var_dtw` - Variance of DTW distances within group
- `n_shared_spikes` - Number of synchronous spikes (within ±1 frame)

### Sheet 4: Bad_ROIs
**Format:** One row per rejected ROI

**Columns:**
- `roi_index` - Suite2p ROI index that was filtered out
- `reason` - Why it was rejected (e.g., "ROI classifier rejection")
- `derivative_skew` - Feature value (if available)
- `spike_prom_mean` - Feature value (if available)

### Sheet 5: ROI_Filter_Summary
**Format:** Single row with overall statistics

**Columns:**
- `total_rois` - Total ROIs from Suite2p
- `good_rois` - ROIs that passed classifier
- `bad_rois` - ROIs that were filtered out
- `filter_rate` - Percentage filtered (%)
- `retention_rate` - Percentage retained (%)
- `neurons_with_spikes` - Good ROIs with ≥1 spike
- `total_spikes` - Total spikes detected
- `sttc_groups` - Number of STTC groups found
- `dtw_groups` - Number of DTW groups found

---

## Timepoint-Level Summary ({timepoint_name}_video_summary.xlsx)

Located in the treatment folder. Contains video-level aggregates for each video in the timepoint.

**Format:** One row per video

**Columns:**
- `video_id` - Video folder name
- `treatment` - Treatment condition
- `n_neurons` - Number of neurons in video
- `total_spikes` - Total spikes across all neurons
- `avg_spike_frequency_hz` - Mean spike frequency across neurons
- `var_spike_frequency_hz` - Variance of spike frequencies
- `avg_amplitude` - Mean amplitude across all spikes
- `var_amplitude` - Variance of amplitudes
- `avg_decay_constant` - Mean decay constant
- `var_decay_constant` - Variance of decay constants
- `avg_rise_constant` - Mean rise constant
- `var_rise_constant` - Variance of rise constants
- `n_sttc_groups` - Number of STTC groups in video
- `n_dtw_groups` - Number of DTW groups in video

---

## Experiment-Level Summary (experiment_summary.xlsx)

Located in the treatment folder. Aggregates data across all timepoints and videos.

### Sheet 1: All_Data
**Format:** All neuron data combined

Contains all neuron-level data from all videos/timepoints combined, with additional columns:
- `timepoint` - Which timepoint this neuron is from
- `treatment` - Treatment condition
- `video_id` - Which video this neuron is from

### Sheet 2: Summary_Stats
**Format:** Grouped statistics

Statistics grouped by timepoint and treatment:
- `n_spikes` - mean, std, count
- `spike_frequency` - mean, std

---

## Usage Examples

### Python
```python
import pandas as pd

# Read video analysis
df = pd.read_excel('path/to/analysis.xlsx', sheet_name='Neuron_Summary')

# Read supplementary data
supp = pd.read_excel('path/to/analysis.xlsx', sheet_name='Supplementary_Data')

# Parse spike arrays
import ast
spike_indices = ast.literal_eval(supp.loc[0, 'spike_indices'])
amplitudes = ast.literal_eval(supp.loc[0, 'amplitudes'])

# Read group analysis
groups = pd.read_excel('path/to/analysis.xlsx', sheet_name='Group_Analysis')
sttc_groups = groups[groups['group_type'] == 'STTC']
```

### Excel
- Open files directly in Excel
- Use Data > Text to Columns to parse array columns
- Use PivotTables for custom aggregations
- Filter by treatment, timepoint, or group_type

---

## Notes

1. **Array Storage:** Spike-level arrays in Supplementary_Data are stored as strings. Parse them in your analysis code.

2. **Missing Values:** NaN indicates insufficient data (e.g., no decay constants fit for those spikes).

3. **Shared Spikes:** Counted as spikes occurring within ±1 frame across multiple neurons in a group.

4. **STTC vs DTW:** 
   - STTC groups have `mean_sttc` and `var_sttc` populated
   - DTW groups have `mean_dtw` and `var_dtw` populated
   - The other metric will be NaN

5. **Filtering:** Check the Bad_ROIs sheet to understand which ROIs were excluded and why.
