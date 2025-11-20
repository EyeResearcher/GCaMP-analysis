# Directory Structure Guide

## Expected Directory Structure

The pipeline now expects the following directory structure:

```
ex337/                                    # Experiment root
└── GCaMP6s_EX_Plastic_CoCl/             # Treatment condition
    ├── Week 1/                           # Timepoint 1
    │   ├── 1-1/                         # Video 1
    │   │   └── suite2p/
    │   │       └── plane0/
    │   │           ├── F.npy
    │   │           ├── Fneu.npy
    │   │           ├── spks.npy
    │   │           ├── stat.npy
    │   │           ├── ops.npy
    │   │           └── iscell.npy
    │   ├── 1-2/                         # Video 2
    │   │   └── suite2p/plane0/...
    │   └── 1-3/                         # Video 3
    │       └── suite2p/plane0/...
    └── Week 2/                           # Timepoint 2
        ├── 2-1/                         # Video 1
        │   └── suite2p/plane0/...
        └── 2-2/                         # Video 2
            └── suite2p/plane0/...
```

## Hierarchy

1. **Experiment** (`ex337`) - Top level experiment identifier
2. **Treatment** (`GCaMP6s_EX_Plastic_CoCl`) - Treatment/condition name
3. **Timepoint** (`Week 1`, `Week 2`) - Temporal sampling points
4. **Video** (`1-1`, `1-2`, etc.) - Individual recording sessions
5. **Suite2p** - Suite2p processing output folder
6. **plane0** - Imaging plane data

## Experiment Name Generation

The `Experiment` class now combines the experiment ID and treatment:

```python
experiment = Experiment(
    base_path=Path("ex337"),
    name="ex337",  # Optional - auto-generated from path
    treatment="GCaMP6s_EX_Plastic_CoCl"
)

# Result: experiment.name = "ex337_GCaMP6s_EX_Plastic_CoCl"
```

## Automatic Structure Loading

Use the utility function to automatically discover all videos:

```python
from utils.io_utils import load_experiment_structure

# Point to the treatment folder
base_path = Path("C:/Users/mzinn1/Desktop/test_ps2p/ex337/GCaMP6s_EX_Plastic_CoCl")

# Automatically discovers all timepoints and videos
experiment = load_experiment_structure(base_path)

print(f"Experiment: {experiment.name}")  # ex337_GCaMP6s_EX_Plastic_CoCl
print(f"Treatment: {experiment.treatment}")  # GCaMP6s_EX_Plastic_CoCl
print(f"Timepoints: {len(experiment.timepoints)}")  # 2 (Week 1, Week 2)
print(f"Total videos: {experiment.get_total_neurons()}")
```

## Metadata Parsing

Each `Video` object automatically parses its location:

```python
video = Video(path=Path("ex337/GCaMP6s_EX_Plastic_CoCl/Week 1/1-1"))

# Automatically parsed:
print(video.experiment_name)  # "ex337"
print(video.treatment)        # "GCaMP6s_EX_Plastic_CoCl"
print(video.timepoint_name)   # "Week 1"
print(video.video_id)         # "1-1"
```

## Example Usage

### Single Video

```python
from pathlib import Path
from utils.io_utils import parse_experiment_path
from data_classes import Experiment, Timepoint, Video

# Your suite2p data
plane0_path = Path("C:/Users/mzinn1/Desktop/test_ps2p/ex337/GCaMP6s_EX_Plastic_CoCl/Week 2/2-1/suite2p/plane0")

# Parse directory structure
metadata = parse_experiment_path(plane0_path)

# Create structured objects
experiment = Experiment(
    base_path=Path("ex337"),
    name=metadata['experiment'],
    treatment=metadata['treatment']
)

timepoint = Timepoint(
    path=plane0_path.parent.parent,
    name=metadata['timepoint'],
    treatment=metadata['treatment']
)

video = Video(
    path=metadata['video_path'],
    timepoint=timepoint
)

experiment.add_timepoint(timepoint)
timepoint.add_video(video)
```

### Multiple Videos (Automatic Discovery)

```python
from pathlib import Path
from utils.io_utils import load_experiment_structure

# Point to treatment folder
base_path = Path("C:/Users/mzinn1/Desktop/test_ps2p/ex337/GCaMP6s_EX_Plastic_CoCl")

# Load everything automatically
experiment = load_experiment_structure(base_path)

# Process all videos
for timepoint in experiment.timepoints:
    print(f"\nTimepoint: {timepoint.name}")
    for video in timepoint.videos:
        print(f"  Video: {video.video_id}")
        suite2p_path = video.path / "suite2p" / "plane0"
        # Run pipeline on suite2p_path
```

## Output Structure

Outputs are organized by experiment, treatment, timepoint, and video:

```
config/outputs/
└── ex337_GCaMP6s_EX_Plastic_CoCl/      # Experiment + Treatment
    ├── Week 1/                          # Timepoint
    │   ├── 1-1/                        # Video
    │   │   ├── summary_report.xlsx
    │   │   └── filtered_suite2p/
    │   │       └── plane0/
    │   │           ├── F.npy
    │   │           └── roi_mapping.csv
    │   └── 1-2/
    │       └── ...
    └── Week 2/
        └── 2-1/
            └── ...
```

## Key Changes

1. **Experiment name** now includes treatment: `ex337_GCaMP6s_EX_Plastic_CoCl`
2. **Treatment field** added to `Experiment`, `Timepoint`, and `Video` classes
3. **Automatic metadata parsing** from directory structure
4. **Utility functions** for discovering and loading experiment hierarchies
5. **Path handling** respects the full 4-level hierarchy

## Migration from Old Structure

If your data has a different structure, you can:

1. **Manually specify** metadata when creating objects
2. **Update `_parse_metadata()`** in `Video` class
3. **Use custom loading** function based on your structure

## Utility Functions

- `parse_experiment_path(plane0_path)` - Extract metadata from path
- `find_suite2p_folders(base_path)` - Find all suite2p/plane0 folders
- `load_experiment_structure(base_path)` - Load entire experiment automatically
- `SummaryFiles(output_dir)` - Manage output file paths
