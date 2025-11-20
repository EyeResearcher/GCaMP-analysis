"""
Example usage of the GCaMP analysis pipeline.
This script demonstrates how to run the complete pipeline on your data.
"""

from pathlib import Path
import yaml
from pipeline.main import run_pipeline
from data_classes import Experiment, Timepoint, Video


def run_simple_example():
    """
    Run pipeline on a single video/session.
    Expected structure: ex337/treatment/timepoint/video/suite2p/plane0/
    """
    # Define paths
    # Example: C:/Users/mzinn1/Desktop/test_ps2p/ex337/GCaMP6s_EX_Plastic_CoCl/Week 2/2-1/suite2p/plane0
    suite2p_path = Path("path/to/your/suite2p/plane0")
    output_dir = Path("config/outputs/example_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open("config/pipeline_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse directory structure to get metadata
    from utils.io_utils import parse_experiment_path
    metadata = parse_experiment_path(suite2p_path)
    
    # Create data structure
    experiment = Experiment(
        base_path=Path(metadata['experiment']),
        name=metadata['experiment'],
        treatment=metadata['treatment']
    )
    timepoint = Timepoint(
        path=suite2p_path.parent.parent,
        name=metadata['timepoint'],
        treatment=metadata['treatment']
    )
    video = Video(
        path=metadata['video_path'],
        timepoint=timepoint
    )
    
    timepoint.add_video(video)
    experiment.add_timepoint(timepoint)
    
    # Run pipeline
    print("Running GCaMP analysis pipeline...")
    results = run_pipeline(
        suite2p_path=suite2p_path,
        output_dir=output_dir,
        config=config
    )
    
    # Access results
    print(f"\nProcessing complete!")
    print(f"Total ROIs: {results['total_rois']}")
    print(f"Good neurons: {results['good_neurons']}")
    print(f"Bad ROIs filtered: {len(results['bad_roi_indices'])}")
    print(f"Total spikes detected: {results['total_spikes']}")
    print(f"STTC groups: {len(results['sttc_groups'])}")
    print(f"DTW groups: {len(results['dtw_groups'])}")
    
    print(f"\nOutputs saved to: {output_dir}")
    print("- Excel report with Bad_ROIs sheet")
    print("- Filtered Suite2p data with roi_mapping.csv")
    

def run_multi_video_example():
    """
    Run pipeline on multiple videos/sessions.
    Expected structure: ex337/treatment/timepoint/video/suite2p/plane0/
    """
    # Define base path (e.g., ex337/GCaMP6s_EX_Plastic_CoCl/)
    base_path = Path("C:/Users/mzinn1/Desktop/test_ps2p/ex337/GCaMP6s_EX_Plastic_CoCl")
    output_dir = Path("config/outputs/multi_video_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open("config/pipeline_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load entire experiment structure automatically
    from utils.io_utils import load_experiment_structure
    experiment = load_experiment_structure(base_path)
    
    print(f"Loaded experiment: {experiment.name}")
    print(f"Treatment: {experiment.treatment}")
    print(f"Timepoints: {len(experiment.timepoints)}")
    
    # Process each timepoint and video
    for timepoint in experiment.timepoints:
        print(f"\nProcessing timepoint: {timepoint.name}")
        
        for video in timepoint.videos:
            # Construct path to Suite2p data
            suite2p_path = video.path / "suite2p" / "plane0"
            
            if not suite2p_path.exists():
                print(f"  Warning: Path does not exist: {suite2p_path}")
                continue
            
            # Run pipeline for this video
            video_name = video.video_id
            video_output_dir = output_dir / timepoint.name / video_name
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Processing {video_name}...")
            results = run_pipeline(
                suite2p_path=suite2p_path,
                output_dir=video_output_dir,
                config=config
            )
            
            print(f"    Good neurons: {results['good_neurons']}")
            print(f"    Bad ROIs: {len(results['bad_roi_indices'])}")
            print(f"    Spikes: {results['total_spikes']}")
    
    # Generate experiment-level summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    exp_summary = experiment.get_summary()
    print(exp_summary)
    
    # Save experiment summary
    exp_summary.to_csv(output_dir / "experiment_summary.csv", index=False)


def run_with_analysis():
    """
    Run pipeline with post-processing analysis.
    """
    from analysis import (
        compare_treatments,
        analyze_group_stability,
        analyze_temporal_patterns
    )
    
    # First run the pipeline (as in previous examples)
    # ... (pipeline code here) ...
    
    # Then run analysis on results
    print("\nRunning post-processing analysis...")
    
    # Example: Compare control vs treatment
    # control_videos = experiment.get_timepoint("control").videos
    # treatment_videos = experiment.get_timepoint("treatment").videos
    
    # comparison = compare_treatments(
    #     control_videos,
    #     treatment_videos,
    #     metric='spike_rate'
    # )
    # print("\nTreatment comparison:")
    # print(comparison)


if __name__ == "__main__":
    print("GCaMP Analysis Pipeline Examples")
    print("=" * 50)
    print("\nChoose an example to run:")
    print("1. Simple single video analysis")
    print("2. Multi-video analysis")
    print("3. Analysis with comparisons")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        run_simple_example()
    elif choice == "2":
        run_multi_video_example()
    elif choice == "3":
        run_with_analysis()
    else:
        print("Invalid choice!")
        print("\nTo use this pipeline:")
        print("1. Update the paths in the example functions")
        print("2. Ensure your Suite2p data has the required files:")
        print("   - F.npy, Fneu.npy, spks.npy, stat.npy, ops.npy, iscell.npy")
        print("3. Train ROI and spike classifiers first (see roi_classifier/train.py)")
        print("4. Update config/pipeline_config.yaml with your model paths")
        print("5. Run this script")
