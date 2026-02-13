"""
Train ROI classifiers (Random Forest and Logistic Regression).

Tests different feature transformations and compares model performance.
Only trains on manually labeled ROIs.
"""
import argparse
from pathlib import Path
from classifier_pipeline.run_pipeline import PipelineRunner
from classifier_pipeline.io_utils import load_config, load_roi_data, load_labeled_roi_data, save_optimization_outputs
from sklearn.model_selection import train_test_split
from classifier_pipeline.verbose_utils import print_split_summary, print_tuned_summary
from classifier_pipeline.optimize import OptimizationResults

def train_roi_classifier(config_path : Path, data_path: Path, 
                        name : str, output_dir: Path,
                        verbose: bool, manual_only: bool,
                        overwrite: bool) -> OptimizationResults:

    config = load_config(config_path)
    data = load_roi_data(data_path, manual_only=manual_only)
    labeled_data = load_labeled_roi_data(data, manual_only=manual_only)
    
    splits = train_test_split(labeled_data[0], labeled_data[1], test_size=0.2, random_state=42)

    if verbose:
        print_split_summary(splits[1], splits[3])

    runner = PipelineRunner(config, labeled_data[2], verbose=verbose)    
    results = runner.run(splits)
    if verbose:
        print_tuned_summary(results)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_optimization_outputs(results, output_dir, name, verbose=verbose, overwrite=overwrite)
        
        if verbose:
            print(f"Saved results to {output_dir}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train ROI classifier with feature transformation testing")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--data_path", type=str, 
                       default="training_data/roi_filtering/all_roi_features.npy",
                       help="Path to ROI data file")
    parser.add_argument("--output_dir", type=str,
                       default="roi_classifier/models",
                       help="Directory to save models")
    parser.add_argument("--name", type=str, default=None,
                       help="Name for the model")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output", default=True)
    parser.add_argument("-m", "--manual_only", action="store_true",
                        help="Use only manually labeled ROIs", default=True)
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite existing models of the same name", default=True)
    args = parser.parse_args()

    train_roi_classifier(
        config_path=Path(args.config),
        data_path=Path(args.data_path),
        name=args.name,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
        manual_only=args.manual_only,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
