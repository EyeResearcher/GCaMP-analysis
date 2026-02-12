"""
Main entry point for the classifier pipeline.
"""
from pathlib import Path
from typing import Any
import numpy as np

from .run_pipeline import PipelineRunner
from .optimize import OptimizationResults
from .io_utils import load_labeled_roi_data, load_roi_data, load_config, save_optimization_outputs
from sklearn.model_selection import train_test_split
from .verbose_utils import print_tuned_summary, print_split_summary
def run_classifier_pipeline(
    config_path: str | Path,
    data_path: str | Path,
    name : str ,
    output_dir: str | Path = None,
    verbose: bool = True,
    manual_only = True,
    overwrite = False
) -> OptimizationResults:
    """
    Run the full classifier pipeline from config and data files.
    
    Parameters
    ----------
    config_path : str | Path
        Path to the hyperparameter config file (YAML/JSON)
    data_path : str | Path
        Path to the data file (NPY/CSV)
    output_dir : str | Path, optional
        Directory to save results, by default None (no saving)
    verbose : bool, optional
        Whether to print progress, by default True
        
    Returns
    -------
    results : OptimizationResults
        Final optimization results
    """
    # Load config and data
    config = load_config(config_path)
    data = load_roi_data(data_path)
    x, y, feat_names, keys = load_labeled_roi_data(data, manual_only=manual_only)
    splits = train_test_split(x, y, test_size=0.2, random_state=42)
    
    if verbose:
        print(f"Loaded config from {config_path}")
        print(f"Loaded data with {len(feat_names)} features")
        print_split_summary(splits[1], splits[3])
    
    # Run pipeline
    runner = PipelineRunner(config, feat_names, verbose=verbose)
    results = runner.run(splits)

    if verbose:
        print_tuned_summary(results)

    # Save results if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_optimization_outputs(results, output_dir, name, verbose=verbose, overwrite=overwrite)
        
        if verbose:
            print(f"Saved results to {output_dir}")
    
    return results


def main():
    """
    CLI entry point.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GCaMP classifier pipeline")
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument("--data", "-d", required=True, help="Path to data file")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    results = run_classifier_pipeline(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    print(f"\nBest model: {results.model.__class__.__name__}")
    print(f"Test accuracy: {results.test_acc:.4f}")
    print(f"ROC AUC: {results.roc_auc:.4f}")


if __name__ == "__main__":
    main()