"""
Main entry point for the classifier pipeline.
"""
from pathlib import Path
from typing import Any
import numpy as np

from .run_pipeline import PipelineRunner
from .optimize import OptimizationResults
from .io_utils import load_labeled_roi_data, save_optimization_outputs
from sklearn.model_selection import train_test_split
from .verbose_utils import print_tuned_summary, print_split_summary


def get_best_model(
    config: dict,
    data: tuple[np.ndarray, np.ndarray, list[str], list[Any]],
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
    config : str | Path
        Path to the hyperparameter config file (YAML/JSON)
    data : str | Path
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
    x, y, feat_names, keys = load_labeled_roi_data(data, manual_only=manual_only)
    splits = train_test_split(x, y, test_size=0.2, random_state=42)
    
    if verbose:
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

