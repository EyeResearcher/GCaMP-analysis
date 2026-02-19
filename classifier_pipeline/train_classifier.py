"""
Generic classifier training entry point.

Provides a single `train_classifier` function that handles both ROI and spike
classifiers, parameterized by a data-loader callable.
"""
from pathlib import Path
from typing import Callable
import pandas as pd

from .run_pipeline import PipelineRunner
from .optimize import OptimizationResults
from .io_utils import load_roi_data, load_labeled_data, save_optimization_outputs
from utils.io_utils import load_config         
from .verbose_utils import print_tuned_summary, print_dataset_summary
from sklearn.model_selection import train_test_split





def train(
    config_path: Path,
    data_path: Path,
    name: str,
    classifier_type: str,
    output_dir: Path = None,
    verbose: bool = True,
    manual_only: bool = True,
) -> OptimizationResults:
    """
    Train a classifier (ROI or spike) end-to-end.

    Loads data via the appropriate loader, splits train/test, runs the
    hyperparameter-search pipeline, and optionally saves the results.

    Parameters
    ----------
    config_path : Path
        Path to the hyperparameter YAML config file.
    data_path : Path
        Path to the .npy data file containing ROI dict.
    name : str
        Model name used for saved filenames.
    classifier_type : str
        ``"roi"`` or ``"spike"`` — selects the data loader.
    output_dir : Path, optional
        Directory to save model + results JSON. None skips saving.
    verbose : bool, optional
        Print progress, by default True.
    manual_only : bool, optional
        Use only manually-labeled samples, by default True.

    Returns
    -------
    results : OptimizationResults
        Trained model and evaluation metrics.

    Raises
    ------
    ValueError
        If *classifier_type* is not ``"roi"`` or ``"spike"``.
    """


    config = load_config(config_path)
    data = load_roi_data(data_path)
    x, y = load_labeled_data(classifier_type, data, manual_only)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    if verbose:
        print_dataset_summary(y_train, y_test, manual_only=manual_only)

    runner = PipelineRunner(config, verbose=verbose)
    results = runner.run(x_train, x_test, y_train, y_test)

    if verbose:
        print_tuned_summary(results)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_optimization_outputs(results, output_dir, name, verbose=verbose)
        if verbose:
            print(f"Saved results to {output_dir}")

    return results

