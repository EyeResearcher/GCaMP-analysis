"""Learning-curve experiments for ROI / spike classifiers.

Trains each model type (RF, LR) at increasing training-set sizes and
plots train vs. held-out test score so you can identify where
performance plateaus—i.e. the minimum number of annotations needed.

The approach:
    1. Hold out a fixed stratified test set (never touched during training).
    2. For each training fraction, draw ``n_seeds`` stratified subsamples
       from the remaining data.
    3. Fit each model type on the subsample -> score on train *and* test.
    4. Average across seeds -> plot mean +/- std bands.

Unlike the full ``PipelineRunner`` (which auto-selects model type,
transform, features, *and* tunes hyperparameters), this deliberately
fixes each model type with default hyperparameters so the curves are
comparable across training sizes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path so imports work when run as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from classifier_pipeline.io_utils import load_roi_data, load_labeled_data
from classifier_pipeline.utils import get_model, train_and_evaluate


# --- Defaults --------------------------------------------------------

_DEFAULT_FRACS: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
_DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
_DEFAULT_MODELS: Tuple[str, ...] = ("RF", "LR")
_DEFAULT_METRIC: str = "roc_auc"


# --- Core -------------------------------------------------------------

def run_learning_curves(
    data_path: Path,
    classifier_type: str,
    output_dir: Path,
    *,
    manual_only: bool = True,
    model_types: Tuple[str, ...] = _DEFAULT_MODELS,
    metric: str = _DEFAULT_METRIC,
    train_fracs: Tuple[float, ...] = _DEFAULT_FRACS,
    seeds: Tuple[int, ...] = _DEFAULT_SEEDS,
    test_size: float = 0.2,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate learning curves for one or more model types.

    Parameters
    ----------
    data_path : Path
        Path to the ``.npy`` ROI data file.
    classifier_type : {'roi', 'spike'}
        Which labelled data to extract.
    output_dir : Path
        Directory for saved plots.
    manual_only : bool
        If True, only use manually labelled samples.
    model_types : tuple of str
        Model identifiers recognised by ``get_model`` (e.g. ``'RF'``,
        ``'LR'``, ``'SVM'``).
    metric : str
        Scoring metric (``'roc_auc'``, ``'accuracy'``,
        ``'balanced_accuracy'``, ``'f1'``).
    train_fracs : tuple of float
        Fractions of the training pool to evaluate.
    seeds : tuple of int
        Random seeds for repeated subsampling.
    test_size : float
        Fraction of data held out for testing.
    verbose : bool
        Print progress updates.

    Returns
    -------
    dict[str, dict[str, ndarray]]
        ``{model_name: {'train_sizes', 'train_mean', 'train_std',
        'test_mean', 'test_std'}}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load & split --------------------------------------------------
    data = load_roi_data(data_path, verbose=verbose)
    X, y = load_labeled_data(classifier_type, data, manual_only)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if verbose:
        print(f"Total labelled samples: {len(y)}  "
              f"(class balance: {y.value_counts().to_dict()})")

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    if verbose:
        print(f"Train pool: {len(y_train_pool)}  |  Held-out test: {len(y_test)}")

    # -- Per-model learning curves -------------------------------------
    results_by_model: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name in model_types:
        if verbose:
            print(f"\n{'---'*17}\nModel: {model_name}\n{'---'*17}")

        # scores[frac] = list of (train_score, test_score) per seed
        scores: Dict[float, List[Tuple[float, float]]] = {}

        for frac in train_fracs:
            for seed in seeds:
                # Stratified subsample of the training pool
                if frac < 1.0:
                    n_sub = max(2, int(round(frac * len(y_train_pool))))
                    X_sub, _, y_sub, _ = train_test_split(
                        X_train_pool, y_train_pool,
                        train_size=n_sub,
                        random_state=seed,
                        stratify=y_train_pool,
                    )
                else:
                    X_sub, y_sub = X_train_pool, y_train_pool

                # Fit once, score on both train and held-out test
                kwargs = {"probability": True} if model_name == "SVM" else {}
                model = get_model(model_name, **kwargs)
                train_score = train_and_evaluate(
                    model, X_sub, y_sub, X_sub, y_sub, metric=metric,
                )
                test_score = train_and_evaluate(
                    model, X_sub, y_sub, X_test, y_test, metric=metric,
                )

                scores.setdefault(frac, []).append((train_score, test_score))

            if verbose:
                pairs = np.array(scores[frac])
                n = int(round(frac * len(y_train_pool)))
                print(f"  n={n:>5d}  |  train {pairs[:,0].mean():.3f} +/- {pairs[:,0].std():.3f}"
                      f"  |  test {pairs[:,1].mean():.3f} +/- {pairs[:,1].std():.3f}")

        # Aggregate across seeds
        fracs_sorted = sorted(scores.keys())
        train_sizes = np.array(
            [int(round(f * len(y_train_pool))) for f in fracs_sorted], dtype=int
        )
        train_means, train_stds, test_means, test_stds = [], [], [], []
        for frac in fracs_sorted:
            pairs = np.array(scores[frac], dtype=float)
            train_means.append(pairs[:, 0].mean())
            train_stds.append(pairs[:, 0].std(ddof=0))
            test_means.append(pairs[:, 1].mean())
            test_stds.append(pairs[:, 1].std(ddof=0))

        result = {
            "train_sizes": train_sizes,
            "train_mean": np.array(train_means),
            "train_std": np.array(train_stds),
            "test_mean": np.array(test_means),
            "test_std": np.array(test_stds),
        }
        results_by_model[model_name] = result

        _plot_curve(result, model_name, classifier_type, metric, output_dir)

    # Combined overlay plot
    if len(results_by_model) > 1:
        _plot_combined(results_by_model, classifier_type, metric, output_dir)

    return results_by_model


# --- Plotting helpers -------------------------------------------------

def _plot_curve(
    result: dict,
    model_name: str,
    classifier_type: str,
    metric: str,
    output_dir: Path,
) -> None:
    """Save a single-model learning-curve plot."""
    sizes = result["train_sizes"]
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(sizes, result["train_mean"], "o-", label="Train")
    ax.fill_between(
        sizes,
        result["train_mean"] - result["train_std"],
        result["train_mean"] + result["train_std"],
        alpha=0.15,
    )
    ax.plot(sizes, result["test_mean"], "s-", label="Test")
    ax.fill_between(
        sizes,
        result["test_mean"] - result["test_std"],
        result["test_mean"] + result["test_std"],
        alpha=0.15,
    )

    ax.set_xlabel("Training set size (n labelled samples)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{classifier_type.upper()} Learning Curve - {model_name}")
    ax.legend()
    fig.tight_layout()

    path = output_dir / f"learning_curve__{classifier_type}__{model_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_combined(
    results_by_model: Dict[str, Dict[str, np.ndarray]],
    classifier_type: str,
    metric: str,
    output_dir: Path,
) -> None:
    """Save an overlay of test-set curves from all models."""
    fig, ax = plt.subplots(figsize=(7.5, 5))

    for model_name, r in results_by_model.items():
        sizes = r["train_sizes"]
        ax.plot(sizes, r["test_mean"], "o-", label=model_name)
        ax.fill_between(
            sizes,
            r["test_mean"] - r["test_std"],
            r["test_mean"] + r["test_std"],
            alpha=0.15,
        )

    ax.set_xlabel("Training set size (n labelled samples)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{classifier_type.upper()} Test Curves - All Models")
    ax.legend()
    fig.tight_layout()

    path = output_dir / f"learning_curve__{classifier_type}__combined.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --- CLI --------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate learning curves to find minimum annotation count."
    )
    parser.add_argument(
        "--data_path", type=Path, default=Path("data/all_roi_features.npy"),
        help="Path to .npy ROI data file",
    )
    parser.add_argument(
        "--classifier_type", choices=["roi", "spike"], default="roi",
        help="Which labelled data to use",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("model_training_curves/outputs"),
        help="Directory for saved plots",
    )
    parser.add_argument(
        "--metric", default=_DEFAULT_METRIC,
        choices=["roc_auc", "accuracy", "balanced_accuracy", "f1"],
        help="Scoring metric",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(_DEFAULT_MODELS),
        help="Model types to evaluate (e.g. RF LR SVM)",
    )
    parser.add_argument(
        "--no-manual_only", dest="manual_only", action="store_false",
        help="Include auto-labelled samples",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output",
    )
    args = parser.parse_args()

    run_learning_curves(
        data_path=args.data_path,
        classifier_type=args.classifier_type,
        output_dir=args.output_dir,
        manual_only=args.manual_only,
        model_types=tuple(args.models),
        metric=args.metric,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
