"""Train ROI classifier.

Thin wrapper around the generic ``classifier_pipeline.main.train_classifier``
with ``classifier_type="roi"``.
"""
import argparse
from pathlib import Path

from classifier_pipeline.train_classifier import train
from classifier_pipeline.optimize import OptimizationResults


def train_roi_classifier(
    config_path: Path,
    data_path: Path,
    name: str,
    output_dir: Path = None,
    verbose: bool = True,
    manual_only: bool = True,
    **kwargs,
) -> OptimizationResults:
    """Train an ROI classifier. See :func:`classifier_pipeline.train_classifier.train`."""
    return train(
        config_path=config_path,
        data_path=data_path,
        name=name,
        classifier_type="roi",
        output_dir=output_dir,
        verbose=verbose,
        manual_only=manual_only,
    )


def main():
    parser = argparse.ArgumentParser(description="Train ROI classifier")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--data_path", type=str,
                       default="data/all_roi_features.npy",
                       help="Path to ROI data file")
    parser.add_argument("--output_dir", type=str,
                       default="roi_classifier/models",
                       help="Directory to save models")
    parser.add_argument("--name", type=str, default=None,
                       help="Name for the model")
    parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction,
                        help="Enable verbose output", default=True)
    parser.add_argument("-m", "--manual_only", action=argparse.BooleanOptionalAction,
                        help="Use only manually labeled ROIs", default=True)
    args = parser.parse_args()

    train_roi_classifier(
        config_path=Path(args.config),
        data_path=Path(args.data_path),
        name=args.name,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
        manual_only=args.manual_only,
    )


if __name__ == "__main__":
    main()
