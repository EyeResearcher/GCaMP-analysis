"""
Prepare ROI features from labels and train the ROI classifier in one go.

Usage:
  python roi_classifier/prepare_and_train.py \
    --labels training_data/roi__filtering/roi_labels.csv \
    --features_out training_data/roi__filtering/roi_features_minmax.csv \
    --model_out roi_classifier/models/roi_classifier.pkl \
    --normalization minmax
"""
import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is on sys.path when running as a script from any CWD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from roi_classifier.feature_extraction import prepare_roi_training_data
from roi_classifier.train import train_roi_classifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description='Prepare ROI features from labels and train classifier')
    ap.add_argument('--labels', type=Path, required=True, help='Path to roi_labels.csv')
    ap.add_argument('--features_out', type=Path, required=True, help='Path to write extracted features CSV')
    ap.add_argument('--model_out', type=Path, required=True, help='Path to write trained model (pkl)')
    ap.add_argument('--normalization', type=str, default='minmax', choices=['minmax', 'deltaf'])
    ap.add_argument('--test_size', type=float, default=0.15)
    args = ap.parse_args()

    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting features from labels: {args.labels}")
    features_df = prepare_roi_training_data(args.labels, args.features_out, normalization=args.normalization)
    logger.info(f"Extracted {len(features_df)} rows to {args.features_out}")

    logger.info("Training ROI classifier...")
    train_roi_classifier(args.features_out, args.model_out, test_size=args.test_size, normalization=args.normalization)
    logger.info(f"Model saved to {args.model_out}")


if __name__ == '__main__':
    main()
