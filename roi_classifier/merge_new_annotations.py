"""
Merge newly annotated ROI labels CSV files into the master roi_labels.csv.

Each input CSV must have columns: source_file,roi_index,label
Duplicates are resolved by keeping the LAST occurrence (newer annotation wins).

Usage:
  python roi_classifier/merge_new_annotations.py \
      --master training_data/roi__filtering/roi_labels.csv \
      --new annotations/session1_rois.csv annotations/session2_rois.csv
"""
import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description='Merge new ROI annotation CSVs into master labels file')
    ap.add_argument('--master', type=Path, required=True, help='Path to existing (or new) master roi_labels.csv')
    ap.add_argument('--new', type=Path, nargs='+', required=True, help='One or more new annotation CSV files to merge')
    args = ap.parse_args()

    # Load existing master or create empty
    if args.master.exists():
        master_df = pd.read_csv(args.master)
        logger.info(f"Loaded master with {len(master_df)} rows")
    else:
        master_df = pd.DataFrame(columns=['source_file','roi_index','label'])
        logger.info("Master file does not exist; starting fresh")

    # Append new annotation files
    added_total = 0
    for new_file in args.new:
        if not new_file.exists():
            logger.warning(f"Skipping missing file: {new_file}")
            continue
        df_new = pd.read_csv(new_file)
        required_cols = {'source_file','roi_index','label'}
        if not required_cols.issubset(df_new.columns):
            logger.warning(f"File {new_file} missing required columns; skipping")
            continue
        before = len(master_df)
        master_df = pd.concat([master_df, df_new], ignore_index=True)
        added_total += len(df_new)
        logger.info(f"Merged {len(df_new)} rows from {new_file}")

    # Deduplicate: keep last label per (source_file, roi_index)
    master_df.sort_values(by=['source_file','roi_index']).drop_duplicates(subset=['source_file','roi_index'], keep='last', inplace=True)

    # Save
    args.master.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(args.master, index=False)
    logger.info(f"Saved merged labels to {args.master} (total unique rows: {len(master_df)}, added raw rows: {added_total})")

if __name__ == '__main__':
    main()
