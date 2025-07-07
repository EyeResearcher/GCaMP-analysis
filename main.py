#import runpy
#runpy.run_path("C:/Users/mzinn1/Desktop/Scripts/GCaMP-analysis/sitecustomize.py")

import os, sys
import numpy

try:
    from data_classes.experiment import Experiment
    print("experiment class imported successfully")
except Exception as e:
    import traceback
    print("Error importing Experiment:")
    traceback.print_exc()
    raise
import tensorflow as tf

from pathlib import Path

import pandas as pd


from Cascade.cascade2p.cascade_wrapper import CascadePredictor

import argparse
from joblib import load

def main():
    """
    Main entry point to run the full experiment pipeline.
    """
    

    parser = argparse.ArgumentParser(description='Run Suite2p analysis pipeline')
    parser.add_argument('experiment_path', help='Path to the experiment folder containing timepoints')
    parser.add_argument('roi_model_path', help='Path to the trained ROI classifier model (joblib file)')
    parser.add_argument('--cascade_model_name', default='Global_EXC_30Hz_smoothing100ms_high_noise', help='Name of pretrained cascade model')
    parser.add_argument('--fs', type=float, default=30.0, help='Sampling frequency in Hz')
    args = parser.parse_args()

    # Debug prints to confirm invocation and arguments
    print(f"[DEBUG] Running main with args: experiment_path={args.experiment_path}, roi_model_path={args.roi_model_path}, fs={args.fs}")
    print(f"[DEBUG] Python executable: {sys.executable}")
    print(f"[DEBUG] Working directory: {os.getcwd()}")

    # Load ROI classification model
    roi_model = load(args.roi_model_path)
    #Download and load cascade model
    project_root = Path(__file__).parent
    cascade_models_dir = project_root / "Cascade" / "Pretrained_models"
    cascade_models_dir.mkdir(parents=True, exist_ok=True)
    
    cascade_model = CascadePredictor(
        model_name=args.cascade_model_name,
        model_folder=str(cascade_models_dir)
    )

    # 2) download (or update) the specific model
    

    # Initialize and run experiment
    exp = Experiment(args.experiment_path, roi_model, cascade_model,  fs = args.fs)
    exp.process_all_timepoints()

    # Save per-timepoint summaries
    for tp in exp.timepoints:
        out_file = tp.save_summary_excel()
        print(f"Saved timepoint summary to {out_file}")

    # Optionally, aggregate into a single DataFrame
    agg_df = exp.aggregate_summary()
    agg_path = Path(args.experiment_path) / f"{exp.name}_aggregate_summary.xlsx"
    with pd.ExcelWriter(agg_path) as writer:
        agg_df.to_excel(writer, sheet_name='Aggregate_Summary')
    print(f"Saved aggregate summary to {agg_path}")

if __name__ == '__main__':
    print("Running main.py as a script")
    main()