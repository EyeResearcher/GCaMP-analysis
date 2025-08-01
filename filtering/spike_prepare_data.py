
import pandas as pd
from pathlib import Path
import argparse

# ---- USER INPUTS ----

def parse_arguments():
    """
    Parse command line arguments for dataset root, model name, and ROI labels path.
    """
    parser = argparse.ArgumentParser(description="Process GCaMP dataset for spike filtering and feature computation.")
    parser.add_argument('-d', '--dataset_root', type=Path, default=r"C:\Users\mzinn1\Desktop\Datasets", help="Path to the dataset root folder.")
    parser.add_argument('-c','--cascade_model_name', type=str, default="Global_EXC_15Hz_smoothing100ms_high_noise", help="Name of the Cascade model to use.")
    parser.add_argument('-m', '--model_version_folder', type=Path, default=r"C:\Users\mzinn1\Desktop\Scripts\GCaMP-analysis\model_runs\GCaMP8s_Olympus_Glass", help="Path to the model version folder.")
    parser.add_argument('--new_spikes', action='store_true', help="Flag to indicate if new spikes should be found.")
    parser.add_argument('--new_model', action='store_true', help="Flag to indicate if a new model should be used for spike probability.")
    parser.add_argument('--features_only', action='store_true', help="Flag to indicate if only features should be computed without finding new spikes.")
    parser.add_argument('--annotate', action='store_true', help="Flag to indicate if spikes should be annotated.")
    parser.add_argument('-n', '--n_annotations', type=int, default=400, help="Number of annotations to perform.")
    parser.add_argument('--merge', action = 'store_true', help='Merge annotation and feature files' )
    parser.add_argument('--visualize', action='store_true', help = 'Visualize and classifications')
    parser.add_argument('--pca', action='store_true', help='Run PCA on the features')
    parser.add_argument('--pairwise', action= 'store_true' , help = 'Plot pairwise scatter plots of features')
    parser.add_argument('--compare' , action = 'store_true', help = 'Compare models')
    parser.add_argument('--errors', action='store_true', help='Visualize errors in spike detection')
    args : argparse.Namespace = parser.parse_args()
    return args 

def main():
    args = parse_arguments()
    dataset_root : Path = args.dataset_root
    model_version_folder : Path = args.model_version_folder
    roi_labels_path : Path = model_version_folder / 'roi_filtering' / 'roi_labels.csv'
    model_name : str = args.cascade_model_name
    if args.new_spikes:
        from .dataset_utils import spike_dataset_feature_computation
        new_model = args.new_model
        features_df = spike_dataset_feature_computation(dataset_root, roi_labels_path, model_name=model_name, new_model=new_model)
        features_df.to_csv(model_version_folder / 'spike_filtering' / 'spike_features.csv', index=False)
        print(f"Computed features for {len(features_df)} spikes.")
    elif args.features_only:
        from .dataset_utils import features_only
        features_df = pd.read_csv(model_version_folder / 'spike_filtering' / 'spike_features.csv')
        features_df_new = features_only(features_df)
        features_df_new.to_csv(model_version_folder / 'spike_filtering' / 'spike_features.csv', index=False)
        print(f"Computed features for {len(features_df_new)} spikes.")
    if args.annotate:
        from .spike_annotation import main_annotate
        features_df = pd.read_csv(model_version_folder / 'spike_filtering' / 'spike_features.csv')
        annotated_df : pd.DataFrame = main_annotate(features_df, n_annotations=args.n_annotations)
        annotated_df.to_csv(model_version_folder / 'spike_filtering' / 'spike_annotations.csv', index=False)
        print(f"Annotation complete and saved to {model_version_folder / 'spike_filtering' / 'spike_annotations.csv'}")
    if args.merge:
        annotations = pd.read_csv(model_version_folder / 'spike_filtering' / 'spike_annotations.csv')
        annotations.to_csv(model_version_folder / 'spike_filtering' / 'spike_features.csv')
    if args.visualize:
        from .spike_visualize import main_plot
        features_df = pd.read_csv(model_version_folder / 'spike_filtering' / 'spike_features.csv')
        annotations_df = pd.read_csv(model_version_folder / 'spike_filtering' / 'spike_annotations.csv')
        main_plot(features_df, annotations_df,pairwise = args.pairwise, pca = args.pca)
    if args.compare:
        from .spike_compare_models import compare_models
        features_df = pd.read_csv(model_version_folder / 'spike_filtering' / 'spike_features.csv')
        annotations_df = pd.read_csv(model_version_folder / 'spike_filtering' / 'spike_annotations.csv')
        merged_df = pd.merge(annotations_df, features_df, on='spike_key', how='left')
        compare_models(merged_df)
    if args.errors:
        from .spike_error_visualize import train_and_visualize_rf
        features_df = pd.read_csv(model_version_folder / 'spike_filtering' / 'spike_features.csv')
        annotations_df = pd.read_csv(model_version_folder / 'spike_filtering' / 'spike_annotations.csv')
        merged_df = pd.merge(annotations_df, features_df, on='spike_key', how='left')
        train_and_visualize_rf(merged_df)
if __name__ == "__main__":
    main()