import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
# 1) Expand your 4×3 columns into a flat DataFrame:
#    e.g. df[['feat1_raw','feat2_raw','feat3_raw',
#              'feat1_zroi', … , 'feat3_zroi_global']]
#    and your df['label'] is already “real”/“not real”.
def unpack_rep(df : pd.DataFrame, rep_col, name_col, prefix):
    """
    Unpack a list-valued column (`rep_col`) with corresponding feature names (`name_col`)
    into a wide DataFrame with one column per feature, prefixed by `prefix`.
    """
    # Select only the relevant columns and explode both name and value
    tmp = df[['spike_key', 'label', name_col, rep_col]].copy()
    tmp = tmp.explode([name_col, rep_col])
    tmp = tmp.rename(columns={name_col: 'feature', rep_col: 'value'})

    # Pivot to wide form
    wide = tmp.pivot(index=['spike_key', 'label'], columns='feature', values='value')

    # Flatten column index and add prefix
    wide.columns = [f"{prefix}__{feat}" for feat in wide.columns]
    return wide.reset_index()
def expand_feature_representations(features_df):
    """
    Given a DataFrame with list-valued feature representation columns,
    returns a new DataFrame where each representation is expanded to wide form.
    """
    # Expand each representation
    raw_wide   = unpack_rep(features_df, 'raw_features',     'feature_names', 'raw')
    zroi_wide  = unpack_rep(features_df, 'raw_z_features',    'feature_names', 'zroi')
    zglob_wide = unpack_rep(features_df, 'z_scored_raw_features', 'feature_names', 'zglob')
    zz_wide    = unpack_rep(features_df, 'z_scored_z_features',      'feature_names', 'zz')

    # Merge all representations on spike_key and label
    df_merged = raw_wide.merge(zroi_wide,  on=['spike_key', 'label'])
    df_merged = df_merged.merge(zglob_wide, on=['spike_key', 'label'])
    df_merged = df_merged.merge(zz_wide,    on=['spike_key', 'label'])
    return df_merged

def visualize_pairwise(wide_df : pd.DataFrame, label_col='label'):
    """
    Create a pairwise scatter plot of all feature columns in `wide_df`, colored by `label_col`.
    """
    # select feature columns (exclude identifiers)
    exclude = {'spike_key', label_col}
    feature_cols = [c for c in wide_df.columns if c not in exclude]
    print(f"feature columns: {feature_cols}")
    # map labels to colors
    color_map = {'real': 'C0', 'not real': 'C1'}
    colors = wide_df[label_col].map(color_map)

    # draw scatter matrix
    scatter_matrix(
        wide_df[feature_cols],
        figsize=(12, 12),
        diagonal='hist',
        color=colors,
        marker='o',
        alpha=0.6,
        hist_kwds={'bins': 20}
    )
    plt.suptitle("Pairwise feature scatter by spike label", y=0.92)
    plt.show()


def get_feature_matrix(df : pd.DataFrame, rep_col : str):
    return np.vstack(df[rep_col].apply(eval).values)

def main_plot(features_df : pd.DataFrame, annotations_df : pd.DataFrame):
    #Step 1: Add features to the annotations DataFrame
    merged_df = pd.merge(annotations_df, features_df, on = 'spike_key' , how = 'left')
    #Step 2: Define labels as the output variable
    y = merged_df['label'].values
    #Step 3: Get feature names from the first row 
    feature_names = eval(merged_df.iloc[0]['feature_names'])
    #Step 4: Define different representations of the features
    rep_cols = [
        'raw_features',
        'raw_z_features',
        'z_scored_raw_features',
        'z_scored_z_features'
    ]
    rep_names = [
        'Raw',
        'ROI Z',
        'Dataset Z',
        'ROI Z then Dataset Z'
    ]

    for rep_col, rep_name in zip(rep_cols, rep_names):
        X = get_feature_matrix(df, rep_col)
        f_vals, mi_vals = variance_explained(X, y)
        print(f"\n{rep_name}:")
        for i, fname in enumerate(feature_names):
            print(f"  {fname:25s}  F={f_vals[i]:.3f}  MI={mi_vals[i]:.3f}")
    trimmed_df = trimmed_df[trimmed_df['label'].astype(str).str.strip() != '']
    if trimmed_df.empty:
        print("No labeled spikes to plot. Check your 'label' column.")
        return
    print(trimmed_df.head)
    expanded_features_df = expand_feature_representations(trimmed_df)
    visualize_pairwise(expanded_features_df)
   
