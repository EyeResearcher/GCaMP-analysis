import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif
 
def plot_scatter(X, y, feature_names, idx1, idx2, rep_name):
    plt.figure(figsize=(7,6))
    colors = np.where(y==1, 'red', 'blue')
    plt.scatter(X[:, idx1], X[:, idx2], c=colors, alpha=0.6)
    plt.xlabel(f"{feature_names[idx1]} ({rep_name})")
    plt.ylabel(f"{feature_names[idx2]} ({rep_name})")
    plt.title(f"Scatter: {feature_names[idx1]} vs {feature_names[idx2]}")
    plt.show()

def variance_explained(X, y):
    # Returns F-value and mutual information for each feature
    f_vals, _ = f_classif(X, y)
    mi_vals = mutual_info_classif(X, y)
    return f_vals, mi_vals


def get_feature_matrix(df: pd.DataFrame, rep_col: str):
    def parse_feature(val):
        if isinstance(val, str):
            return eval(val)
        elif isinstance(val, (list, np.ndarray)):
            return np.array(val)
        else:
            raise ValueError(f"Cannot parse feature value: {val}")
    features = [parse_feature(v) for v in df[rep_col].values]
    return np.vstack(features)
def plot_pca(X_pca, idx1, idx2, y):
    plt.figure(figsize=(7,6))
    colors = np.where(y==1, 'red', 'blue')
    plt.scatter(X_pca[:,idx1], X_pca[:,idx2], c=colors, alpha=0.6)
    plt.xlabel(f'PCA {idx1 + 1}')
    plt.ylabel(f'PCA {idx2 + 1}')
    plt.title('PCA of Top 5 Predictive Features')
    plt.show()
def run_pca(X, y, selected_feature_indices, feature_names):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    X_selected = X[:, selected_feature_indices]

    # Run PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_selected)
    f_vals, mi_vals = variance_explained(X_pca, y)
    # Print exlpained variance
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))

    #Print Component Descriptions 
    for i, comp in enumerate(pca.components_):
        pc = f"PC{i+1}"
        print(f" {pc:5s}  F={f_vals[i]:.3f}  MI={mi_vals[i]:.3f}")
        print(f"{pc} loadings:")
        for idx in selected_feature_indices:
            print(f"  {feature_names[idx]:25s}: {comp[idx]:.3f}")
        
    # Plot
    for i in range(len(pca.components_)):
        for j in range(i+1, len(pca.components_)):
            plot_pca(X_pca, i, j, y)
            user_input = input(f"Plotted PCA {i+1} vs PCA {j+1}. Press Enter for next, or 'q' to quit: ")
            if user_input.lower() == 'q':
                return

def plot_pairwise_scatter(X, y, feature_names, rep_name):
    print(f"\nCycling through all pairwise feature scatter plots for {rep_name}. Press Enter for next plot, or type 'q' to quit.")
    n_features = X.shape[1]
    for i in range(n_features):
        for j in range(i+1, n_features):
            plot_scatter(X, y, feature_names, i, j, rep_name)
            user_input = input(f"Plotted {feature_names[i]} vs {feature_names[j]}. Press Enter for next, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
        else:
            continue
        break
def main_plot(features_df : pd.DataFrame, annotations_df : pd.DataFrame,pairwise = False, pca = False):
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
        X = get_feature_matrix(merged_df, rep_col)
        f_vals, mi_vals = variance_explained(X, y)
        print(f"\n{rep_name}:")
        for i, fname in enumerate(feature_names):
            print(f"  {fname:25s}  F={f_vals[i]:.3f}  MI={mi_vals[i]:.3f}")
    rep_idx = input(f"Choose representation [0:Raw, 1:ROI Z, 2:Dataset Z, 3:ROI Z then Dataset Z]: ")
    try:
        rep_idx = int(rep_idx)
        rep_col = rep_cols[rep_idx]
        rep_name = rep_names[rep_idx]
    except:
        print("Invalid input.")
        return

    X = get_feature_matrix(merged_df, rep_col)
    if pairwise: 
        plot_pairwise_scatter(X, y, feature_names, rep_name)
    if pca:
        selected_features_indices = input(f"""Select features for PCA (comma-separated indices, e.g. 0,1,2) from the list:
                                  {', '.join([f"{i}: {name}" for i, name in enumerate(feature_names)])}""")
        selected_features_indices = [int(i) for i in selected_features_indices.split(',')]
        run_pca(X, y, selected_features_indices, feature_names)
import scipy.stats as stats
import seaborn as sns
def plot_feature_distributions(df: pd.DataFrame, cols=None, bins=40, figsize=(10,4),
                               qq=True, show_stats=True, max_plots=None):
    """
    Plot distribution and Q-Q plot for each numeric column in df (or provided cols).
    Includes an inset box showing basic stats (n, mean, std, skew, kurtosis, Shapiro p).
    Returns a dict of stats for each column.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c in df.columns]
    if max_plots is not None:
        cols = cols[:max_plots]

    results = {}
    for col in cols:
        series = df[col].dropna().astype(float)
        n = len(series)
        if n == 0:
            results[col] = {'n': 0}
            continue

        mean = float(series.mean())
        std = float(series.std(ddof=1))
        skew_v = float(stats.skew(series))
        kurt_v = float(stats.kurtosis(series))  # fisher's (0 for normal)
        try:
            sh_stat, sh_p = stats.shapiro(series) if n <= 5000 else (np.nan, np.nan)
        except Exception:
            sh_stat, sh_p = (np.nan, np.nan)

        results[col] = {
            'n': n, 'mean': mean, 'std': std,
            'skew': skew_v, 'kurtosis': kurt_v,
            'shapiro_stat': sh_stat, 'shapiro_p': sh_p
        }

        # Plotting
        fig, axes = plt.subplots(1, 2 if qq else 1, figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Histogram + KDE
        sns.histplot(series, bins=bins, kde=True, stat='density', ax=axes[0], color='C0')
        axes[0].set_title(f"{col} (n={n})")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Density')

        # Inset with stats (always create, controlled by show_stats)
        if show_stats:
            inset_ax = inset_axes(axes[0], width="32%", height="32%", loc='upper right', borderpad=0.5)
            inset_ax.axis('off')
            stat_lines = [
                f"n = {n}",
                f"mean = {mean:.3f}",
                f"std  = {std:.3f}",
                f"skew = {skew_v:.3f}",
                f"kurt = {kurt_v:.3f}"
            ]
            if not np.isnan(sh_p):
                stat_lines.append(f"Shapiro p = {sh_p:.3g}")
            inset_text = "\n".join(stat_lines)
            inset_ax.text(0, 1, inset_text, va='top', ha='left', fontsize=9,
                          family='monospace', bbox=dict(boxstyle="round", fc="w", alpha=0.9))

        # Q-Q plot
        if qq:
            qq_ax = axes[1]
            stats.probplot(series, dist="norm", plot=qq_ax)
            qq_ax.set_title(f"Q-Q plot: {col}")

        plt.tight_layout()
        plt.show()

    return results