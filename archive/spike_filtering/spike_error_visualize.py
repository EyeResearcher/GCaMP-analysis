import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def visualize_spike_separate(row):
    suite2p_path = row['suite2p_path']
    roi_idx = int(row['roi_index'])
    spike_prob_idx = int(row['spike_prob_index'])
    raw_f_idx = int(row['raw_f_index'])
    spike_key = row['spike_key']

    # Load traces
    raw_f = gaussian_filter1d(np.load(suite2p_path + '/F.npy')[roi_idx], sigma=4.0)
    spike_prob = gaussian_filter1d(np.load(suite2p_path + '/cascade_spike_prob.npy')[roi_idx], sigma=4.0)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Raw trace
    axs[0].plot(raw_f, label='Raw F', linewidth=0.7)
    axs[0].axvline(raw_f_idx, color='red', linestyle='--', label='Spike Index')
    axs[0].set_title(f"Raw Trace | Spike Key: {spike_key}")
    axs[0].legend()

    # Spike probability trace
    axs[1].plot(spike_prob, label='Spike Prob', linewidth=0.7)
    axs[1].axvline(spike_prob_idx, color='purple', linestyle='--', label='Spike Prob Index')
    axs[1].set_title(f"Spike Probability Trace | Spike Key: {spike_key}")
    axs[1].legend()

    plt.xlabel("Frame")
    plt.tight_layout()
    plt.show()
def print_feature_comparison(row, X_sel, y, df_labeled, wanted, wanted_idx):
    # Feature values for this spike
    spike_features = np.array(eval(row['raw_z_features']))[wanted_idx]
    # Means for each class
    pos_mask = y == 1
    neg_mask = y == 0
    pos_mean = X_sel[pos_mask].mean(axis=0)
    neg_mean = X_sel[neg_mask].mean(axis=0)
    print("\nFeature comparison for spike:", row['spike_key'])
    print(f"{'Feature':25s} {'This spike':>12s} {'Mean (pos)':>12s} {'Mean (neg)':>12s}")
    for i, fname in enumerate(wanted):
        print(f"{fname:25s} {spike_features[i]:12.3f} {pos_mean[i]:12.3f} {neg_mean[i]:12.3f}")
def train_and_visualize_rf(merged_df):
    # Feature names in order in raw_z_features
    feature_names = eval(merged_df.iloc[0]['feature_names'])
    wanted = ['left_based_prom', 'spike_prob_value', 'auc', 'max_second_derivative', 'skew_contribution']
    wanted_idx = [feature_names.index(f) for f in wanted]

    # Extract X and y
    X = np.vstack(merged_df['raw_z_features'].apply(eval).values)
    X_sel = X[:, wanted_idx]
    y = merged_df['label'].values

    # Remove unlabeled
    mask = ~pd.isnull(y)
    X_sel = X_sel[mask]
    y = y[mask]
    df_labeled = merged_df[mask].reset_index(drop=True)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_sel, y, df_labeled, test_size=0.8, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Find false positives and false negatives
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

    print("\nCycle through False Positives (pred=1, true=0):")
    for idx in fp_idx:
        print_feature_comparison(df_test.iloc[idx], X_sel, y, df_labeled, wanted, wanted_idx)
        visualize_spike_separate(df_test.iloc[idx])
        input("Press Enter for next FP...")

    print("\nCycle through False Negatives (pred=0, true=1):")
    for idx in fn_idx:
        print_feature_comparison(df_test.iloc[idx], X_sel, y, df_labeled, wanted, wanted_idx)
        visualize_spike_separate(df_test.iloc[idx])
        input("Press Enter for next FN...")

# Usage:
# train_and_visualize_rf(merged_df)