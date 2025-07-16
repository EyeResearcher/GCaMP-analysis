import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ---- Load data ----
CSV_PATH = input("Enter path to spike features CSV: ")
df = pd.read_csv(CSV_PATH)

# Extract features and labels
feature_cols = ['z_scored_features'] if 'z_scored_features' in df.columns else ['raw_features']
X = np.vstack(df[feature_cols[0]].apply(eval).values)
y = df['label'].values

# Remove unlabeled spikes
mask = pd.notnull(y)
X = X[mask]
y = y[mask]
df_labeled = df[mask].reset_index(drop=True)

# ---- Train/test split ----
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(X, y, df_labeled, test_size=0.2, random_state=42)

# ---- Train model ----
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# ---- Model summary ----
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---- Visualize false positives/negatives ----
fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

def plot_spike(row):
    suite2p_path = row['suite2p_path']
    roi_idx = int(row['roi_index'])
    spike_prob_idx = int(row['spike_prob_index'])
    raw_f_idx = int(row['raw_f_index'])
    raw_f = np.load(suite2p_path + '/F.npy')[roi_idx]
    spike_prob = np.load(suite2p_path + '/cascade_spike_prob.npy')[roi_idx]
    plt.figure(figsize=(10,4))
    plt.plot(raw_f, label='Raw F', linewidth=0.7)
    plt.axvline(raw_f_idx, color='red', linestyle='--', label='Spike')
    plt.plot(spike_prob, label='Spike Prob', linewidth=0.7)
    plt.axvline(spike_prob_idx, color='purple', linestyle='--', label='Spike Prob Index')
    plt.legend()
    plt.title(f"ROI {roi_idx} | Spike {spike_prob_idx} | Label: {row['label']}")
    plt.show()

print("\nCycle through False Positives (pred=1, true=0):")
for idx in fp_idx:
    plot_spike(df_test.iloc[idx])
    input("Press Enter for next FP...")

print("\nCycle through False Negatives (pred=0, true=1):")
for idx in fn_idx:
    plot_spike(df_test.iloc[idx])
    input("Press Enter for next FN...")