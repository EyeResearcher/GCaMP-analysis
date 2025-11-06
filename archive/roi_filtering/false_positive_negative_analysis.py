import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from joblib import load, Parallel, delayed
from sklearn.model_selection import train_test_split
from feature_utils import four_primary_roi_features
from pathlib import Path
from scipy.io import loadmat
# === CONFIG ===
MODEL_PATH = r"C:\Users\mzinn1\Desktop\Scripts\post_suite2p_analysis\model_runs\GCaMP8s_Olympus_Glass\roi_filtering\roi_classifier_model.pkl"
LABELS_CSV = r"C:\Users\mzinn1\Desktop\Scripts\post_suite2p_analysis\model_runs\GCaMP8s_Olympus_Glass\roi_filtering\roi_labels.csv"
TEMPLATE_PATH = r"C:\Users\mzinn1\Desktop\Scripts\post_suite2p_analysis\model_runs\GCaMP8s_Olympus_Glass\roi_filtering\template_traces.npz"
SPIKE_TEMPLATE_PATH = r"C:\Users\mzinn1\Desktop\Scripts\post_suite2p_analysis\model_runs\GCaMP8s_Olympus_Glass\roi_filtering\spike_template_norm.npy"
# === LOAD MODEL + TEMPLATE ===
model = load(MODEL_PATH)
df = pd.read_csv(LABELS_CSV)
templates = np.load(TEMPLATE_PATH)
spike_template = np.load(SPIKE_TEMPLATE_PATH)
template_tuple = (templates["few"], templates["med"], templates["many"])
def load_roi_files(plane0, idx):
    f = np.load(plane0 / "F.npy")[idx]
    fneu = np.load(plane0 / "Fneu.npy")[idx]
    spks = np.load(plane0 / "spks.npy")[idx]
    norm = (f - f.min()) / (f.max() - f.min() + 1e-6)
    return f, fneu, spks, norm
# === FEATURE EXTRACTION ===
def compute_features_parallel_df(df = df, template_tuple = template_tuple, spike_template = spike_template, n_jobs=-1):
    def process_row(row):
        try:
            plane0 = Path(row["source_file"]).parent
            f = np.load(row["source_file"])
            fneu = np.load(plane0 / "Fneu.npy")
            spks = np.load(plane0 / "spks.npy")
            raw = f[int(row["roi_index"])]
            spike_prob = loadmat(plane0 / "full_prediction_F.npy.mat")['spike_prob']
            spike_prob_trace = spike_prob[int(row["roi_index"])]
            fneu_trace = fneu[int(row["roi_index"])]
            spks_trace = spks[int(row["roi_index"])]
            norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
            feats = four_primary_roi_features(raw, spike_prob_trace)
            return feats, int(row["label"])
        except Exception as e:
            print(f"Skipped ROI {row['roi_index']} in {row['source_file']}: {e}")
            return None

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(row) for _, row in df.iterrows()
    )

    results = [r for r in results if r is not None]
    if not results:
        return pd.DataFrame(), np.array([])

    feats_dicts, labels = zip(*results)
    X_df = pd.DataFrame(feats_dicts)
    return X_df, np.array(labels)

X_df, y = compute_features_parallel_df(df, template_tuple, spike_template)
# === Get False Negative Probabilities === #

def show_false_negative_probabilities(model, X_test, y_test, df_test_meta, model_name="GradientBoosting"):
    # Get prediction probabilities for class 1
    probs = model.predict_proba(X_test)[:, 1]

    # Identify false negatives
    false_negatives = np.where((y_test == 1) & (model.predict(X_test) == 0))[0]
    false_positives = np.where((y_test == 0) & (model.predict(X_test) == 1))[0]
    def show_prob(group, group_type):
        print(f"\n🔍 {group_type} — {model_name} (missed true class 1s):\n")
        for i in group:
            roi_id = df_test_meta.iloc[i]["roi_index"]
            source = df_test_meta.iloc[i]["source_file"]
            prob = probs[i]
            print(f"ROI {roi_id} from {source} → Predicted Probability: {prob:.4f}")
        return 
    show_prob(false_negatives, "False Negatives")
    show_prob(false_positives, "False Positives")
    return false_negatives, probs

def show_random_features(model, X_test, y_test, df_test_meta, model_name = "Gradient Boosting"):


    # Identify accurate,innacurate ROIs
    true_negatives = np.where((y_test == 0) & (model.predict(X_test) == 0))[0]
    true_positives = np.where((y_test == 1) & (model.predict(X_test) == 1))[0]
    false_negatives = np.where((y_test == 1) & (model.predict(X_test) == 0))[0]
    false_positives = np.where((y_test == 0) & (model.predict(X_test) == 1))[0]
    def show_feature(group, group_type):
        print(f"\n🔍 {group_type} — {model_name} :\n")
        rand_idx = np.random.randint(low=0, high=len(group), size=5)
        rand_group = group[rand_idx]
        for i in rand_group:
            roi_id = df_test_meta.iloc[i]["roi_index"]
            source = df_test_meta.iloc[i]["source_file"]
            plane0 = Path(source).parent
            f, fneu, spks, f_norm = load_roi_files(Path(source).parent, roi_id)
            spike_prob = loadmat(plane0 / "full_prediction_F.npy.mat")['spike_prob']
            spike_prob_trace = spike_prob[int(roi_id)]
            features = four_primary_roi_features(f, spike_prob_trace)
            print(f"ROI {roi_id} from {source}")
            for feature, value in features.items():
                print(f"{feature} : {value}")
            visualize_fn_fp(f, f_norm,roi_id, source, group_type)
        return 
    show_feature(true_negatives, "True Negatives")
    show_feature(true_positives, "True Positives")
    show_feature(false_negatives, "False Negatives")
    show_feature(false_positives, "False Positives")
    return

def visualize_fn_fp(raw_trace, norm_trace, spike_prob, roi_id, source, label):
    """
    raw_trace : 1D np.array of raw F
    norm_trace: 1D np.array of normalized F
    spike_prob: 1D np.array of inferred spike probability
    roi_id    : int or str, your ROI index
    source    : path to the F.npy file (string)
    label     : "False Positive" / "False Negative" / etc
    """
    # set up two stacked axes
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # top panel: raw + normalized
    ax0.plot(raw_trace, color='black', label='Raw F')
    ax0.plot(norm_trace, color='blue', alpha=0.6, label='Norm F')
    ax0.set_ylabel("Fluorescence")
    ax0.legend(loc="upper right")

    # bottom panel: spike probability
    ax1.plot(spike_prob, color='red', label='Spike Prob')
    ax1.set_ylabel("P(spike)")
    ax1.set_xlabel("Frame")
    ax1.legend(loc="upper right")

    # suptitle, nudged down so it isn’t cut off
    title = (
        f"{label} — ROI {roi_id}  from  "
        f"{Path(source).parts[-5]}/{Path(source).parts[-4]}"
    )
    fig.suptitle(title, y=0.93, fontsize=14)

    # leave a bit of room at the top for that title
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

# === SPLIT + TRAIN + EVAL ===
X_df["label"] = y
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X_df.drop(columns=["label"]), y, df, test_size=0.2, stratify=y, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
show_random_features(model, X_test, y_test, df_test)
false_negatives, probs = show_false_negative_probabilities(
    model=model,
    X_test=X_test,
    y_test=y_test,
    df_test_meta=df_test,
    model_name="GradientBoosting"
)
# === ERROR ANALYSIS ===
false_positives = np.where((y_test == 0) & (y_pred == 1))[0]
false_negatives = np.where((y_test == 1) & (y_pred == 0))[0]

print(f"False Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")

df_test.iloc[false_positives].to_csv("false_positives.csv", index=False)
df_test.iloc[false_negatives].to_csv("false_negatives.csv", index=False)

# === FEATURE DIFFERENCE: False Positives vs Class 0 Mean ===
# 1) slice out the FPs/FNs in the test set:
X_fp     = X_test.iloc[false_positives]
X_fn     = X_test.iloc[false_negatives]

# 2) class‐conditional stats:
X_class0 = X_test[y_test == 0]
mean0    = X_class0.mean(axis=0)
std0     = X_class0.std(axis=0, ddof=0).replace(0, 1)

X_class1 = X_test[y_test == 1]
mean1    = X_class1.mean(axis=0)
std1     = X_class1.std(axis=0, ddof=0).replace(0, 1)

# 3) per‐ROI z‐score DataFrames:
z_df_fp = (X_fp - mean0) / std0
z_df_fn = (X_fn - mean1) / std1
# 4) plot the *mean* z‐score per feature:
plt.figure(figsize=(10,6))
(z_df_fp.mean(axis=0)
       .sort_values()
       .plot.barh(color='C0'))
plt.axvline(0, color='k', linestyle='--')
plt.title("Mean Z-Score of False Positives vs Class 0")
plt.xlabel("Z-Score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
(z_df_fn.mean(axis=0)
       .sort_values()
       .plot.barh(color='C1'))
plt.axvline(0, color='k', linestyle='--')
plt.title("Mean Z-Score of False Negatives vs Class 1")
plt.xlabel("Z-Score")
plt.tight_layout()
plt.show()

# 5) now print *each* ROI’s z-score vector:
print("\n--- Per-ROI Z-Scores: False Positives ---\n")
for test_pos in false_positives:
    df_idx = X_test.index[test_pos]       # this is the original row index in df_test
    roi_id = df_test.loc[df_idx, "roi_index"]
    src    = df_test.loc[df_idx, "source_file"]
    z_row  = z_df_fp.loc[df_idx]           # fetch that ROI’s z‐scores
    print(f"FP ROI {roi_id} from {src}")
    print(z_row.sort_values())
    print()

print("\n--- Per-ROI Z-Scores: False Negatives ---\n")
for test_pos in false_negatives:
    df_idx = X_test.index[test_pos]
    roi_id = df_test.loc[df_idx, "roi_index"]
    src    = df_test.loc[df_idx, "source_file"]
    z_row  = z_df_fn.loc[df_idx]
    print(f"FN ROI {roi_id} from {src}")
    print(z_row.sort_values())
    print()

# === GLOBAL FEATURE IMPORTANCE ===
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X_df.drop(columns="label").columns, importances)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
