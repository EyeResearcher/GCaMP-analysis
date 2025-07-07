# Re-run the comparison function definition after reset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from joblib import load, Parallel, delayed
import numpy as np
from sklearn.model_selection import train_test_split
from feature_utils import extract_features, four_primary_roi_features
from scipy.io import loadmat
from pathlib import Path
LABELS_CSV = r"C:\Users\mzinn1\Desktop\Scripts\post_suite2p_analysis\model_runs\GCaMP8s_Olympus_Glass\roi_filtering\roi_labels.csv"
TEMPLATE_PATH = r"C:\Users\mzinn1\Desktop\Scripts\post_suite2p_analysis\model_runs\GCaMP8s_Olympus_Glass\roi_filtering\template_traces.npz"
SPIKE_TEMPLATE_PATH = r"C:\Users\mzinn1\Desktop\Scripts\post_suite2p_analysis\model_runs\GCaMP8s_Olympus_Glass\roi_filtering\spike_template_norm.npy"
def compute_features_parallel_df(df, template_tuple, spike_template, n_jobs=-1):
    def process_row(row):
        try:
            plane0 = Path(row["source_file"]).parent
            f = np.load(row["source_file"])
            fneu = np.load(plane0 / "Fneu.npy")
            spks = np.load(plane0 / "spks.npy")
            raw = f[int(row["roi_index"])]
            spks = np.load(plane0 / "spks.npy")
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

df = pd.read_csv(LABELS_CSV)
templates = np.load(TEMPLATE_PATH)
spike_template = np.load(SPIKE_TEMPLATE_PATH)
template_tuple = (templates["few"], templates["med"], templates["many"])
X_df, y = compute_features_parallel_df(df, template_tuple, spike_template)
X_df["label"] = y
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X_df.drop(columns=["label"]), y, df, test_size=0.2, stratify=y, random_state=42
)
# Update the function to print and return false positive/negative info with per-ROI delta analysis
# Comparison function for RandomForest, GradientBoosting (tuned), and LogisticRegression
from sklearn.linear_model import LogisticRegression

# Add XGBoost to the existing model comparison function
from xgboost import XGBClassifier

def compare_multiple_models_with_xgb(X_train, X_test, y_train, y_test, feature_names, df_test_meta=None):
    results = {}

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)
    print("=== Random Forest ===")
    print(classification_report(y_test, y_rf))
    results["RandomForest"] = {
        "model": rf,
        "y_pred": y_rf,
        "importances": rf.feature_importances_,
        "fp": np.where((y_test == 0) & (y_rf == 1))[0],
        "fn": np.where((y_test == 1) & (y_rf == 0))[0]
    }

    # --- Tuned Gradient Boosting ---
    gb = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=5,
        min_samples_split=2, random_state=42
    )
    gb.fit(X_train, y_train)
    y_gb = gb.predict(X_test)
    print("\n=== Gradient Boosting (Tuned) ===")
    print(classification_report(y_test, y_gb))
    results["GradientBoosting"] = {
        "model": gb,
        "y_pred": y_gb,
        "importances": gb.feature_importances_,
        "fp": np.where((y_test == 0) & (y_gb == 1))[0],
        "fn": np.where((y_test == 1) & (y_gb == 0))[0]
    }

    # --- Logistic Regression ---
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_lr = lr.predict(X_test)
    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, y_lr))
    coef_importance = np.abs(lr.coef_[0])
    results["LogisticRegression"] = {
        "model": lr,
        "y_pred": y_lr,
        "importances": coef_importance,
        "fp": np.where((y_test == 0) & (y_lr == 1))[0],
        "fn": np.where((y_test == 1) & (y_lr == 0))[0]
    }

    # --- XGBoost ---
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_xgb = xgb.predict(X_test)
    print("\n=== XGBoost ===")
    print(classification_report(y_test, y_xgb))
    results["XGBoost"] = {
        "model": xgb,
        "y_pred": y_xgb,
        "importances": xgb.feature_importances_,
        "fp": np.where((y_test == 0) & (y_xgb == 1))[0],
        "fn": np.where((y_test == 1) & (y_xgb == 0))[0]
    }

    # --- Feature Importances ---
    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
    axs = axs.ravel()
    for idx, (name, model_info) in enumerate(results.items()):
        axs[idx].barh(feature_names, model_info["importances"])
        axs[idx].set_title(name)
    plt.tight_layout()
    plt.show()

    # --- Print per-ROI deltas ---
    for model_name, data in results.items():
        print(f"\n--- {model_name} False Positives ---")
        for i in data["fp"]:
            if df_test_meta is not None:
                roi_id = df_test_meta.iloc[i]["roi_index"]
                source = df_test_meta.iloc[i]["source_file"]
                delta = X_test.iloc[i] - X_test[y_test == 0].mean()
                print(f"ROI {roi_id} from {source}")
            else:
                delta = X_test.iloc[i] - X_test[y_test == 0].mean()
            print(delta.sort_values())
            print()

        print(f"--- {model_name} False Negatives ---")
        for i in data["fn"]:
            if df_test_meta is not None:
                roi_id = df_test_meta.iloc[i]["roi_index"]
                source = df_test_meta.iloc[i]["source_file"]
                delta = X_test.iloc[i] - X_test[y_test == 1].mean()
                print(f"ROI {roi_id} from {source}")
            else:
                delta = X_test.iloc[i] - X_test[y_test == 1].mean()
            print(delta.sort_values())
            print()

    return results




results = compare_multiple_models_with_xgb(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=X_train.columns.tolist(),
    df_test_meta=df_test
)
