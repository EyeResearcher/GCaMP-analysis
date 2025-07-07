import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

from joblib import dump, Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from feature_utils import four_primary_roi_features
from tkinter import Tk, Button, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm

from scipy.io import loadmat
def build_template_traces():
    """
    Builds three template traces: few spikes, medium spikes, many spikes.

    Returns:
        tuple of np.arrays: (template_few, template_med, template_many)
    """
    print("Building spike-type-specific waveform templates.")

    categories = ["few", "medium", "many"]
    all_templates = []

    for category in categories:
        print(f"\n→ Building template for ROIs with {category} spikes:")
        paths_input = input(f"Enter paths to .npy files for {category} (comma-separated): ").split(',')
        rois_input = input(f"Enter ROI indices for each file (semicolon-separated lists, same order): ").split(';')

        template_traces = []
        for path_str, roi_list_str in zip(paths_input, rois_input):
            path = Path(path_str.strip())
            rois = list(map(int, roi_list_str.strip().split(',')))
            F = np.load(path)
            for roi in rois:
                template_traces.append(F[roi])

        template = np.mean(template_traces, axis=0)
        all_templates.append(template)

    return tuple(all_templates)  # (template_few, template_med, template_many)


def find_f_paths(base_dir):
    return list(base_dir.rglob("F.npy"))

class LabelerApp:
    def __init__(self, norm_traces, raw_traces, Fneu_traces, spks_traces, on_complete, source_label, roi_indices):
        self.root = Tk()
        self.root.geometry("1200x1000")
        self.source_label = source_label
        self.roi_indices = roi_indices
        self.norm_traces = norm_traces
        self.raw_traces = raw_traces
        self.Fneu_traces = Fneu_traces
        self.spks_traces = spks_traces
        self.labels = []
        self.index = 0
        self.on_complete = on_complete

        main_frame = Label(self.root)
        main_frame.pack()

        self.fig, axs = plt.subplots(3, 1, figsize=(15, 6), sharex=True)
        self.ax_norm, self.ax_raw, self.ax_spks = axs
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack()

        controls_frame = Label(self.root)
        controls_frame.pack()

        self.label = Label(controls_frame, text="Label ROI as Good (1) or Bad (0)")
        self.label.pack()

        Button(controls_frame, text="Good", command=lambda: self.label_trace(1)).pack(side="left", padx=10)
        Button(controls_frame, text="Bad", command=lambda: self.label_trace(0)).pack(side="left", padx=10)

        self.plot_trace()
        self.root.mainloop()

    def plot_trace(self):
        self.ax_norm.clear()
        self.ax_raw.clear()
        self.ax_spks.clear()

        idx = self.index
        roi_id = self.roi_indices[idx]

        self.ax_norm.plot(self.norm_traces[idx], color='blue')
        self.ax_norm.set_title(f"{self.source_label} | ROI {roi_id} — Normalized")
        self.ax_norm.set_ylabel("Norm F")

        self.ax_raw.plot(self.raw_traces[idx], color='black', label='F (raw)', linewidth=0.8)
        self.ax_raw.plot(self.Fneu_traces[idx], color='purple', label='Fneu (neuropil)', linewidth=0.8)
        self.ax_raw.set_title("Raw F vs Fneu (Neuropil)")
        self.ax_raw.set_ylabel("Fluorescence")
        self.ax_raw.legend()

        self.ax_spks.plot(self.spks_traces[idx], color='orange')
        self.ax_spks.set_title("Deconvolved Trace (spks)")
        self.ax_spks.set_xlabel("Frame #")
        self.ax_spks.set_ylabel("Spks")

        self.canvas.draw()

    def label_trace(self, label):
        self.labels.append(label)
        if self.index >= len(self.norm_traces) - 1:
            self.on_complete(self.labels)
            self.root.quit()
            self.root.destroy()
            return
        self.index += 1
        self.plot_trace()



def visualize_fn_fp(i, df, X_test_indices, label):
    row = df.iloc[X_test_indices[i]]
    plane0 = Path(row["source_file"]).parent

    # load raw F trace
    F = np.load(row["source_file"])
    roi_idx = int(row["roi_index"])
    raw_trace = F[roi_idx]
    norm_trace = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min() + 1e-6)

    # load spike_prob from .mat
    mat_path = plane0 / f"full_prediction_{Path(row['source_file']).name}.mat"
    mat = loadmat(mat_path)
    spike_prob = mat["spike_prob"][roi_idx]

    # two-panel figure
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # top: raw + normalized
    ax0.plot(raw_trace, color="black", label="Raw F")
    ax0.plot(norm_trace, color="blue", alpha=0.6, label="Norm F")
    ax0.set_ylabel("Fluorescence")
    ax0.legend(loc="upper right")

    # bottom: spike prob
    ax1.plot(spike_prob, color="red", label="Spike Prob")
    ax1.set_ylabel("P(spike)")
    ax1.set_xlabel("Frame")
    ax1.legend(loc="upper right")

    # suptitle a bit lower (y=0.93) and leave room via tight_layout(rect=...)
    fig.suptitle(
        f"{label} — ROI {roi_idx} from "
        f"{Path(row['source_file']).parts[-5]} / "
        f"{Path(row['source_file']).parts[-4]}",
        y=0.93,
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])  # leave top 10% for the suptitle
    plt.show()
def process_row(row):
    #try:
    plane0 = Path(row["source_file"]).parent
    f = np.load(row["source_file"])
    fneu = np.load(plane0 / "Fneu.npy")
    spks = np.load(plane0 / "spks.npy")
    spike_prob = loadmat(plane0 / "full_prediction_F.npy.mat")['spike_prob']
    spike_prob_trace = spike_prob[int(row["roi_index"])]
    raw_trace = f[int(row["roi_index"])]
    
    fneu_trace = fneu[int(row["roi_index"])]
    spks_trace = spks[int(row["roi_index"])]
    norm_trace = (raw_trace - raw_trace.min()) / (raw_trace.max() - raw_trace.min() + 1e-6)
    features_dict = four_primary_roi_features(raw_trace, spike_prob_trace)
    return features_dict, int(row["label"])

def main():
    # === SETUP ===
    BASE_DIR = Path(input("Where is the dataset located?"))
    SAMPLE_PERCENT = float(input("What proportion of the files should be sampled?"))
    PARENT_SAVE_FOLDER_PATH = Path(input("Path to folder for saving model and data:"))
    ROI_FILTERING_SAVE_FOLDER_PATH = PARENT_SAVE_FOLDER_PATH / "roi_filtering"
    ROI_FILTERING_SAVE_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    SAVE_MODEL_PATH = ROI_FILTERING_SAVE_FOLDER_PATH / "roi_classifier_model.pkl"
    SAVE_LABELS_PATH = ROI_FILTERING_SAVE_FOLDER_PATH / "roi_labels.csv"
    TEMPLATE_PATH = ROI_FILTERING_SAVE_FOLDER_PATH / "template_traces.npz"
    SPIKE_TEMPLATE_PATH = ROI_FILTERING_SAVE_FOLDER_PATH / "spike_template_norm.npy"
    # === TEMPLATE SETUP ===
    if TEMPLATE_PATH.exists() and input("Do you want to rebuild template traces? (y/n): ").strip().lower() != 'y':
        data = np.load(TEMPLATE_PATH)
        template_tuple = (data["few"], data["med"], data["many"])
        spike_template = np.load(SPIKE_TEMPLATE_PATH)
        print("Loaded existing templates.")
    else:
        template_few, template_med, template_many = build_template_traces()
        np.savez(TEMPLATE_PATH, few=template_few, med=template_med, many=template_many)
        template_tuple = (template_few, template_med, template_many)
        print("Saved new templates.")
    # === ANNOTATION ===
    if input("Do you want to annotate more data? (y/n): ").strip().lower() == 'y':
        def make_handler(norms, raws, indices, source, f_path):
            def handle_labels(labels):
                new_rows = pd.DataFrame({
                    "label": labels,
                    "source_file": str(f_path.resolve()),
                    "roi_index": indices
                })

                # Load existing if present
                if SAVE_LABELS_PATH.exists():
                    existing = pd.read_csv(SAVE_LABELS_PATH)

                    # Remove any existing annotations for these source_file + roi_index pairs
                    condition = ~(
                        (existing["source_file"] == str(f_path.resolve())) &
                        (existing["roi_index"].isin(indices))
                    )
                    updated = pd.concat([existing[condition], new_rows], ignore_index=True)
                else:
                    updated = new_rows

                # Overwrite with updated data
                updated.to_csv(SAVE_LABELS_PATH, index=False)
            return handle_labels

        f_paths = list(BASE_DIR.rglob("F.npy"))
        for f_path in f_paths:
            F = np.load(f_path)
            fneu_path = next(f_path.parent.rglob("Fneu.npy"))
            deconv_path = next(f_path.parent.rglob("spks.npy"))
            Fneu = np.load(fneu_path)
            deconv = np.load(deconv_path)
            n_total = F.shape[0]
            n_sample = max(10, int(n_total * SAMPLE_PERCENT))
            sampled_indices = np.random.choice(n_total, size=n_sample, replace=False)
            F_raw = F[sampled_indices]
            Fneu_single_roi = Fneu[sampled_indices]
            deconv_single_roi = deconv[sampled_indices]
            F_norm = (F_raw - F_raw.min(axis=1, keepdims=True)) / (F_raw.max(axis=1, keepdims=True) - F_raw.min(axis=1, keepdims=True) + 1e-6)
            source_label = f_path.parents[2].name
            handler = make_handler(F_norm, F_raw, sampled_indices, source_label, f_path)
            app = LabelerApp(F_norm, F_raw, Fneu_single_roi, deconv_single_roi, handler, source_label, sampled_indices)

    # === FEATURE RECONSTRUCTION ===
    print("Rebuilding features from roi_labels.csv...")
    df = pd.read_csv(SAVE_LABELS_PATH)
    features = []
    labels = []
 
    def compute_features_parallel_dict(df, template_tuple, spike_template, n_jobs=-1):
        
            #except Exception as e:
                #print(f"Error processing ROI {row['roi_index']} in {row['source_file']}: {e}")
                #eturn None

        results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(process_row)(row) for _, row in tqdm(df.iterrows(), total=len(df))
        )

        # Filter out failed jobs
        results = [r for r in results if r is not None]

        # Unpack
        feature_dicts, labels = zip(*results)
        X_df = pd.DataFrame(feature_dicts)
        y = np.array(labels)

        return X_df, y
    X_df,y = compute_features_parallel_dict(df, template_tuple, spike_template) 
    # === MODEL TRAINING ===
    df["label"] = y
    X_df["label"] = y  # Ensure alignment for safe splitting

    # Split using X_df (features) and y (labels), keeping index tracking
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_df.drop(columns=["label"]), y, df, test_size=0.2, random_state=42, stratify=y
    )

    # Save test indices to match back later
    X_test_indices = df_test.index.to_numpy()

    # Train model
    model = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=5,
        min_samples_split=2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Report performance
    print(classification_report(y_test, y_pred))

    # Optional error visualization
    if input("Do you want to view false negatives and positives? ").strip().lower() == "y":
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        false_negatives = np.where((y_test == 1) & (y_pred == 0))[0]
        false_positives = np.where((y_test == 0) & (y_pred == 1))[0]

        for i in false_negatives:
            visualize_fn_fp(i, df, X_test_indices, "False Negative")

        for i in false_positives:
            visualize_fn_fp(i, df, X_test_indices, "False Positive")
    dump(model, SAVE_MODEL_PATH)
if __name__ == "__main__":
    main()
