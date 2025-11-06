import sys

from pathlib import Path
import numpy as np 
import scipy.io
from joblib import load
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt 
from Cascade.cascade2p.cascade_wrapper import CascadePredictor
from typing import Dict
class SummaryFiles:
    """Loads Suite2p summary files and provides ROI data.
    Parameters:
        folder (Path or str): Path to the Suite2p plane0 folder.
        cascade_model (CascadePredictor): Preloaded CascadePredictor instance.
        new_model (bool): If True, recompute spike probabilities even if they exist on disk
    Attributes:
        folder (Path): Path to the Suite2p plane0 folder.
        f (np.ndarray): Raw fluorescence data (n_rois x n_timepoints).
        Fneu (np.ndarray): Neuropil fluorescence data (n_rois x n_timepoints).
        spks (np.ndarray): Deconvolved spike data (n_rois x n_timepoints).
        iscell (np.ndarray): Binary array indicating if each ROI is a cell.
        stat (np.ndarray): Array of ROI statistics and metadata.
        ops (dict): Suite2p operations and parameters.
        sp (np.ndarray): Spike probability data (n_rois x n_timepoints).
        sampling_rate (float): Sampling rate in Hz.
        smooth_f (np.ndarray): Smoothed fluorescence data.
        smooth_sp (np.ndarray): Smoothed spike probability data.
        cascade_model (CascadePredictor): Preloaded CascadePredictor instance.
    """
    slots = ['folder', 'f', 'Fneu', 'spks', 'iscell', 'stat', 'ops', 'sp', 'sampling_rate', 'smooth_f', 'smooth_sp', 'cascade_model']
    def __init__(self, folder : Path = None, cascade_model: CascadePredictor = None, new_model = False, summary_dict : dict = {}):
        self.cascade_model = cascade_model
        self.folder           : Path       = folder if folder.parts[-1] == "plane0" else folder / 'suite2p' / 'plane0'
        self.F                : np.ndarray = summary_dict.get('F', np.array([]))
        self.Fneu             : np.ndarray = summary_dict.get('Fneu', np.array([]))
        self.spks             : np.ndarray = summary_dict.get('spks', np.array([]))
        self.iscell           : np.ndarray = summary_dict.get('iscell', np.array([]))
        self.stat             : np.ndarray = summary_dict.get('stat', np.array([]))
        self.ops              : dict       = summary_dict.get('ops', {})
        self.sampling_rate    : int        = summary_dict.get('ops', {}).get('fs', 15)
        self.cascade_prob     : np.ndarray = summary_dict.get('cascade_prob', np.array([]))
        self.smooth_f         : np.ndarray = summary_dict.get('smooth_f', np.array([]))
        self.smooth_sp        : np.ndarray = summary_dict.get('smooth_sp', np.array([]))


    def _create_spike_prob(self, new_model = True):
        spike_prob_path = self.folder / 'cascade_prob.npy'
        if spike_prob_path.exists():

            #Compute new probvabilities as indicated by new model flag
            if new_model == True:
                spike_prob_path.unlink()
                self.cascade_prob = self.cascade_model.predict(self.F)
            else:
                self.cascade_prob = self._load_npy('cascade_prob.npy')
        else:
            self.cascade_prob = self.cascade_model.predict(self.F)
            self.cascade_prob = self._load_npy('cascade_prob.npy')
        
    def load_files(self):
        """
        Load all necessary files from the Suite2p folder.
        """
        self.f = self._load_npy('F.npy')
        self.Fneu = self._load_npy('Fneu.npy')
        self.spks = self._load_npy('spks.npy')
        self.iscell = self._load_npy('iscell.npy')
        self.stat = self._load_npy('stat.npy', allow_pickle=True)
        self.ops = self._load_npy('ops.npy', allow_pickle=True, as_dict=True)
        self.sampling_rate = self.ops.get('fs', 15)

    def _load_npy(self, filename, allow_pickle=False, as_dict=False):
        path : Path = self.folder / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        
        arr = np.load(path, allow_pickle=allow_pickle)
        if as_dict:
            return arr.item() if isinstance(arr, np.ndarray) else arr
        return arr

    def _load_mat(self, filename):

        path = self.folder / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing MAT file: {path}")
        
        data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        arrays = {k: v for k, v in data.items() if not k.startswith('__')}

        return next(iter(arrays.values())) if len(arrays)==1 else arrays

    def get_roi_data(self, idx):

        raw_f = self.f[idx]
        neu_f = self.Fneu[idx]
        
        return {
            'F': raw_f,
            'smooth_f' : self.smooth_f[idx],
            'Fneu': neu_f,
            'cascade_prob': self.cascade_prob[idx],
            'smooth_sp': self.smooth_sp[idx],
            'spks': self.spks[idx],
            'iscell': self.iscell[idx],
            'stat': self.stat[idx],
            'ops': self.ops
        }
    
    def _smooth_signal(self, sigma=2):
        from scipy.ndimage import gaussian_filter1d
        self.smooth_f = gaussian_filter1d(self.f, sigma=sigma, axis=1)
        self.smooth_sp = gaussian_filter1d(self.sp, sigma=sigma, axis=1)
        return self.smooth_f, self.smooth_sp

def load_filtering_model(model_path: str):
    
    model = load(model_path)
    return model


def save_sttc_heatmap(sttc_matrix: np.ndarray, output_path: Path):
    """
    Saves the STTC matrix as a heatmap PNG with values scaled from -1 to 1.
    """
    plt.figure(figsize=(8, 6))
    im = plt.imshow(sttc_matrix, vmin=-1, vmax=1, cmap='coolwarm')
    plt.colorbar(im, label='STTC')
    plt.title('Spike Time Tiling Coefficient (STTC) Heatmap')
    plt.xlabel('Neuron')
    plt.ylabel('Neuron')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_video_metrics(video_path: Path,
                       filtered_summary: dict,
                       summary_df: pd.DataFrame,
                       experiment_name: str,
                       timepoint_name: str,
                       sttc_matrix: np.ndarray) -> Path:
    """
    Creates a "metrics" folder under the given video_path, then:
      - Saves the per-neuron summary DataFrame as an Excel file named
        [experiment]_[timepoint]_[video_id].xlsx
      - Writes each array in filtered_summary (except 'ops') as .npy files
        under metrics/filtered_s2p/plane0

    Returns:
        Path to the metrics folder.
    """
    video_path = Path(video_path)
    metrics_folder = video_path / 'metrics'
    metrics_folder.mkdir(exist_ok=True)

    # 1) Save summary DataFrame
    fname = f"{experiment_name}_{timepoint_name}_{video_path.name}.xlsx"
    summary_path = metrics_folder / fname
    summary_df.to_excel(summary_path, index=True)

    # 2) Save filtered summary arrays
    filtered_folder = metrics_folder / 'filtered_s2p' / 'plane0'
    filtered_folder.mkdir(parents=True, exist_ok=True)
    for key, arr in filtered_summary.items():
        if isinstance(arr, dict) or key == 'ops':
            np.save(filtered_folder / f"{key}.npy", arr, allow_pickle = True)
    # 3) Save STTC matrix
    if sttc_matrix is not None:
        sttc_path = metrics_folder / 'sttc_matrix.npy'
        save_sttc_heatmap(sttc_matrix, sttc_path.with_suffix('.png'))
        np.save(sttc_path, sttc_matrix)  
    else:
        print("STTC matrix is None, not saving.")

    return metrics_folder


def save_timepoint_summary( experiment_name: str, timepoint_name: str,
                            timepoint_df: pd.DataFrame, video_dfs: Dict[str,pd.DataFrame], 
                            output_dir : Path, filename: str = None) -> Path:
    """
    Saves a combined Excel summary for a Timepoint:
      - Sheet 'Timepoint_Summary' with per-video stats
      - One sheet per video (video.video_id) with its summary_df

    Returns:
        Path to the created Excel file.
    """
    
    if filename is None:
        filename = f"{experiment_name}_{timepoint_name}_summary.xlsx"
    out_path = output_dir / filename

    with ExcelWriter(out_path) as writer:
        # Top‐level video summary
        timepoint_df.to_excel(writer, sheet_name='Timepoint_Summary')

        # Each video's detailed summary
        for video_id, video_df in video_dfs.items():
            if not video_df.empty:
                sheet = video_id[:31]
                video_df.to_excel(writer, sheet_name=sheet)

    return out_path
def save_cascade_predictions(folder: Path, spike_prob: np.ndarray):
    p = folder / 'cascade_spike_prob.npy'
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, spike_prob)
    return p

