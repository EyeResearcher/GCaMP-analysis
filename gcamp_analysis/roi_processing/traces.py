from typing import TYPE_CHECKING

from gcamp_analysis.reports import TraceReport  # shared report dataclasses stay in pipeline/
if TYPE_CHECKING:
    from gcamp_analysis.data_classes.video import Video
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler


def normalize_minmax(f: np.ndarray) -> np.ndarray:
    """Min-max normalize fluorescence traces."""
    scaler = MinMaxScaler()
    flat_f = f.reshape(-1, 1)
    scaled_flat = scaler.fit_transform(flat_f)
    return scaled_flat.reshape(f.shape)

def get_savgol_params(fs: float, sensor_type: str = "gcamp8s") -> tuple[int, int]:
    if sensor_type == "gcamp8s":
        window_frames = int(0.6 * fs)
    elif sensor_type == "gcamp6f":
        window_frames = int(0.3 * fs)
    else:
        window_frames = int(0.4 * fs)
    window_length = 2 * (window_frames // 2) + 1
    window_length = max(9, window_length)
    return window_length, 3

@dataclass
class TraceService:
    smooth_sigma: float = 4.0
    sensor_type: str = "gcamp8s"
    def _process_traces(self, traces, fs) -> TraceReport:
        norm = normalize_minmax(traces)
        smoothed = gaussian_filter1d(norm, sigma=self.smooth_sigma, axis=1)
        wl, po = get_savgol_params(fs=fs, sensor_type=self.sensor_type)
        savgol = savgol_filter(norm, window_length=wl, polyorder=po, axis=1)
        return norm, smoothed, savgol
    def run(self, video: "Video") -> TraceReport:
        fs = float(video.fs)
        F = video.suite2p_data["F"]

        if video.is_concatenated and video.split_frame is not None:
            # --- Per-segment normalization ---
            sf = video.split_frame
            F_bl, F_tx = F[:, :sf], F[:, sf:]

            # Baseline
            norm_bl, sm_bl, sg_bl = self._process_traces(F_bl, fs)

            video.baseline_norm_f = norm_bl
            video.baseline_norm_sm_f = sm_bl
            video.baseline_norm_sg_f = sg_bl

            # Treatment
            norm_tx, sm_tx, sg_tx = self._process_traces(F_tx, fs)

            video.treatment_norm_f = norm_tx
            video.treatment_norm_sm_f = sm_tx
            video.treatment_norm_sg_f = sg_tx

            # Also populate full-trace fields by concatenating the
            video.norm_f = np.concatenate([norm_bl, norm_tx], axis=1)
            video.norm_sm_f = np.concatenate([sm_bl, sm_tx], axis=1)
            video.norm_sg_f = np.concatenate([sg_bl, sg_tx], axis=1)

            np.save(video.suite2p_path / "F_minmax.npy", video.norm_f)
        else:
            # --- Original single-video path ---
            norm, smoothed, savgol = self._process_traces(F, fs)
            video.norm_f = norm
            np.save(video.suite2p_path / "F_minmax.npy", video.norm_f)
            video.norm_sm_f = smoothed
            video.norm_sg_f = savgol

        return TraceReport(n_rois=int(video.n_rois), n_frames=int(video.n_frames), fs=fs)
