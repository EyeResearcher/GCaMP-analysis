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


@dataclass
class TraceService:
    smooth_sigma: float = 4.0
    sensor_type: str = "gcamp8s"

    @staticmethod
    def _resolve_savgol_window(n_frames: int, preferred: int = 51, polyorder: int = 3) -> int | None:
        """Return a valid odd Savitzky-Golay window for the given trace length."""
        if n_frames <= polyorder:
            return None
        window = min(preferred, n_frames)
        if window % 2 == 0:
            window -= 1
        if window <= polyorder:
            minimum = polyorder + 1
            window = minimum + 1 if minimum % 2 == 0 else minimum
        return window if window <= n_frames else None

    def _process_traces(self, traces, fs):
        norm = normalize_minmax(traces)
        smoothed = gaussian_filter1d(norm, sigma=self.smooth_sigma, axis=1)
        window = self._resolve_savgol_window(norm.shape[1])
        mean = np.mean(traces, axis=1, keepdims=True)
        std = np.std(traces, axis=1, keepdims=True)
        zscore = (traces - mean) / std
        if window is None:
            savgol = norm.copy()
            savgol_z = zscore.copy()
        else:
            savgol = savgol_filter(norm, window_length=window, polyorder=3, axis=1)
            savgol_z = savgol_filter(zscore, window_length=window, polyorder=3, axis=1)

        return norm, smoothed, savgol, zscore, savgol_z

    def run(self, video: "Video") -> TraceReport:
        fs = float(video.fs)
        F = video.suite2p_data["F"]
        norm, smoothed, savgol, zscore, savgol_z = self._process_traces(F, fs)

        video.norm_f = norm
        video.norm_sm_f = smoothed
        video.norm_sg_f = savgol
        video.z_f = zscore
        video.savgol_z_f = savgol_z

        return TraceReport(n_rois=int(video.n_rois), n_frames=int(video.n_frames), fs=fs)
