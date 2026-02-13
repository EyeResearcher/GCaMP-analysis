from typing import TYPE_CHECKING

from pipeline.reports import TraceReport
if TYPE_CHECKING:
    from data_classes.video import Video
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from utils.preprocessing import normalize_minmax

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

    def run(self, video: "Video") -> TraceReport:
        fs = float(video.suite2p_data.get("fs", 30.0))
        video.norm_f = normalize_minmax(video.suite2p_data["F"])
        np.save(video.suite2p_path / "F_minmax.npy", video.norm_f)
        video.norm_sm_f = gaussian_filter1d(video.norm_f, sigma=self.smooth_sigma, axis=1)

        wl, po = get_savgol_params(fs, sensor_type=self.sensor_type)
        video.norm_sg_f = savgol_filter(video.norm_f, window_length=wl, polyorder=po, axis=1)
        return TraceReport(n_rois=int(video.n_rois), n_frames=int(video.n_frames), fs=fs)
