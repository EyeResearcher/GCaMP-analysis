"""ROI preprocessing module for the modular pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler

from Cascade.cascade2p.cascade_wrapper import CascadePredictor

from roi_filtering.feature_utils import four_primary_roi_features
from utils.io_utils import SummaryFiles

from .config import PreprocessingConfig


@dataclass(slots=True)
class PreprocessingResult:
    """Outputs after ROI preprocessing for a single video."""

    summary: SummaryFiles
    roi_table: pd.DataFrame
    kept_indices: np.ndarray
    fluorescence: np.ndarray
    cascade_prob: np.ndarray
    smooth_f: np.ndarray
    smooth_prob: np.ndarray

    def iter_kept(self) -> Iterable[int]:
        return iter(self.kept_indices.tolist())


def _maybe_gaussian(trace: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return trace
    return gaussian_filter1d(trace, sigma=sigma, axis=-1)


def _compute_roi_features(fluorescence: np.ndarray, cascade_prob: np.ndarray) -> pd.DataFrame:
    feature_rows = []
    for idx in range(fluorescence.shape[0]):
        derivative_skew, _, _, prom_mean = four_primary_roi_features(
            fluorescence[idx], cascade_prob[idx]
        )
        feature_rows.append((derivative_skew, prom_mean))

    features = pd.DataFrame.from_records(
        feature_rows,
        columns=["derivative_skew", "spike_prom_mean"],
    )
    return features


def _scale_features(features: pd.DataFrame, mode: str) -> pd.DataFrame:
    if features.empty:
        return features
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features.values)
    scaled_df = pd.DataFrame(scaled, columns=[f"mm_{c}" for c in features.columns])
    return scaled_df


def _predict_rois(
    feature_matrix: pd.DataFrame,
    roi_model,
    probability_threshold: Optional[float] = None,
) -> pd.DataFrame:
    if feature_matrix.empty:
        return pd.DataFrame(columns=["prediction", "probability"])

    predictions = roi_model.predict(feature_matrix.values)

    proba = None
    if hasattr(roi_model, "predict_proba"):
        proba = roi_model.predict_proba(feature_matrix.values)[:, -1]
    elif hasattr(roi_model, "decision_function"):
        raw = roi_model.decision_function(feature_matrix.values)
        proba = 1 / (1 + np.exp(-raw))

    data = {"prediction": predictions}
    if proba is not None:
        data["probability"] = proba
        if probability_threshold is not None:
            data["prediction"] = (proba >= probability_threshold).astype(int)

    return pd.DataFrame(data)


def _filter_summary(summary: SummaryFiles, kept: np.ndarray) -> SummaryFiles:
    filtered_summary = SummaryFiles(
        folder=summary.folder,
        cascade_model=summary.cascade_model,
        summary_dict={
            "F": summary.f[kept],
            "Fneu": summary.Fneu[kept],
            "spks": summary.spks[kept],
            "iscell": summary.iscell[kept],
            "stat": summary.stat[kept],
            "ops": summary.ops,
            "cascade_prob": summary.cascade_prob[kept],
            "smooth_f": summary.smooth_f[kept],
            "smooth_sp": summary.smooth_sp[kept],
        },
    )
    # SummaryFiles expects both upper and lower case attributes in legacy code paths
    filtered_summary.f = filtered_summary.F
    filtered_summary.sp = filtered_summary.cascade_prob
    return filtered_summary


def _resolve_plane_dir(video_path: Path) -> Path:
    candidates = [
        video_path / "suite2p" / "plane0",
        video_path / "plane0",
        video_path,
    ]

    for candidate in candidates:
        if candidate.name != "plane0":
            continue
        if candidate.exists() and (candidate / "F.npy").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate Suite2p plane0 directory under "
        f"{video_path}. Expected a folder containing F.npy."
    )


def run_preprocessing(
    video_path: Path,
    cascade_model: CascadePredictor,
    roi_model,
    config: PreprocessingConfig,
    probability_threshold: Optional[float] = None,
) -> PreprocessingResult:
    """Load Suite2p outputs, compute ROI features, and filter neurons."""

    plane_dir = _resolve_plane_dir(Path(video_path))

    summary = SummaryFiles(folder=plane_dir, cascade_model=cascade_model)
    summary.load_files()
    summary.F = summary.f  # Legacy compatibility

    # Ensure cascade probabilities exist
    if summary.cascade_model is None:
        summary.cascade_model = cascade_model

    existing_prob = plane_dir / "cascade_prob.npy"
    recompute = config.recompute_cascade or not existing_prob.exists()
    if recompute:
        preds = cascade_model.predict(summary.f)
        summary.cascade_prob = np.asarray(preds)
        np.save(existing_prob, summary.cascade_prob)
    else:
        summary.cascade_prob = np.load(existing_prob)

    summary.sp = summary.cascade_prob

    # Smooth traces if requested
    summary.smooth_f = _maybe_gaussian(summary.f, config.smoothing_sigma_f)
    summary.smooth_sp = _maybe_gaussian(summary.cascade_prob, config.smoothing_sigma_prob)

    features = _compute_roi_features(summary.smooth_f, summary.smooth_sp)
    scaled_features = _scale_features(features, config.minmax_mode)

    roi_predictions = _predict_rois(features, roi_model, probability_threshold)
    roi_table = pd.concat([features, scaled_features, roi_predictions], axis=1)
    roi_table.index.name = "roi_index"

    kept_mask = roi_table["prediction"].astype(bool).to_numpy()
    kept_indices = np.flatnonzero(kept_mask)

    if config.min_spikes_per_roi > 0:
        valid = []
        for idx in kept_indices:
            trace = summary.cascade_prob[idx]
            if np.count_nonzero(trace > 0.01) >= config.min_spikes_per_roi:
                valid.append(idx)
        kept_indices = np.array(valid, dtype=int)

    fluorescence = summary.f[kept_indices]
    cascade_prob = summary.cascade_prob[kept_indices]
    smooth_f = summary.smooth_f[kept_indices]
    smooth_prob = summary.smooth_sp[kept_indices]

    filtered_summary = _filter_summary(summary, kept_indices)

    return PreprocessingResult(
        summary=filtered_summary,
        roi_table=roi_table,
        kept_indices=kept_indices,
        fluorescence=fluorescence,
        cascade_prob=cascade_prob,
        smooth_f=smooth_f,
        smooth_prob=smooth_prob,
    )
