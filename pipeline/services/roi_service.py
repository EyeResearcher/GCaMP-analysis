from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data_classes.roi import ROI
from data_classes.neuron import Neuron
from typing import TYPE_CHECKING

from pipeline.reports import ROIReport
if TYPE_CHECKING:
    from data_classes.video import Video

def _expected_feature_names(model: RandomForestClassifier | LogisticRegression, fallback_config_path: Optional[Path]) -> Optional[list[str]]:
    expected = None
    if hasattr(model, "feature_names_in_"):
        try:
            expected = list(model.feature_names_in_)
        except Exception:
            expected = None

    if expected is None and fallback_config_path and fallback_config_path.exists():
        try:
            cfg = json.load(open(fallback_config_path))
            expected = cfg.get("feature_names")
        except Exception:
            expected = None
    return expected

@dataclass
class ROIService:
    n_jobs: int = -1
    roi_config_path: Optional[Path] = Path("roi_classifier/models/roi_classifier_config.json")

    def create_rois(self, video: "Video") -> List[ROI]:
        rois: List[ROI] = []
        for i in range(video.n_rois):
            rois.append(
                ROI(
                    index=i,
                    f_trace=video.suite2p_data["F"][i, :],
                    stats=video.suite2p_data["stat"][i] if "stat" in video.suite2p_data else None,
                    fneu=video.suite2p_data["Fneu"][i] if "Fneu" in video.suite2p_data else None,
                )
            )
        return rois

    def filter_rois(self, video: "Video", all_rois: List[ROI], roi_model: RandomForestClassifier | LogisticRegression) -> tuple[List[ROI], np.ndarray]:
        if roi_model is None:
            raise RuntimeError("ROI classifier model is not provided.")

        # Extract features in parallel
        all_feats = Parallel(n_jobs=self.n_jobs)(
            delayed(roi.extract_features)(video.norm_sm_f[i, :])
            for i, roi in enumerate(all_rois)
        )
        for roi, feats in zip(all_rois, all_feats):
            roi.features = feats

        feats_df = pd.DataFrame(all_feats)

        expected = _expected_feature_names(roi_model, self.roi_config_path)
        if expected:
            for col in expected:
                if col not in feats_df.columns:
                    feats_df[col] = np.nan
            X = feats_df[expected].values
        else:
            X = feats_df.values

        preds = roi_model.predict(X).astype(bool)
        good_roi_mask = np.asarray(preds, dtype=bool)

        # Preserve explicitly marked-bad ROIs (your current logic)
        for i, roi in enumerate(all_rois):
            if roi.is_good is False:
                good_roi_mask[i] = False
                roi.is_good = False
            else:
                roi.is_good = bool(good_roi_mask[i])

        good_rois = [roi for roi in all_rois if roi.is_good]
        video.bad_rois = [roi for roi in all_rois if not roi.is_good]
        video.n_good_rois = len(good_rois)
        video.n_bad_rois = len(video.bad_rois)

        return good_rois, good_roi_mask

    def build_bad_rois_features_df(self, video: "Video") -> pd.DataFrame:
        if not video.bad_rois:
            video.bad_rois_features = pd.DataFrame()
            return video.bad_rois_features

        features_list = [roi.features for roi in video.bad_rois]
        indices = [roi.index for roi in video.bad_rois]
        df = pd.DataFrame(features_list, index=indices)
        df.index.name = "roi_index"
        video.bad_rois_features = df
        return df

    def create_neurons(self, video: "Video", good_rois: List[ROI]) -> List[Neuron]:
        neurons: List[Neuron] = []

        # Prefer video.fs if present; fallback to suite2p ops or default
        fs = float(getattr(video, "fs", None) or video.suite2p_data.get("ops", {}).get("fs", 30.0))

        for filtered_index, roi in enumerate(good_rois):
            neurons.append(
                Neuron(
                    roi=roi,                       # <-- wrapper style
                    filtered_index=filtered_index,
                    fs=fs,
                )
            )

        video.neurons = neurons
        return neurons
   
    def run(self, video: "Video", roi_model: Any) -> ROIReport:
        """
        Populates on video:
          - bad_rois, n_good_rois, n_bad_rois
          - bad_rois_features (optional)
          - neurons (created from good ROIs)
        Returns counts for narration.
        """
        all_rois = self.create_rois(video)
        good_rois, good_roi_mask = self.filter_rois(video, all_rois, roi_model)

        self.build_bad_rois_features_df(video)

        if len(good_rois) > 0:
            self.create_neurons(video, good_rois)
        else:
            video.neurons = []

        n_total = len(all_rois)
        n_good = len(good_rois)
        n_bad = n_total - n_good

        return ROIReport(
            n_rois_total=n_total,
            n_rois_good=n_good,
            n_rois_bad=n_bad,
            pass_rate=(n_good / n_total) if n_total else 0.0,
        )
