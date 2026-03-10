from dataclasses import dataclass
from typing import List, Optional, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from gcamp_analysis.data_classes.roi import ROI
from gcamp_analysis.data_classes.neuron import Neuron
from typing import TYPE_CHECKING

from gcamp_analysis.reports import ROIReport
from utils.inference import prepare_features

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.video import Video


@dataclass
class ROIService:
    n_jobs: int = -1

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
    def _get_preds(self,traces, rois : list[ROI], model : RandomForestClassifier | LogisticRegression, transform):
        feats = Parallel(n_jobs=self.n_jobs)(
                delayed(roi.extract_features)(traces[i, :])
                for i, roi in enumerate(rois)
            )
        feats_df = pd.DataFrame(feats)
        X = prepare_features(feats_df, model, transform)
        preds = model.predict(X).astype(bool)
        return preds, feats

    def _assign_roi_status(
        self,
        rois: List[ROI],
        good_roi_mask: np.ndarray,
        feats: list,
        preds_bl: Optional[np.ndarray] = None,
        preds_tx: Optional[np.ndarray] = None,
    ) -> None:
        for i, roi in enumerate(rois):
            if feats:
                roi.features = feats[i]
            if preds_bl is not None and preds_tx is not None:
                roi.active_segments = {
                    "baseline": bool(preds_bl[i]),
                    "treatment": bool(preds_tx[i]),
                }
            if roi.is_good is False:
                good_roi_mask[i] = False
                roi.is_good = False
                continue  
            roi.is_good = bool(good_roi_mask[i])
    def filter_rois(
        self,
        video: "Video",
        all_rois: List[ROI],
        roi_model: RandomForestClassifier | LogisticRegression,
        model_config: Optional[dict] = None,
    ) -> tuple[List[ROI], np.ndarray]:
        """
        Filter ROIs using the trained classifier.
        
        In concatenated mode, features are extracted separately for each
        segment (baseline / treatment) and the classifier is run on each.
        An ROI is kept if *either* half passes (union logic).  Per-segment
        pass/fail is recorded in ``roi.active_segments``.

        Parameters
        ----------
        video : Video
            Video object with normalized traces
        all_rois : List[ROI]
            All ROIs to filter
        roi_model : RandomForestClassifier | LogisticRegression
            Trained classifier
        model_config : dict, optional
            Config with 'transform' key, by default None
            
        Returns
        -------
        good_rois : List[ROI]
            ROIs classified as good
        good_roi_mask : np.ndarray
            Boolean mask of good ROIs
        """

        transform = model_config.get("transform") if model_config else None

        if video.is_concatenated and video.split_frame is not None:
            # --- Concatenated piecemeal filtering ---
            baseline_smoothed = video.baseline_norm_sm_f   # (n_rois, baseline_frames)
            treatment_smoothed = video.treatment_norm_sm_f  # (n_rois, treatment_frames)
            preds_bl, _ = self._get_preds(baseline_smoothed[:,:-2], all_rois, roi_model, transform)
            preds_tx, treatment_feats = self._get_preds(treatment_smoothed, all_rois, roi_model, transform)

            good_roi_mask = np.asarray(preds_bl | preds_tx, dtype=bool)

            self._assign_roi_status(all_rois, good_roi_mask, treatment_feats, preds_bl, preds_tx)
        else:
            smoothed = video.norm_sm_f 
            preds, _ = self._get_preds(smoothed, all_rois, roi_model, transform)
            good_roi_mask = np.asarray(preds, dtype=bool)

            self._assign_roi_status(all_rois, good_roi_mask, None)

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

        fs = float(video.fs)

        for filtered_index, roi in enumerate(good_rois):
            neurons.append(
                Neuron(
                    roi=roi,                       
                    filtered_index=filtered_index,
                    fs=fs,
                )
            )

        video.neurons = neurons
        return neurons
   
    def run(self, video: "Video", roi_model: Any, model_config: Optional[dict] = None) -> ROIReport:
        """
        Populates on video:
          - bad_rois, n_good_rois, n_bad_rois
          - bad_rois_features (optional)
          - neurons (created from good ROIs)
        Returns counts for narration.
        """
        all_rois = self.create_rois(video)
        good_rois, good_roi_mask = self.filter_rois(video, all_rois, roi_model, model_config=model_config)

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
