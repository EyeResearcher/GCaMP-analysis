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
    manual_labels: Optional[dict] = None

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
        feats: Optional[list],
        section_predictions: Optional[dict[str, np.ndarray]] = None,
        section_features: Optional[dict[str, list]] = None,
    ) -> None:
        baseline_features = None
        if section_features:
            baseline_features = section_features.get("baseline")

        for i, roi in enumerate(rois):
            if feats:
                roi.features = feats[i]
            elif section_predictions is not None:
                roi.active_segments = {
                    section_key: bool(preds[i])
                    for section_key, preds in section_predictions.items()
                }

                chosen_features = None
                if baseline_features is not None and roi.active_segments.get("baseline", False):
                    chosen_features = baseline_features[i]
                else:
                    for section in section_features or {}:
                        if roi.active_segments.get(section, False):
                            chosen_features = section_features[section][i]
                            break
                if chosen_features is None and baseline_features is not None:
                    chosen_features = baseline_features[i]
                if chosen_features is not None:
                    roi.features = chosen_features
            if roi.is_good is False:
                good_roi_mask[i] = False
                roi.is_good = False
                continue  
            roi.is_good = bool(good_roi_mask[i])
    def filter_rois_manual(
        self,
        video: "Video",
        all_rois: List[ROI],
    ) -> tuple[List[ROI], np.ndarray]:
        """Filter ROIs using pre-existing manual labels.

        Looks up each ROI by ``{video.video_id}_{roi.index}`` in
        ``self.manual_labels``.  ROIs with label value 1 are kept;
        everything else (0, -1, or missing) is rejected.
        """
        from utils.label_utils import get_label_value

        good_roi_mask = np.zeros(len(all_rois), dtype=bool)
        for i, roi in enumerate(all_rois):
            key = f"{video.video_id}_{roi.index}"
            entry = self.manual_labels.get(key)
            if entry is not None:
                good_roi_mask[i] = get_label_value(entry.get("label", -1)) == 1

        self._assign_roi_status(all_rois, good_roi_mask, None)
        return [roi for roi in all_rois if roi.is_good], good_roi_mask

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
        parsed concat section and the classifier is run on each section.
        An ROI is kept if *any* section passes (union logic). Per-section
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

        if video.is_concatenated and video.concat_sections:
            section_predictions: dict[str, np.ndarray] = {}
            section_features: dict[str, list] = {}

            for section in video.concat_sections:
                smoothed = video.section_traces[section.section_key]["norm_sm_f"]
                if section.section_key == "baseline" and smoothed.shape[1] > 2:
                    section_input = smoothed[:, :-2]
                else:
                    section_input = smoothed
                preds, feats = self._get_preds(section_input, all_rois, roi_model, transform)
                section_predictions[section.section_key] = np.asarray(preds, dtype=bool)
                section_features[section.section_key] = feats

            good_roi_mask = np.zeros(len(all_rois), dtype=bool)
            for preds in section_predictions.values():
                good_roi_mask |= preds

            self._assign_roi_status(
                all_rois,
                good_roi_mask,
                None,
                section_predictions=section_predictions,
                section_features=section_features,
            )
        else:
            smoothed = video.norm_sm_f 
            preds, feats = self._get_preds(smoothed, all_rois, roi_model, transform)
            good_roi_mask = np.asarray(preds, dtype=bool)

            self._assign_roi_status(all_rois, good_roi_mask, feats)

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
        if self.manual_labels is not None:
            good_rois, good_roi_mask = self.filter_rois_manual(video, all_rois)
        else:
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
