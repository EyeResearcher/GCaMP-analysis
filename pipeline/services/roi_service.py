from dataclasses import dataclass
from typing import List, Optional, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from classifier_pipeline.datasets import apply_transform
from data_classes.roi import ROI
from data_classes.neuron import Neuron
from typing import TYPE_CHECKING

from pipeline.reports import ROIReport
if TYPE_CHECKING:
    from data_classes.video import Video


def _get_feature_names(model: Any) -> list[str]:
    """
    Get feature names from a trained model.
    
    Parameters
    ----------
    model : Any
        Trained sklearn model
        
    Returns
    -------
    feature_names : list[str]
        Ordered feature names the model was trained on
        
    Raises
    ------
    ValueError
        If model doesn't have feature_names_in_ attribute
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise ValueError("Model was not trained with feature names. Retrain on a DataFrame.")
    return list(names)

def _prepare_features(
    feats_df: pd.DataFrame, 
    model: Any, 
    transform: str = None
) -> pd.DataFrame:
    """
    Prepare features for inference with correct ordering and transform.
    
    Parameters
    ----------
    feats_df : pd.DataFrame
        Raw extracted features
    model : Any
        Trained model with feature_names_in_
    transform : str, optional
        Transform to apply, by default None
        
    Returns
    -------
    X : pd.DataFrame
        Features ready for prediction
        
    Raises
    ------
    ValueError
        If feats_df is missing required columns
    """
    expected = _get_feature_names(model)
    
    missing = set(expected) - set(feats_df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    # Select and reorder to match training
    X = feats_df[expected].copy()
    
    if transform:
        X = apply_transform(X, transform)
    
    return X


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

    def filter_rois(
        self,
        video: "Video",
        all_rois: List[ROI],
        roi_model: RandomForestClassifier | LogisticRegression,
        model_config: Optional[dict] = None,
    ) -> tuple[List[ROI], np.ndarray]:
        """
        Filter ROIs using the trained classifier.
        
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
        
        # Prepare features with correct ordering and transform
        transform = model_config.get("transform") if model_config else None
        X = _prepare_features(feats_df, roi_model, transform)

        preds = roi_model.predict(X).astype(bool)
        good_roi_mask = np.asarray(preds, dtype=bool)

        # Preserve explicitly marked-bad ROIs
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
        fs = float(getattr(video, "fs", None) or video.suite2p_data.get("ops", {}).get("fs", 15.0))

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
