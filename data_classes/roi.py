from filtering.feature_utils import four_primary_roi_features
import pandas as pd
from utils.io_utils import SummaryFiles
class ROI:
    def __init__(self, row_index, summary_files: SummaryFiles, model):
        self.row_index = row_index
        self.summary_files = summary_files 
        self.model = model
        self.features = None
        self.is_good_cell = None

    def _extract_features(self):
        # extract_features should return a dict of feature_name->value
        data_dict = self.summary_files.get_roi_data(self.row_index)
        try:
            feat_dict = four_primary_roi_features(data_dict["F"], data_dict["spike_prob"])
        except Exception:
            print(data_dict["spike_prob"])
            raise ValueError
        # wrap into a single‐row DataFrame for scikit-learn
        self.features = pd.DataFrame([feat_dict])
    def predict_status(self):
        if self.features is None:
            self._extract_features()
        self.is_good_cell = bool(self.model.predict(self.features)[0])

    def _filter_roi(self):
        self._extract_features()
        self.predict_status()