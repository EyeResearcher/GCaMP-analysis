from pathlib import Path
from data_classes.timepoint import Timepoint
import pandas as pd

class Experiment:
    def __init__(self, experiment_path: str, roi_model, cascade_model, fs = 30):
        """Initialize Experiment with timepoints."""
        self.path = Path(experiment_path)
        self.roi_model = roi_model
        self.name = self.path.name
        self.timepoint_paths = [p for p in self.path.iterdir() if p.is_dir()]
        self.timepoints = [Timepoint(folder, roi_model, cascade_model, fs) for folder in self.timepoint_paths]
        self.summary_df = pd.DataFrame()
    
    def process_all_timepoints(self):
        for tp in self.timepoints:
            tp._timepoint_main()

    def aggregate_summary(self):
        """
        Aggregate summaries across all timepoints into a single DataFrame.
        """
        records = []
        for tp in self.timepoints:
            # Use the Timepoint summary DataFrame
            tp_df = tp.video_summary_neurons
            for video_id, row in tp_df.iterrows():
                rec = {'timepoint': tp.name, 'video_id': video_id}
                rec.update(row.to_dict())
                records.append(rec)
        return pd.DataFrame(records)
