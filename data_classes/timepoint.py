
import sys


from pathlib import Path
from .video import Video
import pandas as pd
import numpy as np

class Timepoint:
    def __init__(self, timepoint_folder: Path, roi_model, cascade_model, fs=30):
        self.path = timepoint_folder
        self.name = self.path.name
        self.experiment = self.path.parent.name
        self.video_paths = [p for p in self.path.iterdir() if p.is_dir()]
        self.videos = [Video(v, roi_model, cascade_model, fs) for v in self.video_paths]
        self.video_summary_neurons = pd.DataFrame()
    def process_all_videos(self):
        for video in self.videos:
            video.video_main()

    def summarize_videos_neurons(self):
        """
        Create a summary DataFrame where each row is a video and columns include:
          - number of cells in video
          - average number of spikes per cell
          - average amplitude of spikes per cell
          - number of groups in video
          - group indices
          - average number of cells per group
          - percent of all neurons in a group
          - average spikes per neuron in groups
          - average amplitude for neurons in groups
          - average spikes per neuron not in groups
          - average amplitude for neurons not in groups
        """
        records = []
        for video in self.videos:
            df = video.summary_df
            num_cells = len(df)
            avg_spikes = df['num_spikes'].mean() if num_cells else np.nan
            avg_amp = df['fluorescence_values'].apply(np.mean).mean() if num_cells else np.nan
            num_groups = len(video.neuron_groups)
            group_idxs = [[n.filtered_index for n in grp] for grp in video.neuron_groups]
            avg_cells_per_group = np.mean([len(grp) for grp in video.neuron_groups]) if num_groups else np.nan
            pct_in_groups = (sum(len(grp) for grp in video.neuron_groups) / num_cells) if num_cells else np.nan
            # flattened group neurons
            group_neurons = [n for grp in video.neuron_groups for n in grp]
            if group_neurons:
                avg_spikes_grp = np.mean([n.num_spikes for n in group_neurons])
                avg_amp_grp = np.mean([n.f_peak_values.mean() for n in group_neurons])
            else:
                avg_spikes_grp = np.nan
                avg_amp_grp = np.nan
            non_group_neurons = [n for n in video.neurons if n not in group_neurons]
            if non_group_neurons:
                avg_spikes_non = np.mean([n.num_spikes for n in non_group_neurons])
                avg_amp_non = np.mean([n.f_peak_values.mean() for n in non_group_neurons])
            else:
                avg_spikes_non = np.nan
                avg_amp_non = np.nan
            rec = {
                'video_id': video.video_id,
                'num_cells': num_cells,
                'avg_spikes_per_cell': avg_spikes,
                'avg_amplitude_per_cell': avg_amp,
                'num_groups': num_groups,
                'group_indices': group_idxs,
                'avg_cells_per_group': avg_cells_per_group,
                'percent_in_groups': pct_in_groups,
                'avg_spikes_in_groups': avg_spikes_grp,
                'avg_amplitude_in_groups': avg_amp_grp,
                'avg_spikes_not_in_groups': avg_spikes_non,
                'avg_amplitude_not_in_groups': avg_amp_non
            }
            records.append(rec)
        self.video_summary_neurons = pd.DataFrame(records).set_index('video_id')
    
    def save_summary_excel(self, filename: str = None) -> Path:
        """
        Delegate to io_utils.save_timepoint_summary_data without passing the Timepoint object directly.

        Args:
            experiment_name: name of the parent experiment
            filename: optional Excel filename; defaults to "<experiment>_<timepoint>_summary.xlsx"
        Returns:
            Path to the generated Excel file
        """
        from utils.io_utils import save_timepoint_summary
        # Assemble data
        # Collect individual video DataFrames in a dict
        video_dfs = {video.video_id: video.summary_df for video in self.videos}
        # Determine output folder
        output_dir = self.path
        # Call I/O utility
        return save_timepoint_summary(
            experiment_name=self.experiment,
            timepoint_name=self.name,
            timepoint_df=self.video_summary_neurons,
            video_dfs=video_dfs,
            output_dir=output_dir,
            filename=filename
        )
    def _timepoint_main(self):
        self.process_all_videos()
        self.summarize_videos_neurons()
        self.save_summary_excel()
    