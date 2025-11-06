"""Pipeline function exports for clean importing."""
# Preprocessing
from .preprocessing import (
    load_suite2p_data,
    compute_cascade_probabilities,
    smooth_cascade_prob
)

# ROI processing
from .roi_processing import (
    extract_roi_features,
    filter_rois,
    scale_roi_features
)

# Spike detection
from .spike_detection import (
    detect_spikes_from_cascade,
    find_spike_peaks,
    create_spike_windows
)

# Spike filtering
from .spike_filtering import (
    extract_spike_features,
    filter_spikes,
    compute_spike_metrics
)

# Neuron grouping
from .neuron_grouping import (
    group_neurons_by_sttc,
    group_neurons_by_dtw,
    compute_sttc_matrix,
    compute_dtw_matrix,
    compare_groupings
)

# IO handlers
from .io_handlers import (
    save_video_summary,
    save_timepoint_summary,
    save_experiment_summary,
    create_excel_report,
    save_filtered_suite2p
)

__all__ = [
    # Preprocessing
    'load_suite2p_data',
    'compute_cascade_probabilities',
    'smooth_cascade_prob',
    # ROI processing
    'extract_roi_features',
    'filter_rois',
    'scale_roi_features',
    # Spike detection
    'detect_spikes_from_cascade',
    'find_spike_peaks',
    'create_spike_windows',
    # Spike filtering
    'extract_spike_features',
    'filter_spikes',
    'compute_spike_metrics',
    # Neuron grouping
    'group_neurons_by_sttc',
    'group_neurons_by_dtw',
    'compute_sttc_matrix',
    'compute_dtw_matrix',
    'compare_groupings',
    # IO handlers
    'save_video_summary',
    'save_timepoint_summary',
    'save_experiment_summary',
    'create_excel_report',
    'save_filtered_suite2p'
]