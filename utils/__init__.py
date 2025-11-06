"""
Utility functions for GCaMP analysis pipeline.
"""

# I/O utilities
from .io_utils import (
    load_npy_file,
    find_suite2p_folders,
    load_experiment_structure,
    SummaryFiles
)

# Cascade utilities
from .cascade_utils import (
    CascadeWrapper,
    batch_predict_cascade,
    load_cascade_predictions,
    load_cascade_model
)

# Statistical utilities
from .stats_utils import (
    compute_cohen_d,
    compute_hedges_g,
    perform_permutation_test,
    compute_bootstrap_ci,
    compare_distributions,
    multiple_comparison_correction,
    detect_outliers,
    compute_correlation_significance
)

__all__ = [
    # I/O
    'load_npy_file',
    'find_suite2p_folders',
    'load_experiment_structure',
    'SummaryFiles',
    # Cascade
    'CascadeWrapper',
    'batch_predict_cascade',
    'load_cascade_predictions',
    # Visualization
    'plot_neuron_traces',
    'plot_spike_raster',
    'plot_correlation_matrix',
    'plot_group_comparison',
    'create_summary_figure',
    # Statistics
    'compute_cohen_d',
    'compute_hedges_g',
    'perform_permutation_test',
    'compute_bootstrap_ci',
    'compare_distributions',
    'multiple_comparison_correction',
    'detect_outliers',
    'compute_correlation_significance'
]
