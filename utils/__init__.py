"""
Utility functions for GCaMP analysis pipeline.
"""

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
    # Statistics
    'compute_cohen_d',
    'compute_hedges_g',
    'perform_permutation_test',
    'compute_bootstrap_ci',
    'compare_distributions',
    'multiple_comparison_correction',
    'detect_outliers',
    'compute_correlation_significance',
]
