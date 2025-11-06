"""
Analysis package for GCaMP data.
Provides higher-level analysis functions for processed data.
"""

from .group_analysis import (
    analyze_group_stability,
    compute_group_coherence,
    analyze_group_dynamics,
    compare_group_methods
)

from .treatment_comparison import (
    compare_treatments,
    analyze_treatment_effects,
    compute_treatment_statistics,
    plot_treatment_comparison
)

from .temporal_analysis import (
    analyze_temporal_patterns,
    compute_burst_statistics,
    analyze_synchrony_over_time,
    detect_network_events
)

__all__ = [
    'analyze_group_stability',
    'compute_group_coherence',
    'analyze_group_dynamics',
    'compare_group_methods',
    'compare_treatments',
    'analyze_treatment_effects',
    'compute_treatment_statistics',
    'plot_treatment_comparison',
    'analyze_temporal_patterns',
    'compute_burst_statistics',
    'analyze_synchrony_over_time',
    'detect_network_events'
]
