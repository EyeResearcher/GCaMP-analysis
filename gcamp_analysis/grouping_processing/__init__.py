from .strategies import (
    GroupingResult,
    GroupingStrategy,
    CorrelationStrategy,
    STTCStrategy,
    DTWStrategy,
    STRATEGY_REGISTRY,
)
from .service import (
    GroupingService,
    compute_pairwise_agreement,
    build_combined_summary,
    compute_group_summary_rows,
    make_matrix_heatmap,
    visualize_grouping,
)
from .treatment_comparison import (
    TreatmentComparisonService,
    TreatmentComparisonResult,
    METRIC_REGISTRY as TREATMENT_METRIC_REGISTRY,
)

__all__ = [
    "GroupingResult",
    "GroupingStrategy",
    "CorrelationStrategy",
    "STTCStrategy",
    "DTWStrategy",
    "STRATEGY_REGISTRY",
    "GroupingService",
    "compute_pairwise_agreement",
    "build_combined_summary",
    "compute_group_summary_rows",
    "make_matrix_heatmap",
    "visualize_grouping",
    "TreatmentComparisonService",
    "TreatmentComparisonResult",
    "TREATMENT_METRIC_REGISTRY",
]
