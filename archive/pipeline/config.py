"""Configuration models and loader for the modular pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    """Filesystem locations for the experiment and outputs."""

    root_dir: Path
    output_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None

    def resolve(self, base_path: Optional[Path] = None) -> "ExperimentConfig":
        base = base_path or Path.cwd()
        root = (base / self.root_dir).resolve() if not self.root_dir.is_absolute() else self.root_dir
        output = None
        if self.output_dir is not None:
            output = (base / self.output_dir).resolve() if not self.output_dir.is_absolute() else self.output_dir
        cache = None
        if self.cache_dir is not None:
            cache = (base / self.cache_dir).resolve() if not self.cache_dir.is_absolute() else self.cache_dir
        print("→ using experiment root:", root)
        print("→ using experiment output:", output)
        print("→ using experiment cache:", cache)

        return ExperimentConfig(root_dir=root, output_dir=output, cache_dir=cache)


@dataclass(slots=True)
class ModelConfig:
    """Locations of trained models used throughout the pipeline."""

    roi_model_path: Path
    spike_model_path: Optional[Path] = None
    cascade_model_name: str = "Global_EXC_30Hz_smoothing100ms_high_noise"
    cascade_model_dir: Optional[Path] = None

    def resolve(self, base_path: Optional[Path] = None) -> "ModelConfig":
        base = base_path or Path.cwd()
        roi_model = (base / self.roi_model_path).resolve() if not self.roi_model_path.is_absolute() else self.roi_model_path
        spike_model = None
        if self.spike_model_path is not None:
            if not self.spike_model_path.is_absolute():
                spike_model = (base / self.spike_model_path).resolve()
            else:
                spike_model = self.spike_model_path
        cascade_dir = None
        if self.cascade_model_dir is not None:
            if not self.cascade_model_dir.is_absolute():
                cascade_dir = (base / self.cascade_model_dir).resolve()
            else:
                cascade_dir = self.cascade_model_dir
        return ModelConfig(
            roi_model_path=roi_model,
            spike_model_path=spike_model,
            cascade_model_name=self.cascade_model_name,
            cascade_model_dir=cascade_dir,
        )


@dataclass(slots=True)
class PreprocessingConfig:
    """Parameters for Suite2p data loading and ROI preprocessing."""

    smoothing_sigma_f: float = 2.0
    smoothing_sigma_prob: float = 2.0
    minmax_mode: str = "video"  # or "roi"
    recompute_cascade: bool = False
    cascade_batch_size: Optional[int] = None
    min_spikes_per_roi: int = 0


@dataclass(slots=True)
class SpikeDetectionConfig:
    """Parameters controlling spike detection from Cascade probabilities."""

    prob_sigma: float = 2.0
    window_size: int = 5
    min_prominence: float = 0.05
    min_distance: int = 8
    edge_trim: int = 32


@dataclass(slots=True)
class SpikeFilteringConfig:
    """Parameters for spike-level feature computation and classification."""

    probability_threshold: Optional[float] = 0.5
    feature_columns: tuple[str, ...] = (
        "prominence",
        "prob_height",
        "fluorescence_peak",
        "rise_slope",
        "decay_tau",
        "window_auc",
        "window_width",
        "baseline_delta",
    )


@dataclass(slots=True)
class FeatureExtractionConfig:
    """Parameters controlling neuron-level feature aggregation."""

    baseline_percentile: float = 10.0
    min_spike_count: int = 0


@dataclass(slots=True)
class GroupingConfig:
    """Parameters for STTC computation and downstream grouping."""

    dt_ms: float = 150.0
    sttc_threshold: float = 0.7
    min_group_size: int = 2


@dataclass(slots=True)
class OutputConfig:
    """Preferences for saving intermediate artifacts and summaries."""

    save_intermediate: bool = True
    overwrite_existing: bool = False
    video_summary_filename: str = "video_summary.xlsx"


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration with nested module configs."""

    experiment: ExperimentConfig
    models: ModelConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    spike_detection: SpikeDetectionConfig = field(default_factory=SpikeDetectionConfig)
    spike_filtering: SpikeFilteringConfig = field(default_factory=SpikeFilteringConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    def resolve(self, base_path: Optional[Path] = None) -> "PipelineConfig":
        base = base_path or Path.cwd()
        return PipelineConfig(
            experiment=self.experiment.resolve(base),
            models=self.models.resolve(base),
            preprocessing=self.preprocessing,
            spike_detection=self.spike_detection,
            spike_filtering=self.spike_filtering,
            feature_extraction=self.feature_extraction,
            grouping=self.grouping,
            outputs=self.outputs,
        )


def _as_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _dataclass_from_dict(cls, data: Dict[str, Any]):
    return cls(**data) if data is not None else cls()


def load_config(config_path: Path) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a YAML file."""

    with Path(config_path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    experiment_cfg = raw.get("experiment", {})
    models_cfg = raw.get("models", {})
    preprocessing_cfg = raw.get("preprocessing")
    spike_detection_cfg = raw.get("spike_detection")
    spike_filtering_cfg = raw.get("spike_filtering")
    feature_extraction_cfg = raw.get("feature_extraction")
    grouping_cfg = raw.get("grouping")
    outputs_cfg = raw.get("outputs")

    pipeline = PipelineConfig(
        experiment=_dataclass_from_dict(
            ExperimentConfig,
            {
                "root_dir": _as_path(experiment_cfg.get("root_dir", ".")),
                "output_dir": _as_path(experiment_cfg["output_dir"]) if experiment_cfg.get("output_dir") else None,
                "cache_dir": _as_path(experiment_cfg["cache_dir"]) if experiment_cfg.get("cache_dir") else None,
            },
        ),
        models=_dataclass_from_dict(
            ModelConfig,
            {
                "roi_model_path": _as_path(models_cfg["roi_model_path"]),
                "spike_model_path": _as_path(models_cfg["spike_model_path"]) if models_cfg.get("spike_model_path") else None,
                "cascade_model_name": models_cfg.get("cascade_model_name", "Global_EXC_30Hz_smoothing100ms_high_noise"),
                "cascade_model_dir": _as_path(models_cfg["cascade_model_dir"]) if models_cfg.get("cascade_model_dir") else None,
            },
        ),
        preprocessing=_dataclass_from_dict(PreprocessingConfig, preprocessing_cfg),
        spike_detection=_dataclass_from_dict(SpikeDetectionConfig, spike_detection_cfg),
        spike_filtering=_dataclass_from_dict(SpikeFilteringConfig, spike_filtering_cfg),
        feature_extraction=_dataclass_from_dict(FeatureExtractionConfig, feature_extraction_cfg),
        grouping=_dataclass_from_dict(GroupingConfig, grouping_cfg),
        outputs=_dataclass_from_dict(OutputConfig, outputs_cfg),
    )
    return pipeline.resolve(Path(config_path).parent)
