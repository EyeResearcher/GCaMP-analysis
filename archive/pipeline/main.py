"""Entry point for the modular GCaMP analysis pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from joblib import load

from Cascade.cascade2p.cascade_wrapper import CascadePredictor

from .config import PipelineConfig, load_config
from .feature_extraction import compute_neuron_features
from .io_handlers import (
    VideoArtifacts,
    compile_timepoint_summary,
    write_timepoint_outputs,
    write_video_outputs,
)
from .neuron_grouping import group_neurons
from .preprocessing import run_preprocessing
from .spike_detection import detect_spikes
from .spike_filtering import filter_spikes


def _load_models(config: PipelineConfig) -> tuple:
    roi_model = load(config.models.roi_model_path)
    spike_model = load(config.models.spike_model_path) if config.models.spike_model_path else None

    cascade_dir = config.models.cascade_model_dir
    if cascade_dir is None:
        cascade_dir = Path(__file__).resolve().parent.parent / "Cascade" / "Pretrained_models"

    cascade_model = CascadePredictor(
        model_name=config.models.cascade_model_name,
        model_folder=str(cascade_dir),
    )
    return roi_model, spike_model, cascade_model


def _iter_timepoints(root_dir: Path) -> List[Path]:
    return [p for p in sorted(root_dir.iterdir()) if p.is_dir()]


def _iter_videos(timepoint_dir: Path) -> List[Path]:
    return [p for p in sorted(timepoint_dir.iterdir()) if p.is_dir()]


def process_video(
    video_path: Path,
    config: PipelineConfig,
    cascade_model: CascadePredictor,
    roi_model,
    spike_model,
) -> tuple[VideoArtifacts, pd.DataFrame]:
    preprocessed = run_preprocessing(
        video_path=video_path,
        cascade_model=cascade_model,
        roi_model=roi_model,
        config=config.preprocessing,
        probability_threshold=None,
    )

    detection = detect_spikes(
        fluorescence=preprocessed.fluorescence,
        cascade_prob=preprocessed.cascade_prob,
        config=config.spike_detection,
    )

    fs = float(preprocessed.summary.ops.get("fs", 30.0))
    filtering = filter_spikes(
        fluorescence=preprocessed.fluorescence,
        detection=detection.spikes_per_neuron,
        fs=fs,
        config=config.spike_filtering,
        spike_model=spike_model,
    )

    features = compute_neuron_features(
        fluorescence=preprocessed.fluorescence,
        filtering_result=filtering,
        ops=preprocessed.summary.ops,
        config=config.feature_extraction,
    )

    nframes = int(preprocessed.summary.ops.get("nframes", preprocessed.fluorescence.shape[-1]))
    grouping = group_neurons(
        spike_indices=features.spike_trains,
        summary_stat=preprocessed.summary.stat,
        fs=fs,
        nframes=nframes,
        config=config.grouping,
    )

    artifacts = write_video_outputs(
        video_path=video_path,
        roi_table=preprocessed.roi_table,
        spike_table=filtering.table,
        neuron_table=features.neuron_table,
        grouping=grouping,
        config=config.outputs,
    )
    return artifacts, features.neuron_table


def run_pipeline(config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    roi_model, spike_model, cascade_model = _load_models(config)

    experiment_records: List[Dict[str, object]] = []
    timepoint_outputs: Dict[str, pd.DataFrame] = {}

    for timepoint_dir in _iter_timepoints(config.experiment.root_dir):
        video_artifacts: List[VideoArtifacts] = []
        video_tables: Dict[str, pd.DataFrame] = {}
        for video_dir in _iter_videos(timepoint_dir):
            artifacts, neuron_table = process_video(
                video_path=video_dir,
                config=config,
                cascade_model=cascade_model,
                roi_model=roi_model,
                spike_model=spike_model,
            )
            video_artifacts.append(artifacts)
            video_tables[artifacts.video_id] = neuron_table
            record = dict(artifacts.summary_row)
            record["timepoint"] = timepoint_dir.name
            experiment_records.append(record)

        summary_df = compile_timepoint_summary(video_artifacts)
        write_timepoint_outputs(
            experiment_name=config.experiment.root_dir.name,
            timepoint_name=timepoint_dir.name,
            timepoint_path=timepoint_dir,
            summary_df=summary_df,
            video_tables=video_tables,
        )
        timepoint_outputs[timepoint_dir.name] = summary_df

    experiment_summary = pd.DataFrame.from_records(experiment_records)
    if not experiment_summary.empty:
        experiment_summary = experiment_summary.set_index(["timepoint", "video_id"])
    output_path = None
    if config.experiment.output_dir is not None:
        config.experiment.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = config.experiment.output_dir / f"{config.experiment.root_dir.name}_summary.xlsx"
        with pd.ExcelWriter(output_path) as writer:
            experiment_summary.to_excel(writer, sheet_name="Experiment_Summary")
            for timepoint_name, df in timepoint_outputs.items():
                df.to_excel(writer, sheet_name=timepoint_name[:31])

    return {
        "experiment_summary": experiment_summary,
        "experiment_summary_path": output_path,
        "timepoint_summaries": timepoint_outputs,
    }


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular GCaMP pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "pipeline_config.yaml",
        help="Path to the pipeline configuration YAML",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> Dict[str, pd.DataFrame]:
    args = parse_args(argv)
    config = load_config(args.config)
    return run_pipeline(config)


if __name__ == "__main__":  # pragma: no cover
    main()
