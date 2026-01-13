from __future__ import annotations

from pathlib import Path

# ---- Pipeline pieces
from pipeline.video_runner import VideoPipelineRunner
from pipeline.services.trace_service import TraceService
from pipeline.services.roi_service import ROIService
from pipeline.services.spike_service import SpikeService
from pipeline.services.grouping_service import GroupingService

# ---- Models/config
from pipeline.io_handlers import load_config, load_models  # if you already have these

# ---- Experiment tree + batch
from experiments.tree import ExperimentTreeBuilder, is_video_dir
from experiments.processor import ExperimentProcessor
from experiments.compare import ExperimentComparer, BasicSiblingComparator


def build_runner(config: dict) -> VideoPipelineRunner:
    """Construct the runner once, reuse for all videos in the experiment."""
    n_jobs = config.get("parallel", {}).get("n_jobs", -1)

    trace = TraceService(
        smooth_sigma=config.get("traces", {}).get("smooth_sigma", 4.0),
        sensor_type=config.get("traces", {}).get("sensor_type", "gcamp8s"),
    )
    roi = ROIService(
        n_jobs=n_jobs,
        roi_config_path=Path(config["models"]["roi_config_path"]) if config.get("models", {}).get("roi_config_path") else None,
    )
    spike = SpikeService(
        n_jobs=n_jobs,
        spike_config_path=Path(config["models"]["spike_config_path"]) if config.get("models", {}).get("spike_config_path") else None,
    )
    grouping = GroupingService(
        enable_dtw=config.get("grouping", {}).get("enable_dtw", False),
    )

    return VideoPipelineRunner(trace=trace, roi=roi, spike=spike, grouping=grouping)


def main(
    experiment_root: Path,
    output_root: Path | None = None,
    verbose: bool = True,
) -> None:
    """
    Run the pipeline across an entire experiment directory tree, then compare siblings.
    """

    experiment_root = Path(experiment_root)
    output_root = Path(output_root) if output_root else experiment_root

    # 1) Load config + models once
    config = load_config()
    models = load_models(config)

    # 2) Build runner once
    runner = build_runner(config)

    # 3) Build experiment tree (directory-agnostic)
    builder = ExperimentTreeBuilder(is_video_dir=is_video_dir)
    tree = builder.build(experiment_root)

    # 4) Process all video leaves and attach payloads
    processor = ExperimentProcessor(
        runner=runner,
        models=models,
        config=config,
        output_root=output_root,
    )
    processor.process_tree(tree, verbose=verbose)

    # 5) Compare siblings at every internal node (treatment vs treatment, week vs week, etc.)
    comparer = ExperimentComparer(comparator=BasicSiblingComparator())
    sibling_tables = comparer.compare_all(tree)

    # 6) Print a few summary tables (optional)
    if verbose:
        print("\n=== Sibling comparisons (by node) ===")
        # Print the top-level node comparison if present
        if experiment_root in sibling_tables:
            print(f"\nNode: {experiment_root}")
            print(sibling_tables[experiment_root].to_string(index=False))

        # Print one level down comparisons too (often treatments)
        for node_path, df in sibling_tables.items():
            if node_path == experiment_root:
                continue
            # keep output readable: only print “interesting” nodes
            if len(df) >= 2:
                print(f"\nNode: {node_path}")
                print(df.to_string(index=False))

    # 7) (Optional) Save sibling tables to disk
    # You can store these under output_root / "_comparisons"
    comparisons_dir = output_root / "_comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    for node_path, df in sibling_tables.items():
        safe_name = "_".join(node_path.relative_to(output_root).parts) if node_path.is_relative_to(output_root) else node_path.name
        out_csv = comparisons_dir / f"{safe_name}_siblings.csv"
        df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    # Example usage
    main(
        experiment_root=Path(r"X:\data\Experiment337"),
        output_root=None,   # defaults to experiment_root
        verbose=True,
    )
