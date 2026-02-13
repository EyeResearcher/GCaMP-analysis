from __future__ import annotations

from pathlib import Path
from typing import Optional

# ---- Pipeline pieces

from pipeline.video_runner import VideoPipelineRunner
from pipeline.services.trace_service import TraceService
from pipeline.services.roi_service import ROIService
from pipeline.services.spike_service import SpikeService
from pipeline.services.grouping_service import GroupingService

# ---- Models/config
from utils.io_utils import load_config, load_model

# ---- Experiment tree + batch
from experiments.io import save_node_level_comparisons_with_legend
from experiments.tree import ExperimentTreeBuilder, is_video_dir
from experiments.processor import ExperimentProcessor
from experiments.compare import ExperimentComparer, BasicSiblingComparator




def main(
    experiment_root: Path,
    config_path: Path = Path("config.yaml"),
    output_root: Path | None = None,
    verbose: bool = True,
) -> None:
    """
    Run the pipeline across an entire experiment directory tree, then compare siblings.
    """

    experiment_root = Path(experiment_root)
    output_root = Path(output_root) if output_root else experiment_root

    # 1) Load config + models once
    config = load_config(config_path)

    roi_model, roi_cfg = load_model(config["models"], which="roi")
    spike_model, spike_cfg = load_model(config["models"], which="spike")

    models = {
        "roi": roi_model,
        "roi_config": roi_cfg,
        "spike": spike_model,
        "spike_config": spike_cfg,
    }
    # 2) Build runner once
    runner = VideoPipelineRunner.build(config)

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

    save_node_level_comparisons_with_legend(
        root=tree,
        sibling_tables=sibling_tables,
        output_subdir="metrics",
        filename="sibling_comparisons.xlsx",
    )
    # 6) Print a few summary tables (optional)
    if verbose:
        print("\n=== Sibling comparisons (by node) ===")
        
        if experiment_root in sibling_tables:
            print(f"\nNode: {experiment_root}")
            print(sibling_tables[experiment_root].to_string(index=False))

       
        for node_path, df in sibling_tables.items():
            if node_path == experiment_root:
                continue
    
            if len(df) >= 2:
                print(f"\nNode: {node_path}")
                print(df.to_string(index=False))

   


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GCaMP analysis pipeline")
    parser.add_argument("experiment_root", type=Path, help="Root directory of the experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline_config.yaml"),
        help="Path to pipeline config YAML (default: config/pipeline_config.yaml)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output root (defaults to experiment_root)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    main(
        experiment_root=args.experiment_root,
        config_path=args.config,
        output_root=args.output,
        verbose=not args.quiet,
    )
