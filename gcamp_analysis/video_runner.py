from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gcamp_analysis.data_classes.video import Video
from gcamp_analysis.grouping_processing.service import GroupingService
from gcamp_analysis.roi_processing.filtering import ROIService
from gcamp_analysis.roi_processing.traces import TraceService
from gcamp_analysis.spike_processing.filtering import SpikeService


@dataclass
class VideoPipelineRunner:
    """Runs the full per-video pipeline (traces -> ROIs -> spikes -> grouping)."""

    trace: TraceService
    roi: ROIService
    spike: SpikeService
    grouping: GroupingService

    roi_model: Any = None
    roi_model_config: Optional[dict] = None
    spike_model: Any = None
    spike_model_config: Optional[dict] = None
    grouping_cfg: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        config: dict[str, dict | np.ndarray],
        models: dict[str, Any],
        sensor_type: str | None = None,
    ) -> "VideoPipelineRunner":
        """Construct the runner once, reuse for all videos in the experiment."""
        n_jobs = config.get("parallel", {}).get("n_jobs", -1)
        resolved_sensor = sensor_type or config.get("traces", {}).get("sensor_type", "gcamp8s")

        trace = TraceService(
            smooth_sigma=config.get("traces", {}).get("smooth_sigma", 4.0),
            sensor_type=resolved_sensor,
        )

        manual_labels = None
        manual_path = config.get("models", {}).get("roi_manual_labels_path")
        if manual_path:
            manual_path = Path(manual_path)
            if manual_path.exists():
                manual_labels = np.load(manual_path, allow_pickle=True).item()

        roi = ROIService(n_jobs=n_jobs, manual_labels=manual_labels)
        spike = SpikeService(n_jobs=n_jobs)

        grouping_cfg = config.get("grouping", {})
        strategies = grouping_cfg.get("strategies", ["combined"])
        strategies = [
            strategy
            for strategy in strategies
            if grouping_cfg.get(strategy, {}).get("enabled", True)
        ]
        grp = GroupingService(strategies=strategies)

        return cls(
            trace=trace,
            roi=roi,
            spike=spike,
            grouping=grp,
            roi_model=models["roi"],
            roi_model_config=models.get("roi_config"),
            spike_model=models["spike"],
            spike_model_config=models.get("spike_config"),
            grouping_cfg=grouping_cfg,
        )

    def run(self, video: Video, verbose: bool = True) -> None:
        if verbose:
            print(f"\n Processing: {video.video_id}")

        trace_report = self.trace.run(video)
        if verbose:
            print(f"  Traces: {trace_report.n_rois} ROIs, {trace_report.n_frames} frames @ {trace_report.fs:.1f} Hz")

        roi_report = self.roi.run(video, self.roi_model, model_config=self.roi_model_config)
        if verbose:
            print(f"  ROI filter: {roi_report.n_rois_good}/{roi_report.n_rois_total} kept ({roi_report.pass_rate:.1%})")

        if roi_report.n_rois_good == 0:
            if verbose:
                print("  No ROIs kept - skipping spikes and grouping.")
            return

        spike_report = self.spike.run(video, self.spike_model, model_config=self.spike_model_config)
        if verbose:
            print(
                f"  Spikes: {spike_report.n_spikes_kept}/{spike_report.n_spikes_raw} kept | "
                f"neurons {spike_report.n_neurons_in} -> {spike_report.n_neurons_out}"
            )

        if spike_report.n_neurons_out < 2:
            if verbose:
                print("  <2 neurons with spikes - skipping grouping.")
            return

        grouping_report = self.grouping.run(video, self.grouping_cfg)
        if verbose and grouping_report is not None:
            parts = [f"Grouping ({'+'.join(grouping_report.strategies_run)}):"]
            for name, count in grouping_report.n_groups.items():
                parts.append(f"{name}={count}")
            print("  " + " | ".join(parts))

            for strategy_name, result in video.grouping_results.items():
                if not isinstance(getattr(result, "metadata", None), dict):
                    continue
                if result.metadata.get("skipped") and result.metadata.get("reason") == "active_neuron_cap":
                    print(
                        "  "
                        f"Grouping ({strategy_name}) skipped: "
                        f"active_neurons={result.metadata.get('active_neurons')} > "
                        f"cap={result.metadata.get('max_neurons_for_grouping')}"
                    )

