from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Any, Optional

from gcamp_analysis.data_classes.video import Video
from gcamp_analysis.roi_processing.traces import TraceService
from gcamp_analysis.roi_processing.filtering import ROIService
from gcamp_analysis.spike_processing.service import SpikeService
from gcamp_analysis.grouping_processing.service import GroupingService


@dataclass
class VideoPipelineRunner:
    """Runs the full per-video pipeline (traces → ROIs → spikes → grouping).

    Build once via :meth:`build`, then call :meth:`run` for each video.
    Models and grouping config are stored at build time so callers
    only need to pass the ``Video`` object.
    """

    trace: TraceService
    roi: ROIService
    spike: SpikeService
    grouping: GroupingService

    # Models & configs — set once, reused for every video
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
        roi = ROIService(n_jobs=n_jobs)
        spike = SpikeService(n_jobs=n_jobs)
        grp = GroupingService(
            enable_dtw=config.get("grouping", {}).get("enable_dtw", False),
        )

        return cls(
            trace=trace,
            roi=roi,
            spike=spike,
            grouping=grp,
            roi_model=models["roi"],
            roi_model_config=models.get("roi_config"),
            spike_model=models["spike"],
            spike_model_config=models.get("spike_config"),
            grouping_cfg=config.get("grouping", {}),
        )

    def run(self, video: "Video", verbose: bool = True) -> None:
        if verbose:
            print(f"\n Processing: {video.video_id}")

        # Step 1–2: traces
        tr = self.trace.run(video)
        if verbose:
            print(f"  Traces: {tr.n_rois} ROIs, {tr.n_frames} frames @ {tr.fs:.1f} Hz")

        # Step 3–4: ROI filtering
        rr = self.roi.run(video, self.roi_model, model_config=self.roi_model_config)
        if verbose:
            print(f"  ROI filter: {rr.n_rois_good}/{rr.n_rois_total} kept ({rr.pass_rate:.1%})")

        if rr.n_rois_good == 0:
            if verbose:
                print("  No ROIs kept — skipping spikes and grouping.")
            return

        # Step 5–6: spikes
        sr = self.spike.run(video, self.spike_model, model_config=self.spike_model_config)
        if verbose:
            print(
                f"  Spikes: {sr.n_spikes_kept}/{sr.n_spikes_raw} kept | "
                f"neurons {sr.n_neurons_in} → {sr.n_neurons_out}"
            )

        if sr.n_neurons_out < 2:
            if verbose:
                print("  <2 neurons with spikes — skipping grouping.")
            return

        # Step 7: grouping
        gr = self.grouping.run(video, self.grouping_cfg)
        if verbose:
            if gr.agreement is None:
                print(f"  Grouping ({gr.method}): {gr.n_groups} groups")
            else:
                print(f"  Grouping ({gr.method}): {gr.n_groups} groups | agreement={gr.agreement:.2f}")


