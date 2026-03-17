from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Any, Optional

from gcamp_analysis.data_classes.video import Video
from gcamp_analysis.roi_processing.traces import TraceService
from gcamp_analysis.roi_processing.filtering import ROIService
from gcamp_analysis.spike_processing.filtering import SpikeService
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

    # Concatenated-video support
    is_concatenated: bool = False
    split_frame: Optional[int] = None

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

        # Load manual ROI labels if configured
        manual_labels = None
        manual_path = config.get("models", {}).get("roi_manual_labels_path")
        if manual_path:
            manual_path = Path(manual_path)
            if manual_path.exists():
                manual_labels = np.load(manual_path, allow_pickle=True).item()

        roi = ROIService(n_jobs=n_jobs, manual_labels=manual_labels)
        spike = SpikeService(n_jobs=n_jobs)
        # Resolve grouping strategies from config
        grouping_cfg = config.get("grouping", {})
        strategies = grouping_cfg.get("strategies", ["corr"])

        # Filter out strategies whose per-strategy config has enabled: false
        strategies = [
            s for s in strategies
            if grouping_cfg.get(s, {}).get("enabled", True)
        ]

        grp = GroupingService(strategies=strategies)

        # Concatenated-video support
        concat_cfg = config.get("concatenated", {})
        is_concat = bool(concat_cfg.get("enabled", False))
        split_frame = concat_cfg.get("split_frame", None)
        if is_concat and split_frame is not None:
            split_frame = int(split_frame)

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
            is_concatenated=is_concat,
            split_frame=split_frame,
        )

    def run(self, video: "Video", verbose: bool = True) -> None:
        if verbose:
            print(f"\n Processing: {video.video_id}")

        # Set concatenated mode on the video object
        video.is_concatenated = self.is_concatenated
        video.split_frame = self.split_frame

        if video.is_concatenated and verbose:
            print(f"  Concatenated mode: split at frame {video.split_frame}")

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
            parts = [f"Grouping ({'+'.join(gr.strategies_run)}):"]
            for name, count in gr.n_groups.items():
                parts.append(f"{name}={count}")
            if gr.agreements:
                for pair, val in gr.agreements.items():
                    parts.append(f"{pair}={val:.2f}")
            print("  " + " | ".join(parts))

            # Treatment comparison summary
            if video.is_concatenated and video.treatment_comparison_results:
                for strat_name, tc in video.treatment_comparison_results.items():
                    n_groups_tc = len(tc.group_metrics)
                    mean_delta = np.nanmean([
                        gm.get("delta_mean_corr", float("nan"))
                        for gm in tc.group_metrics
                    ]) if tc.group_metrics else float("nan")
                    n_subs = sum(
                        gm.get("n_treatment_subgroups", 0)
                        for gm in tc.group_metrics
                    )
                    print(
                        f"  Treatment comparison ({strat_name}): "
                        f"{n_groups_tc} groups | "
                        f"mean Δcorr={mean_delta:+.3f} | "
                        f"{n_subs} surviving sub-groups"
                    )


