from dataclasses import dataclass
from typing import Any, Optional
import time
from data_classes.video import Video
from pipeline.services import grouping_service, roi_service, spike_service, trace_service

@dataclass
class VideoPipelineRunner:
    def __init__(self, trace : trace_service.TraceService, roi: roi_service.ROIService,
                spike: spike_service.SpikeService, grouping: grouping_service.GroupingService):
        self.trace = trace
        self.roi = roi
        self.spike = spike
        self.grouping = grouping

    def run(self, video: "Video", models: dict[str, Any], config: dict, verbose: bool = True) -> None:
        if verbose:
            print(f"\n Processing: {video.video_id}")

        # Step 1–2: traces
        tr = self.trace.run(video)
        if verbose:
            print(f"  Traces: {tr.n_rois} ROIs, {tr.n_frames} frames @ {tr.fs:.1f} Hz")

        # Step 3–4: ROI filtering
        rr = self.roi.run(video, models["roi_classifier"])
        if verbose:
            print(f"  ROI filter: {rr.n_rois_good}/{rr.n_rois_total} kept ({rr.pass_rate:.1%})")

        if rr.n_rois_good == 0:
            if verbose:
                print("  No ROIs kept — skipping spikes and grouping.")
            return

        # Step 5–6: spikes
        sr = self.spike.run(video, models["spike_classifier"])
        if verbose:
            print(
                f"  Spikes: {sr.n_spikes_kept}/{sr.n_spikes_raw} kept | "
                f"neurons {sr.n_neurons_in} → {sr.n_neurons_out}"
            )

        if sr.n_neurons_out < 2:
            if verbose:
                print("  <2 neurons with spikes — skipping grouping.")
            return

        # Step 8: grouping
        gr = self.grouping.run(video, config.get("grouping", {}))
        if verbose:
            if gr.agreement is None:
                print(f"  Grouping ({gr.method}): {gr.n_groups} groups")
            else:
                print(f"  Grouping ({gr.method}): {gr.n_groups} groups | agreement={gr.agreement:.2f}")
        self.grouping.visualize(video, which="sttc")

        return video
