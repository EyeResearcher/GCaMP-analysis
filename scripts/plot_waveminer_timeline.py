"""Plot threshold-stable WaveMiner-compatible event proposals."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.results / "threshold_stable_level_components.csv")
    output = args.output or args.results / "threshold_stable_timeline.png"

    fig, (timeline, metrics) = plt.subplots(
        2,
        1,
        figsize=(13, 6.5),
        gridspec_kw={"height_ratios": [1, 1.5]},
        constrained_layout=True,
    )
    for _, row in data.iterrows():
        start = row["consensus_start_frame"] / 15
        end = row["consensus_end_frame"] / 15
        event = int(row["event_id"])
        timeline.plot([start, end], [event, event], linewidth=7, color="#377eb8")
        timeline.text((start + end) / 2, event + 0.22, str(event), ha="center")
    timeline.set_xlim(0, 1818 / 15)
    timeline.set_yticks([])
    timeline.set_xlabel("recording time (s)")
    timeline.set_title(
        "12 threshold-stable x–y–t component proposals "
        "(overlap in time does not imply separate waves)"
    )
    timeline.grid(axis="x", alpha=0.25)

    x = data["event_id"].to_numpy()
    area = data["area_mm2_middle"].to_numpy()
    speed = data["leading_speed_um_s_middle"].to_numpy()
    r2 = data["arrival_r2_middle"].to_numpy()
    metrics.bar(x - 0.22, area, width=0.42, label="footprint area (mm²)")
    metrics.bar(x + 0.22, r2, width=0.42, label="arrival-field R²")
    metrics.set_xticks(x)
    metrics.set_xlabel("event proposal")
    metrics.set_ylabel("area or R²")
    metrics.legend(loc="upper left")
    speed_axis = metrics.twinx()
    speed_axis.plot(x, speed, "o-", color="#e41a1c", label="leading-edge speed")
    speed_axis.set_ylabel("leading-edge speed (µm/s)")
    speed_axis.legend(loc="upper right")
    fig.suptitle(
        "1-1 Day10 WaveMiner-compatible proposals at the middle threshold (z≥2.5)"
    )
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
