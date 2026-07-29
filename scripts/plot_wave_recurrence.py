"""Summarize recurrence of fitted propagation patterns within recordings."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    episodes = pd.read_csv(args.episodes_csv)
    significant = episodes[episodes["significant_wave"].astype(bool)].copy()
    clustered = significant[significant["recurrence_cluster"] >= 0].copy()
    summary = (
        clustered.groupby(["day", "treatment", "recording", "recurrence_cluster"])
        .agg(
            n_episodes=("episode_id", "size"),
            n_movie_corroborated=("movie_validated", "sum"),
            first_seconds=("center_seconds", "min"),
            last_seconds=("center_seconds", "max"),
            median_r2=("propagation_r2", "median"),
            median_speed_um_s=("speed_um_s", "median"),
            model=("model", lambda values: ",".join(sorted(set(values)))),
        )
        .reset_index()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "wave_recurrence_summary.csv", index=False)

    recording = significant[
        (significant["recording"] == "1-1_Day10")
        & (significant["treatment"] == "BP")
    ].sort_values("center_seconds")
    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    colors = {-1: "0.65", 0: "#2563eb", 1: "#f97316", 2: "#16a34a"}
    labels = {-1: "No repeat assignment", 0: "Repeat pattern 1", 1: "Repeat pattern 2", 2: "Repeat pattern 3"}
    y_values = {-1: 0, 0: 1, 1: 2, 2: 3}
    for cluster, group in recording.groupby("recurrence_cluster"):
        marker = "o" if (group["model"] == "planar").all() else "D"
        axis.scatter(
            group["center_seconds"],
            [y_values[int(cluster)]] * len(group),
            s=np.where(group["movie_validated"].astype(bool), 115, 55),
            marker=marker,
            facecolors=colors[int(cluster)],
            edgecolors=np.where(group["movie_validated"].astype(bool), "black", colors[int(cluster)]),
            linewidths=np.where(group["movie_validated"].astype(bool), 1.2, 0.5),
            label=labels[int(cluster)],
        )
        for _, row in group.iterrows():
            axis.annotate(
                f"R² {row['propagation_r2']:.2f}",
                (row["center_seconds"], y_values[int(cluster)]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set(
        xlabel="Recording time (s)",
        ylabel="Fitted source/direction cluster",
        title="BP 1-1 Day 10: recurrence of significant propagation patterns",
        yticks=[0, 1, 2, 3],
        yticklabels=[
            "Unique",
            "Pattern 1\nplanar",
            "Pattern 2\nplanar",
            "Pattern 3\nradial",
        ],
        ylim=(-0.55, 3.55),
    )
    axis.grid(axis="x", alpha=0.25)
    axis.text(
        0.99,
        0.02,
        "Large black-edged markers: independently movie-corroborated",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    figure.savefig(args.output_dir / "wave_recurrence_timeline.png", dpi=200)
    plt.close(figure)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
