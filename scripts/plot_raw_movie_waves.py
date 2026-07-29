"""Plot movie-gradient candidates and fitted block-arrival maps."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from gcamp_analysis.waves.raw_movie import (
    RawMovieWaveConfig,
    preprocess_block_movie,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=Path)
    return parser.parse_args()


def arrival_map(
    temporal: np.ndarray,
    center: int,
    config: RawMovieWaveConfig,
) -> np.ndarray:
    start = max(0, center - config.fit_half_window_frames)
    stop = min(temporal.shape[0], center + config.fit_half_window_frames + 1)
    window = np.maximum(np.asarray(temporal[start:stop]), 0.0)
    amplitude = window.max(axis=0)
    arrival = np.argmax(window, axis=0)
    active = amplitude >= np.quantile(amplitude, config.active_block_quantile)
    active &= arrival > 0
    active &= arrival < window.shape[0] - 1
    result = np.full(amplitude.shape, np.nan)
    result[active] = arrival[active] + start - center
    return result


def main() -> None:
    args = parse_args()
    config = RawMovieWaveConfig()
    episodes = pd.read_csv(args.analysis_dir / "movie_neighbor_confirmation.csv")
    candidates = pd.read_csv(args.analysis_dir / "movie_wave_candidates.csv")
    metrics = pd.read_csv(args.analysis_dir / "movie_gradient_metrics.csv")
    temporal = np.load(
        args.analysis_dir / "movie_temporal_gradient.npy", mmap_mode="r"
    )
    selected_frames = [282, 657, 824]
    figure = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=[0.78, 1.0])
    axis_early = figure.add_subplot(grid[0, :2])
    axis_late = figure.add_subplot(grid[0, 2])
    for axis, start, stop in (
        (axis_early, 0, 300),
        (axis_late, 650, 850),
    ):
        frames = np.arange(start, stop)
        axis.plot(
            frames,
            metrics.loc[start : stop - 1, "discovery_score"],
            color="#3b4c5c",
            linewidth=1.0,
        )
        subset = candidates[
            candidates["center_frame"].between(start, stop - 1)
        ]
        accepted = subset[subset["movie_significant_wave"]]
        axis.scatter(
            accepted["center_frame"],
            accepted["discovery_score"],
            color="#c23b46",
            s=36,
            zorder=4,
            label="Accepted propagation" if start == 0 else None,
        )
        rejected = subset[~subset["movie_significant_wave"]]
        if not rejected.empty:
            axis.scatter(
                rejected["center_frame"],
                rejected["discovery_score"],
                facecolors="none",
                edgecolors="#8493a0",
                s=28,
                linewidths=0.8,
                zorder=4,
                label="Tested, rejected" if start == 0 else None,
            )
        for frame in selected_frames:
            if start <= frame < stop:
                axis.axvline(frame, color="#d8912d", linewidth=1.2)
        axis.set_xlim(start, stop)
        axis.set_xlabel("Frame")
        axis.set_ylabel("Gradient discovery score")
        axis.spines[["top", "right"]].set_visible(False)
    axis_early.set_title("Frames 0–300")
    axis_late.set_title("Frames 650–850")
    axis_early.legend(frameon=False, loc="upper left")
    for column, frame in enumerate(selected_frames):
        axis = figure.add_subplot(grid[1, column])
        image = arrival_map(temporal, frame, config)
        row = episodes.loc[episodes["center_frame"].eq(frame)].iloc[0]
        shown = axis.imshow(
            image,
            cmap="turbo",
            vmin=-config.fit_half_window_frames,
            vmax=config.fit_half_window_frames,
            interpolation="nearest",
            origin="upper",
        )
        source_x = row["movie_source_x_px"] / config.block_size_pixels - 0.5
        source_y = row["movie_source_y_px"] / config.block_size_pixels - 0.5
        axis.scatter(
            source_x,
            source_y,
            marker="*",
            s=180,
            color="white",
            edgecolor="black",
            linewidth=1.0,
        )
        axis.set_title(
            f"Frame {frame} ({frame / 15:.1f} s)\n"
            f"{row['movie_model']}, movie R²={row['movie_propagation_r2']:.2f}; "
            f"ROI p={row['xcorr_propagation_p']:.3f}"
        )
        axis.set_xlabel("Movie block x")
        axis.set_ylabel("Movie block y")
        colorbar = figure.colorbar(shown, ax=axis, shrink=0.78)
        colorbar.set_label("Arrival relative to center (frames)")
    figure.suptitle(
        "Raw-movie spatiotemporal-gradient candidates in 1-1_Day10",
        fontsize=14,
    )
    figure.savefig(
        args.analysis_dir / "movie_gradient_candidate_diagnostics.png",
        dpi=180,
    )
    plt.close(figure)

    block_movie = np.load(args.analysis_dir / "block_movie.npy", mmap_mode="r")
    processed = preprocess_block_movie(block_movie, config)
    start, stop = 268, 307
    scale = float(np.quantile(np.abs(processed[start:stop]), 0.995))
    animation_figure, animation_axis = plt.subplots(figsize=(5, 5))
    image = animation_axis.imshow(
        processed[start],
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
        interpolation="bicubic",
    )
    animation_axis.set_axis_off()
    title = animation_axis.set_title("")
    animation_figure.colorbar(image, ax=animation_axis, label="Global-median-subtracted dF/F")

    def update(frame: int):
        image.set_data(processed[frame])
        title.set_text(f"1-1_Day10 · frame {frame} · {frame / 15:.2f} s")
        return image, title

    animation = FuncAnimation(
        animation_figure,
        update,
        frames=range(start, stop, 2),
        interval=2000 / 15,
        blit=False,
    )
    animation.save(
        args.analysis_dir / "frame_282_movie_front.gif",
        writer=PillowWriter(fps=15),
        dpi=80,
    )
    plt.close(animation_figure)


if __name__ == "__main__":
    main()
