"""Create source-distance mask maps and distance-sorted trace rasters."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from scipy.ndimage import gaussian_filter1d


def _distance_from_source(
    coords: np.ndarray,
    row: pd.Series,
    pixel_size_um: float,
) -> tuple[np.ndarray, str]:
    source = np.asarray([row["source_x_px"], row["source_y_px"]], dtype=float)
    if row["model"] == "planar" and np.isfinite(row["direction_degrees"]):
        angle = math.radians(float(row["direction_degrees"]))
        unit = np.asarray([math.cos(angle), math.sin(angle)])
        distance_px = (coords - source[None, :]) @ unit
        distance_px -= min(0.0, float(distance_px.min()))
        label = "Distance along propagation axis (µm)"
    else:
        distance_px = np.linalg.norm(coords - source[None, :], axis=1)
        label = "Radial distance from fitted origin (µm)"
    return distance_px * pixel_size_um, label


def _mask_image(
    stat: np.ndarray,
    participant_indices: np.ndarray,
    distances_um: np.ndarray,
    ly: int,
    lx: int,
) -> tuple[np.ndarray, colors.Normalize]:
    rgba = np.zeros((ly, lx, 4), dtype=np.float32)
    for roi in stat:
        ypix = np.asarray(roi["ypix"], dtype=int)
        xpix = np.asarray(roi["xpix"], dtype=int)
        valid = (ypix >= 0) & (ypix < ly) & (xpix >= 0) & (xpix < lx)
        rgba[ypix[valid], xpix[valid]] = (0.60, 0.60, 0.60, 0.18)
    norm = colors.Normalize(
        vmin=float(np.nanmin(distances_um)),
        vmax=float(np.nanmax(distances_um))
        if float(np.nanmax(distances_um)) > float(np.nanmin(distances_um))
        else float(np.nanmin(distances_um)) + 1.0,
    )
    colormap = matplotlib.colormaps["viridis"]
    for roi_index, distance in zip(participant_indices, distances_um):
        roi = stat[int(roi_index)]
        ypix = np.asarray(roi["ypix"], dtype=int)
        xpix = np.asarray(roi["xpix"], dtype=int)
        valid = (ypix >= 0) & (ypix < ly) & (xpix >= 0) & (xpix < lx)
        color = colormap(norm(float(distance)))
        rgba[ypix[valid], xpix[valid]] = color
    return rgba, norm


def plot_episode(
    row: pd.Series,
    output_path: Path,
    *,
    max_traces: int = 55,
    half_window_frames: int = 120,
    trace_smoothing_sigma_frames: float = 0.6,
) -> dict:
    recording = Path(row["recording_path"])
    suite2p = recording / "suite2p" / "plane0"
    fluorescence = np.load(suite2p / "F.npy", mmap_mode="r")
    stat = np.load(suite2p / "stat.npy", allow_pickle=True)
    ops = np.load(suite2p / "ops.npy", allow_pickle=True).item()
    fs = float(ops.get("fs", 15.0))
    ly = int(ops.get("Ly", 1024))
    lx = int(ops.get("Lx", 1024))
    pixel_size = float(row.get("pixel_size_um", 1.0))
    participant_indices = np.asarray(json.loads(row["roi_indices"]), dtype=int)
    onset_frames = np.asarray(json.loads(row["onset_frames"]), dtype=int)
    coords = np.asarray(
        [
            [float(stat[index]["med"][1]), float(stat[index]["med"][0])]
            for index in participant_indices
        ]
    )
    distances_um, distance_label = _distance_from_source(coords, row, pixel_size)
    mask_rgba, norm = _mask_image(
        stat, participant_indices, distances_um, ly=ly, lx=lx
    )

    order = np.argsort(distances_um)
    if order.size > max_traces:
        sampled = np.unique(
            np.round(np.linspace(0, order.size - 1, max_traces)).astype(int)
        )
        order = order[sampled]
    selected_indices = participant_indices[order]
    selected_distances = distances_um[order]
    selected_onsets = onset_frames[order]
    center = int(row["center_frame"])
    start = max(0, center - half_window_frames)
    stop = min(fluorescence.shape[1], center + half_window_frames + 1)
    full_traces = np.asarray(fluorescence[selected_indices], dtype=float)
    trace_mean = full_traces.mean(axis=1, keepdims=True)
    trace_std = full_traces.std(axis=1, keepdims=True)
    full_traces = np.divide(
        full_traces - trace_mean,
        trace_std,
        out=np.zeros_like(full_traces),
        where=trace_std > 0,
    )
    traces = full_traces[:, start:stop]
    if trace_smoothing_sigma_frames > 0:
        traces = gaussian_filter1d(
            traces, sigma=trace_smoothing_sigma_frames, axis=1
        )
    time = (np.arange(start, stop) - center) / fs
    vertical_step = 4.0
    colormap = matplotlib.colormaps["viridis"]

    figure, (mask_axis, trace_axis, relation_axis) = plt.subplots(
        1,
        3,
        figsize=(19, 11),
        gridspec_kw={"width_ratios": [1.0, 1.55, 0.85]},
        constrained_layout=True,
    )
    mask_axis.imshow(mask_rgba, origin="upper")
    mask_axis.scatter(
        row["source_x_px"],
        row["source_y_px"],
        marker="*",
        s=220,
        color="red",
        edgecolor="white",
        linewidth=1.0,
        label="Fitted source",
    )
    if row["model"] == "planar" and np.isfinite(row["direction_degrees"]):
        angle = math.radians(float(row["direction_degrees"]))
        mask_axis.arrow(
            row["source_x_px"],
            row["source_y_px"],
            180 * math.cos(angle),
            180 * math.sin(angle),
            width=5,
            head_width=35,
            head_length=45,
            color="red",
            length_includes_head=True,
        )
    mask_axis.set(
        xlim=(0, lx),
        ylim=(ly, 0),
        xlabel="x (pixels)",
        ylabel="y (pixels)",
        title=f"Participating ROI masks: {len(participant_indices)} cells",
    )
    mask_axis.set_aspect("equal")
    mask_axis.legend(loc="lower right")
    scalar = cm.ScalarMappable(norm=norm, cmap=colormap)
    figure.colorbar(scalar, ax=mask_axis, label=distance_label, fraction=0.046)

    for rank, (trace, distance, onset) in enumerate(
        zip(traces, selected_distances, selected_onsets)
    ):
        offset = rank * vertical_step
        color = colormap(norm(float(distance)))
        trace_axis.plot(time, trace + offset, color=color, linewidth=0.8)
        if start <= onset < stop:
            onset_x = (onset - center) / fs
            onset_y = trace[onset - start] + offset
            trace_axis.vlines(
                onset_x,
                offset - 1.35,
                offset + 1.35,
                color="black",
                linewidth=1.5,
                zorder=3,
            )
            trace_axis.scatter(
                onset_x,
                onset_y,
                s=28,
                facecolor="red",
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
    trace_axis.axvline(0, color="tab:red", linewidth=1, linestyle="--")
    trace_axis.set(
        xlabel="Time relative to population-event center (s)",
        ylabel="Participating cells, near → far",
        title=(
            f"Distance-sorted z-scored fluorescence "
            f"({len(selected_indices)} sampled cells; σ={trace_smoothing_sigma_frames:g} frame)"
        ),
    )
    trace_axis.set_yticks([])
    trace_axis.text(
        0.01,
        0.99,
        f"near: {selected_distances[0]:.0f} µm",
        transform=trace_axis.transAxes,
        va="top",
    )
    trace_axis.text(
        0.01,
        0.01,
        f"far: {selected_distances[-1]:.0f} µm",
        transform=trace_axis.transAxes,
        va="bottom",
    )
    trace_axis.invert_yaxis()
    onset_seconds = (onset_frames - center) / fs
    relation_axis.scatter(
        distances_um,
        onset_seconds,
        c=distances_um,
        cmap=colormap,
        norm=norm,
        s=14,
        alpha=0.65,
        edgecolors="none",
    )
    if len(distances_um) >= 2 and np.ptp(distances_um) > 0:
        coefficients = np.polyfit(distances_um, onset_seconds, deg=1)
        fitted = np.polyval(coefficients, distances_um)
        total = float(np.sum((onset_seconds - onset_seconds.mean()) ** 2))
        residual = float(np.sum((onset_seconds - fitted) ** 2))
        simple_r2 = max(0.0, 1.0 - residual / total) if total > 0 else 0.0
        correlation = float(np.corrcoef(distances_um, onset_seconds)[0, 1])
        line_x = np.linspace(distances_um.min(), distances_um.max(), 200)
        relation_axis.plot(
            line_x,
            np.polyval(coefficients, line_x),
            color="red",
            linewidth=2,
        )
    else:
        simple_r2 = 0.0
        correlation = float("nan")
    relation_axis.axhline(0, color="tab:red", linewidth=1, linestyle="--")
    relation_axis.set(
        xlabel=distance_label,
        ylabel="Detected onset relative to event center (s)",
        title=f"Timing–distance relationship\nr={correlation:.2f}, R²={simple_r2:.2f}",
    )
    relation_axis.grid(alpha=0.2)
    cluster = int(row.get("recurrence_cluster", -1))
    repeat_text = f" · repeat cluster {cluster + 1}" if cluster >= 0 else ""
    figure.suptitle(
        f"{row['treatment']} {row['recording']} · {row['center_seconds']:.1f} s · "
        f"{row['model']} front · ROI R²={row['propagation_r2']:.2f} · "
        f"movie-corroborated{repeat_text}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "day": int(row["day"]),
        "treatment": row["treatment"],
        "recording": row["recording"],
        "center_seconds": float(row["center_seconds"]),
        "model": row["model"],
        "n_participants": int(len(participant_indices)),
        "propagation_r2": float(row["propagation_r2"]),
        "simple_distance_r2": simple_r2,
        "recurrence_cluster": cluster,
        "path": str(output_path),
    }


def _contact_sheet(paths: list[Path], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    thumbnails = []
    for path in paths:
        with Image.open(path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((900, 510))
            thumbnails.append(preview.copy())
    columns = 2
    rows = math.ceil(len(thumbnails) / columns)
    width = max(image.width for image in thumbnails) * columns
    height = max(image.height for image in thumbnails) * rows
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    cell_width = width // columns
    cell_height = height // rows
    for index, image in enumerate(thumbnails):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x, y))
        draw.rectangle(
            (x, y, x + cell_width - 1, y + cell_height - 1),
            outline=(180, 180, 180),
            width=1,
        )
    sheet.save(output_path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-all-significant", action="store_true")
    parser.add_argument("--max-traces", type=int, default=55)
    parser.add_argument("--half-window-frames", type=int, default=120)
    parser.add_argument("--trace-sigma", type=float, default=0.6)
    args = parser.parse_args()
    episodes = pd.read_csv(args.episodes_csv)
    selected = episodes[episodes["significant_wave"].astype(bool)]
    if not args.include_all_significant:
        selected = selected[selected["movie_validated"].astype(bool)]
    records = []
    paths = []
    for _, row in selected.iterrows():
        filename = (
            f"day{int(row['day'])}_{row['treatment']}_{row['recording']}_"
            f"t{float(row['center_seconds']):06.2f}s.png"
        )
        path = args.output_dir / filename
        records.append(
            plot_episode(
                row,
                path,
                max_traces=args.max_traces,
                half_window_frames=args.half_window_frames,
                trace_smoothing_sigma_frames=args.trace_sigma,
            )
        )
        paths.append(path)
    pd.DataFrame(records).to_csv(args.output_dir / "wave_figure_index.csv", index=False)
    if paths:
        _contact_sheet(paths, args.output_dir / "wave_figure_contact_sheet.jpg")
    print(f"Wrote {len(paths)} wave figures to {args.output_dir}")


if __name__ == "__main__":
    main()
