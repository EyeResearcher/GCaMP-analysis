"""Dataset discovery, anchor-group selection, and longitudinal reporting."""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion

from gcamp_analysis.recording_discovery import parse_region_day
from .models import CellMatch, RecordingRef, RegistrationResult
from .registration import (
    estimate_snap_translation,
    image_correlation_for_shift,
    match_rois_to_anchor,
    shift_image,
    stat_to_masks,
)


def discover_recordings(experiment_root: Path) -> list[RecordingRef]:
    """Discover ``1-1`` / ``1-1_DayN`` recordings under treatments.

    The complete base name (for example ``1-1`` versus ``1-2``) is the region
    identity. A folder without ``_DayN`` is day 1.
    """
    root = Path(experiment_root)
    recordings: list[RecordingRef] = []
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")
    for treatment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for video_dir in sorted(path for path in treatment_dir.iterdir() if path.is_dir()):
            parsed = parse_region_day(video_dir.name)
            if parsed is None:
                continue
            region, day = parsed
            plane0 = video_dir / "suite2p" / "plane0"
            if not (plane0 / "ops.npy").is_file() or not (plane0 / "stat.npy").is_file():
                continue
            metrics = video_dir / "metrics" / f"{video_dir.name}_metrics.xlsx"
            recordings.append(
                RecordingRef(
                    treatment=treatment_dir.name,
                    region=region,
                    day=day,
                    recording_name=video_dir.name,
                    video_dir=video_dir,
                    plane0_dir=plane0,
                    metrics_path=metrics,
                )
            )
    return sorted(recordings, key=lambda item: (item.treatment, item.region, item.day))


def _load_suite2p(recording: RecordingRef) -> tuple[np.ndarray, np.ndarray]:
    ops = np.load(recording.plane0_dir / "ops.npy", allow_pickle=True).item()
    image = ops.get("meanImg")
    if image is None:
        image = ops.get("refImg")
    if image is None:
        raise KeyError(f"{recording.plane0_dir / 'ops.npy'} has no meanImg or refImg.")
    stat = np.load(recording.plane0_dir / "stat.npy", allow_pickle=True)
    return np.asarray(image, dtype=np.float32), stat


def _find_snap(recording: RecordingRef) -> Path:
    preferred = recording.video_dir / f"{recording.recording_name}_snap.tif"
    if preferred.is_file():
        return preferred
    candidates = sorted(recording.video_dir.glob("*_snap.tif"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Expected exactly one snap TIFF for {recording.recording_name}; "
        f"found {len(candidates)}."
    )


def _parse_indices(value) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(item) for item in value]
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (list, tuple, np.ndarray)):
        raise ValueError(f"Expected a list of neuron indices, got {value!r}.")
    return [int(item) for item in parsed]


def _load_groups(recording: RecordingRef, strategy: str) -> dict[str, set[int]]:
    if not recording.metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics workbook: {recording.metrics_path}")
    try:
        groups = pd.read_excel(recording.metrics_path, sheet_name="grouping_stats")
    except ValueError:
        # The reporting layer omits this sheet when a recording has no groups.
        return {}
    if "method" in groups.columns:
        groups = groups.loc[groups["method"].astype(str) == strategy]
    result: dict[str, set[int]] = {}
    for _, row in groups.iterrows():
        result[str(row["group_id"])] = set(_parse_indices(row["neuron_indices"]))
    return result


def _load_active_neurons(recording: RecordingRef) -> set[int]:
    if not recording.metrics_path.is_file():
        return set()
    try:
        summary = pd.read_excel(recording.metrics_path, sheet_name="spike_summary")
    except ValueError:
        return set()
    if "neuron_idx" not in summary.columns:
        return set()
    return set(pd.to_numeric(summary["neuron_idx"], errors="coerce").dropna().astype(int))


def _normalize_background(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    lo, hi = np.nanpercentile(values, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        gray = np.zeros(values.shape, dtype=np.uint8)
    else:
        gray = (np.clip((values - lo) / (hi - lo), 0, 1) * 180).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _palette(n: int) -> list[tuple[int, int, int]]:
    cmap = plt.get_cmap("tab20", max(n, 1))
    return [tuple(int(round(channel * 255)) for channel in cmap(i)[:3]) for i in range(n)]


def _mask_outline(
    linear_indices: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a visible two-sided outline around a flattened ROI mask."""
    mask = np.zeros(image_shape, dtype=bool)
    yy, xx = np.divmod(np.asarray(linear_indices, dtype=int), image_shape[1])
    valid = (
        (yy >= 0)
        & (yy < image_shape[0])
        & (xx >= 0)
        & (xx < image_shape[1])
    )
    mask[yy[valid], xx[valid]] = True
    outline = binary_dilation(mask, iterations=1) & ~binary_erosion(mask, iterations=1)
    return np.nonzero(outline)


def _write_montage(
    frames: list[np.ndarray],
    days: list[int],
    path: Path,
) -> None:
    """Write a compact chronological PNG preview of a TIFF frame sequence."""
    columns = min(4, max(1, len(frames)))
    rows = int(math.ceil(len(frames) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for axis in axes_array:
        axis.axis("off")
    for axis, frame, day in zip(axes_array, frames, days):
        axis.imshow(frame)
        axis.set_title(f"Day {day}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


@dataclass
class _DayMatches:
    """Per-day matching products shared by the longitudinal reporters."""

    matches_by_day: dict[int, dict[int, CellMatch]]
    inverse_matches_by_day: dict[int, dict[int, int]]
    moving_masks_by_day: dict[int, list[np.ndarray]]
    aligned_images: dict[int, np.ndarray]
    match_rows: list[dict]
    registration_rows: list[dict]


@dataclass
class LongitudinalTracker:
    """Track anchor-day ROI masks and largest functional groups across days."""

    experiment_root: Path
    strategy: str = "combined"
    max_registration_shift: int = 80
    max_centroid_distance: float = 10.0
    min_iou: float = 0.05
    min_match_score: float = 0.24
    ambiguity_margin: float = 0.08

    def recordings_for(self, treatment: str, region: str) -> list[RecordingRef]:
        return [
            recording
            for recording in discover_recordings(self.experiment_root)
            if recording.treatment == treatment and recording.region == region
        ]

    def _sequential_snap_registrations(
        self,
        recordings: list[RecordingRef],
        *,
        anchor_day: int,
        mask_shape: tuple[int, int],
    ) -> tuple[dict[int, RegistrationResult], list[dict], dict[int, dict]]:
        """Compose adjacent snap translations into anchor mask coordinates."""
        snap_paths = {recording.day: _find_snap(recording) for recording in recordings}
        snaps = {
            recording.day: np.asarray(tifffile.imread(snap_paths[recording.day]))
            for recording in recordings
        }
        snap_shapes = {image.shape for image in snaps.values()}
        if len(snap_shapes) != 1:
            raise ValueError(f"Snap shapes differ within one region: {sorted(snap_shapes)}")
        snap_shape = next(iter(snap_shapes))
        if len(snap_shape) != 2:
            raise ValueError(f"Expected 2-D snaps, got shape {snap_shape}.")
        scale_y = mask_shape[0] / snap_shape[0]
        scale_x = mask_shape[1] / snap_shape[1]
        max_snap_shift = int(
            math.ceil(
                self.max_registration_shift / max(min(scale_y, scale_x), 1e-9)
            )
        )

        edges: list[RegistrationResult] = []
        pairwise_rows: list[dict] = []
        for moving, anchor in zip(recordings[:-1], recordings[1:]):
            edge = estimate_snap_translation(
                snaps[anchor.day],
                snaps[moving.day],
                max_shift=max_snap_shift,
            )
            edges.append(edge)
            pairwise_rows.append(
                {
                    "treatment": moving.treatment,
                    "region": moving.region,
                    "moving_day": moving.day,
                    "anchor_day": anchor.day,
                    "moving_snap": str(snap_paths[moving.day]),
                    "anchor_snap": str(snap_paths[anchor.day]),
                    "snap_shift_y_px": edge.shift_y,
                    "snap_shift_x_px": edge.shift_x,
                    "mask_shift_y_px": int(round(edge.shift_y * scale_y)),
                    "mask_shift_x_px": int(round(edge.shift_x * scale_x)),
                    "phase_correlation": edge.correlation,
                }
            )

        anchor_index = next(
            index for index, recording in enumerate(recordings)
            if recording.day == anchor_day
        )
        registrations: dict[int, RegistrationResult] = {}
        details: dict[int, dict] = {}
        for index, recording in enumerate(recordings):
            if index < anchor_index:
                path_edges = edges[index:anchor_index]
                snap_shift_y = sum(edge.shift_y for edge in path_edges)
                snap_shift_x = sum(edge.shift_x for edge in path_edges)
            elif index > anchor_index:
                path_edges = edges[anchor_index:index]
                snap_shift_y = -sum(edge.shift_y for edge in path_edges)
                snap_shift_x = -sum(edge.shift_x for edge in path_edges)
            else:
                path_edges = []
                snap_shift_y = 0
                snap_shift_x = 0
            path_quality = (
                min(edge.correlation for edge in path_edges)
                if path_edges else 1.0
            )
            mask_shift_y = int(round(snap_shift_y * scale_y))
            mask_shift_x = int(round(snap_shift_x * scale_x))
            registrations[recording.day] = RegistrationResult(
                mask_shift_y,
                mask_shift_x,
                float(path_quality),
                "sequential_snap_phase_correlation",
            )
            details[recording.day] = {
                "snap_path": str(snap_paths[recording.day]),
                "snap_shift_y_px": snap_shift_y,
                "snap_shift_x_px": snap_shift_x,
                "snap_to_mask_scale_y": scale_y,
                "snap_to_mask_scale_x": scale_x,
            }
        return registrations, pairwise_rows, details

    def _select_recordings(
        self, treatment: str, region: str
    ) -> tuple[list[RecordingRef], list[dict]]:
        """Split discovered recordings into usable (snap present) and skipped."""
        recordings: list[RecordingRef] = []
        skipped: list[dict] = []
        for recording in self.recordings_for(treatment, region):
            try:
                _find_snap(recording)
            except FileNotFoundError as exc:
                skipped.append(
                    {
                        "treatment": treatment,
                        "region": region,
                        "day": recording.day,
                        "recording": recording.recording_name,
                        "reason": str(exc),
                    }
                )
            else:
                recordings.append(recording)
        if len(recordings) < 2:
            raise ValueError(
                f"Need at least two days for treatment={treatment!r}, "
                f"region={region!r}; found {len(recordings)}."
            )
        return recordings, skipped

    @staticmethod
    def _resolve_anchor(
        recordings: list[RecordingRef], anchor_day: int | None
    ) -> tuple[RecordingRef, int]:
        """Choose the single anchor recording (latest day by default)."""
        days = [recording.day for recording in recordings]
        chosen = max(days) if anchor_day is None else int(anchor_day)
        anchors = [recording for recording in recordings if recording.day == chosen]
        if len(anchors) != 1:
            raise ValueError(
                f"Expected one anchor recording on day {chosen}; found {len(anchors)}."
            )
        return anchors[0], chosen

    @staticmethod
    def _select_anchor_groups(
        anchor_groups: dict[str, set[int]],
        *,
        top_fraction: float,
        top_n: int | None,
    ) -> dict[str, set[int]]:
        """Rank anchor groups by size and keep the requested top fraction/number."""
        ranked_groups = sorted(
            anchor_groups.items(),
            key=lambda item: (-len(item[1]), str(item[0])),
        )
        if top_n is None:
            if not 0 < top_fraction <= 1:
                raise ValueError("top_fraction must be in (0, 1].")
            keep = max(1, int(math.ceil(len(ranked_groups) * top_fraction)))
        else:
            keep = max(1, min(int(top_n), len(ranked_groups)))
        return dict(ranked_groups[:keep])

    def _match_all_days(
        self,
        *,
        recordings: list[RecordingRef],
        anchor_image: np.ndarray,
        anchor_stat: np.ndarray,
        image_shape: tuple[int, int],
        registrations: dict[int, RegistrationResult],
        snap_details: dict[int, dict],
        treatment: str,
        region: str,
        chosen_anchor_day: int,
    ) -> _DayMatches:
        """Match every day's ROIs to the anchor and collect per-day products."""
        result = _DayMatches({}, {}, {}, {}, [], [])
        for recording in recordings:
            moving_image, moving_stat = _load_suite2p(recording)
            if moving_image.shape != anchor_image.shape:
                raise ValueError(
                    f"Image shape differs for {recording.recording_name}: "
                    f"{moving_image.shape} versus anchor {anchor_image.shape}."
                )
            registration = registrations[recording.day]
            if recording.day == chosen_anchor_day:
                matches = [
                    CellMatch(index, index, 1.0, 1.0, 0.0, False)
                    for index in range(len(anchor_stat))
                ]
                moving_masks, _ = stat_to_masks(moving_stat, image_shape)
            else:
                matches, moving_masks = match_rois_to_anchor(
                    anchor_stat,
                    moving_stat,
                    image_shape,
                    registration,
                    max_centroid_distance=self.max_centroid_distance,
                    min_iou=self.min_iou,
                    min_score=self.min_match_score,
                    ambiguity_margin=self.ambiguity_margin,
                )
            result.matches_by_day[recording.day] = {
                match.anchor_roi: match for match in matches
            }
            result.inverse_matches_by_day[recording.day] = {
                match.moving_roi: match.anchor_roi for match in matches
            }
            result.moving_masks_by_day[recording.day] = moving_masks
            result.aligned_images[recording.day] = shift_image(
                moving_image, registration.shift_y, registration.shift_x
            )
            for match in matches:
                result.match_rows.append(
                    {
                        "treatment": treatment,
                        "region": region,
                        "anchor_day": chosen_anchor_day,
                        "day": recording.day,
                        "anchor_roi": match.anchor_roi,
                        "day_roi": match.moving_roi,
                        "match_score": match.score,
                        "mask_iou": match.iou,
                        "centroid_distance_px": match.centroid_distance,
                        "ambiguous": match.ambiguous,
                    }
                )
            result.registration_rows.append(
                {
                    "treatment": treatment,
                    "region": region,
                    "anchor_day": chosen_anchor_day,
                    "day": recording.day,
                    "recording": recording.recording_name,
                    "shift_y_px": registration.shift_y,
                    "shift_x_px": registration.shift_x,
                    "registration_method": registration.method,
                    "snap_path": snap_details[recording.day]["snap_path"],
                    "snap_shift_y_px": snap_details[recording.day]["snap_shift_y_px"],
                    "snap_shift_x_px": snap_details[recording.day]["snap_shift_x_px"],
                    "snap_path_min_correlation": registration.correlation,
                    "image_correlation": image_correlation_for_shift(
                        anchor_image,
                        moving_image,
                        registration.shift_y,
                        registration.shift_x,
                    ),
                    "n_anchor_rois": len(anchor_stat),
                    "n_day_rois": len(moving_stat),
                    "n_matches": len(matches),
                    "n_ambiguous": sum(match.ambiguous for match in matches),
                    "mean_match_iou": (
                        float(np.mean([match.iou for match in matches]))
                        if matches else float("nan")
                    ),
                }
            )
        return result

    def _render_overlays(
        self,
        *,
        recordings: list[RecordingRef],
        selected_groups: dict[str, set[int]],
        selected_anchor_cells: list[int],
        anchor_group_for_cell: dict[int, str],
        matches_by_day: dict[int, dict[int, CellMatch]],
        moving_masks_by_day: dict[int, list[np.ndarray]],
        aligned_images: dict[int, np.ndarray],
        groups_by_day: dict[int, dict[str, set[int]]],
        image_shape: tuple[int, int],
        paths: dict[str, Path],
    ) -> None:
        """Render registered membership overlays and write TIFF/montage/legend."""
        colors = dict(zip(selected_groups, _palette(len(selected_groups))))
        frames: list[np.ndarray] = []
        for recording in recordings:
            frame = _normalize_background(aligned_images[recording.day])
            matches = matches_by_day[recording.day]
            masks = moving_masks_by_day[recording.day]
            daily_group_sets = list(groups_by_day[recording.day].values())
            for anchor_roi in selected_anchor_cells:
                match = matches.get(anchor_roi)
                if match is None:
                    continue
                roi = match.moving_roi
                color = colors[anchor_group_for_cell[anchor_roi]]
                linear = masks[roi]
                if linear.size == 0:
                    continue
                grouped = any(roi in members for members in daily_group_sets)
                if grouped:
                    yy, xx = np.divmod(linear, image_shape[1])
                    existing = frame[yy, xx].astype(float)
                    frame[yy, xx] = np.clip(
                        0.25 * existing + 0.75 * np.asarray(color), 0, 255
                    ).astype(np.uint8)
                else:
                    yy, xx = _mask_outline(linear, image_shape)
                    frame[yy, xx] = np.asarray(color, dtype=np.uint8)
            frames.append(frame)
        tifffile.imwrite(
            paths["overlay_tiff"],
            np.stack(frames),
            photometric="rgb",
            metadata={"axes": "TYXS"},
        )
        _write_montage(
            frames,
            [recording.day for recording in recordings],
            paths["overlay_montage"],
        )
        legend_rows = [
            {
                "anchor_group_id": group_id,
                "red": color[0],
                "green": color[1],
                "blue": color[2],
                "meaning": (
                    "anchor group identity; solid fill = grouped that day; "
                    "outline = detected but not grouped that day"
                ),
            }
            for group_id, color in colors.items()
        ]
        pd.DataFrame(legend_rows).to_csv(paths["overlay_legend"], index=False)

    def run(
        self,
        *,
        treatment: str,
        region: str,
        output_dir: Path,
        anchor_day: int | None = None,
        top_fraction: float = 0.10,
        top_n: int | None = None,
    ) -> dict[str, Path]:
        """Run one treatment/region track and return the output manifest."""
        recordings, skipped_recordings = self._select_recordings(treatment, region)
        anchor, chosen_anchor_day = self._resolve_anchor(recordings, anchor_day)
        anchor_image, anchor_stat = _load_suite2p(anchor)
        image_shape = tuple(int(value) for value in anchor_image.shape)
        registrations, snap_pairwise_rows, snap_details = (
            self._sequential_snap_registrations(
                recordings,
                anchor_day=chosen_anchor_day,
                mask_shape=image_shape,
            )
        )

        groups_by_day = {
            recording.day: _load_groups(recording, self.strategy)
            for recording in recordings
        }
        active_by_day = {
            recording.day: _load_active_neurons(recording)
            for recording in recordings
        }
        anchor_groups = groups_by_day[chosen_anchor_day]
        if not anchor_groups:
            raise ValueError(
                f"No {self.strategy!r} groups found in anchor workbook {anchor.metrics_path}."
            )
        selected_groups = self._select_anchor_groups(
            anchor_groups, top_fraction=top_fraction, top_n=top_n
        )
        selected_anchor_cells = sorted(set().union(*selected_groups.values()))

        out_dir = Path(output_dir) / treatment / region
        out_dir.mkdir(parents=True, exist_ok=True)

        day_matches = self._match_all_days(
            recordings=recordings,
            anchor_image=anchor_image,
            anchor_stat=anchor_stat,
            image_shape=image_shape,
            registrations=registrations,
            snap_details=snap_details,
            treatment=treatment,
            region=region,
            chosen_anchor_day=chosen_anchor_day,
        )
        matches_by_day = day_matches.matches_by_day
        inverse_matches_by_day = day_matches.inverse_matches_by_day
        moving_masks_by_day = day_matches.moving_masks_by_day
        aligned_images = day_matches.aligned_images
        all_match_rows = day_matches.match_rows
        registration_rows = day_matches.registration_rows

        anchor_group_for_cell = {
            cell: group_id
            for group_id, members in selected_groups.items()
            for cell in members
        }
        history_rows: list[dict] = []
        for anchor_roi in selected_anchor_cells:
            anchor_group_id = anchor_group_for_cell[anchor_roi]
            for recording in recordings:
                match = matches_by_day[recording.day].get(anchor_roi)
                day_roi = match.moving_roi if match else None
                daily_groups = sorted(
                    group_id
                    for group_id, members in groups_by_day[recording.day].items()
                    if day_roi is not None and day_roi in members
                )
                history_rows.append(
                    {
                        "treatment": treatment,
                        "region": region,
                        "anchor_day": chosen_anchor_day,
                        "anchor_group_id": anchor_group_id,
                        "anchor_group_size": len(selected_groups[anchor_group_id]),
                        "anchor_roi": anchor_roi,
                        "day": recording.day,
                        "recording": recording.recording_name,
                        "day_roi": day_roi,
                        "detected": match is not None,
                        "active": bool(
                            day_roi is not None and day_roi in active_by_day[recording.day]
                        ),
                        "grouped": bool(daily_groups),
                        "daily_group_ids": ";".join(daily_groups),
                        "match_score": match.score if match else float("nan"),
                        "mask_iou": match.iou if match else float("nan"),
                        "centroid_distance_px": (
                            match.centroid_distance if match else float("nan")
                        ),
                        "ambiguous_match": match.ambiguous if match else False,
                    }
                )
        history = pd.DataFrame(history_rows)
        for column, source_column in (
            ("first_detected_day", "detected"),
            ("first_active_day", "active"),
            ("first_grouped_day", "grouped"),
        ):
            first = (
                history.loc[history[source_column]]
                .groupby("anchor_roi")["day"]
                .min()
            )
            history[column] = history["anchor_roi"].map(first)

        group_day_rows: list[dict] = []
        for anchor_group_id, anchor_members in selected_groups.items():
            for recording in recordings:
                day = recording.day
                mapping = matches_by_day[day]
                inverse = inverse_matches_by_day[day]
                detected_day_rois = {
                    mapping[cell].moving_roi for cell in anchor_members if cell in mapping
                }
                overlaps: list[tuple[str, int, float]] = []
                for daily_group_id, daily_members in groups_by_day[day].items():
                    mapped_anchor_members = {
                        inverse[roi] for roi in daily_members if roi in inverse
                    }
                    intersection = len(anchor_members & mapped_anchor_members)
                    union = len(anchor_members | mapped_anchor_members)
                    jaccard = intersection / union if union else 0.0
                    if intersection:
                        overlaps.append((daily_group_id, intersection, jaccard))
                overlaps.sort(key=lambda item: (-item[1], -item[2], item[0]))
                dominant = overlaps[0] if overlaps else ("", 0, 0.0)
                n_active = sum(
                    roi in active_by_day[day] for roi in detected_day_rois
                )
                n_grouped = sum(
                    any(roi in members for members in groups_by_day[day].values())
                    for roi in detected_day_rois
                )
                group_day_rows.append(
                    {
                        "treatment": treatment,
                        "region": region,
                        "anchor_day": chosen_anchor_day,
                        "anchor_group_id": anchor_group_id,
                        "anchor_group_size": len(anchor_members),
                        "day": day,
                        "recording": recording.recording_name,
                        "n_detected": len(detected_day_rois),
                        "n_active": n_active,
                        "n_grouped_anywhere": n_grouped,
                        "dominant_daily_group_id": dominant[0],
                        "n_anchor_members_in_dominant_group": dominant[1],
                        "anchor_recovery_fraction": (
                            dominant[1] / len(anchor_members) if anchor_members else 0.0
                        ),
                        "detected_together_fraction": (
                            dominant[1] / len(detected_day_rois)
                            if detected_day_rois else float("nan")
                        ),
                        "best_group_jaccard": dominant[2],
                        "overlapping_daily_groups": ";".join(
                            f"{group_id}:{count}" for group_id, count, _ in overlaps
                        ),
                    }
                )
        group_day_summary = pd.DataFrame(group_day_rows)

        anchor_group_rows = [
            {
                "treatment": treatment,
                "region": region,
                "anchor_day": chosen_anchor_day,
                "anchor_group_id": group_id,
                "anchor_group_size": len(members),
                "anchor_neuron_indices": sorted(members),
            }
            for group_id, members in selected_groups.items()
        ]

        prefix = f"{treatment}_{region}_anchor-day-{chosen_anchor_day}_{self.strategy}"
        paths = {
            "registrations": out_dir / f"{prefix}_registrations.csv",
            "skipped_recordings": out_dir / f"{prefix}_skipped_recordings.csv",
            "snap_pairwise_registrations": (
                out_dir / f"{prefix}_snap_pairwise_registrations.csv"
            ),
            "cell_matches": out_dir / f"{prefix}_cell_matches.csv",
            "anchor_groups": out_dir / f"{prefix}_anchor_groups.csv",
            "cell_history": out_dir / f"{prefix}_cell_history.csv",
            "group_day_summary": out_dir / f"{prefix}_group_day_summary.csv",
            "overlay_tiff": out_dir / f"{prefix}_membership_overlays.tif",
            "overlay_montage": out_dir / f"{prefix}_overlay_montage.png",
            "overlay_legend": out_dir / f"{prefix}_overlay_legend.csv",
        }
        pd.DataFrame(registration_rows).to_csv(paths["registrations"], index=False)
        pd.DataFrame(
            skipped_recordings,
            columns=["treatment", "region", "day", "recording", "reason"],
        ).to_csv(paths["skipped_recordings"], index=False)
        pd.DataFrame(snap_pairwise_rows).to_csv(
            paths["snap_pairwise_registrations"], index=False
        )
        pd.DataFrame(all_match_rows).to_csv(paths["cell_matches"], index=False)
        pd.DataFrame(anchor_group_rows).to_csv(paths["anchor_groups"], index=False)
        history.to_csv(paths["cell_history"], index=False)
        group_day_summary.to_csv(paths["group_day_summary"], index=False)

        self._render_overlays(
            recordings=recordings,
            selected_groups=selected_groups,
            selected_anchor_cells=selected_anchor_cells,
            anchor_group_for_cell=anchor_group_for_cell,
            matches_by_day=matches_by_day,
            moving_masks_by_day=moving_masks_by_day,
            aligned_images=aligned_images,
            groups_by_day=groups_by_day,
            image_shape=image_shape,
            paths=paths,
        )
        return paths
