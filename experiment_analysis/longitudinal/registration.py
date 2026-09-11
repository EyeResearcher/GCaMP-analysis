"""Rigid image registration and one-to-one Suite2p ROI-mask matching."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from .models import CellMatch, RegistrationResult


def _standardize_image(image: np.ndarray) -> np.ndarray:
    """Return a finite, robustly scaled image suitable for phase correlation."""
    values = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Registration image contains no finite pixels.")
    fill = float(np.nanmedian(values[finite]))
    values = np.where(finite, values, fill)
    lo, hi = np.percentile(values, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(values)
    values = np.clip(values, lo, hi)
    values = (values - values.mean()) / max(float(values.std()), 1e-6)
    # A cosine taper prevents hard image borders from dominating the FFT.
    wy = np.hanning(values.shape[0])
    wx = np.hanning(values.shape[1])
    return values * wy[:, None] * wx[None, :]


def _prepare_snap(image: np.ndarray, highpass_sigma: float = 30.0) -> np.ndarray:
    """Log-transform and high-pass a snap before phase correlation."""
    values = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Snap contains no finite pixels.")
    fill = float(np.nanmedian(values[finite]))
    values = np.where(finite, values, fill)
    lo, hi = np.percentile(values, [1.0, 99.8])
    values = np.log1p(np.clip(values - lo, 0.0, max(float(hi - lo), 1.0)))
    values = values - gaussian_filter(values, sigma=float(highpass_sigma))
    values = values - values.mean()
    values = values / max(float(values.std()), 1e-6)
    wy = np.hanning(values.shape[0])
    wx = np.hanning(values.shape[1])
    return values * wy[:, None] * wx[None, :]


def _estimate_prepared_translation(
    anchor: np.ndarray,
    moving: np.ndarray,
    *,
    max_shift: int,
    method: str,
) -> RegistrationResult:
    """Estimate a bounded integer phase-correlation translation."""
    if anchor.shape != moving.shape:
        raise ValueError(
            f"Registration images must have equal shapes, got {anchor.shape} "
            f"and {moving.shape}."
        )
    cross_power = np.fft.fft2(anchor) * np.conj(np.fft.fft2(moving))
    magnitude = np.abs(cross_power)
    cross_power = np.divide(
        cross_power,
        magnitude,
        out=np.zeros_like(cross_power),
        where=magnitude > 1e-12,
    )
    surface = np.abs(np.fft.ifft2(cross_power))
    height, width = anchor.shape
    ys = np.arange(-max_shift, max_shift + 1)
    xs = np.arange(-max_shift, max_shift + 1)
    window = surface[np.ix_(ys % height, xs % width)]
    peak_y, peak_x = np.unravel_index(int(np.argmax(window)), window.shape)
    shift_y = int(ys[peak_y])
    shift_x = int(xs[peak_x])
    aligned = shift_image(moving, shift_y, shift_x)
    y0 = max(0, shift_y)
    y1 = min(height, height + shift_y)
    x0 = max(0, shift_x)
    x1 = min(width, width + shift_x)
    a = anchor[y0:y1, x0:x1].ravel()
    b = aligned[y0:y1, x0:x1].ravel()
    if a.size < 2 or a.std() <= 1e-8 or b.std() <= 1e-8:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return RegistrationResult(shift_y, shift_x, corr, method)


def shift_image(image: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    """Translate an image into anchor coordinates without wraparound."""
    return ndi_shift(
        np.asarray(image),
        shift=(int(shift_y), int(shift_x)),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def estimate_translation(
    anchor_image: np.ndarray,
    moving_image: np.ndarray,
    *,
    max_shift: int = 80,
) -> RegistrationResult:
    """Estimate the integer translation that aligns *moving_image* to anchor.

    Phase correlation supplies the shift. The returned correlation is an
    ordinary Pearson correlation between the standardized anchor and the
    translated moving image, restricted to their non-padded overlap.
    """
    anchor = _standardize_image(anchor_image)
    moving = _standardize_image(moving_image)
    return _estimate_prepared_translation(
        anchor,
        moving,
        max_shift=max_shift,
        method="mean_image_phase_correlation",
    )


def estimate_snap_translation(
    anchor_snap: np.ndarray,
    moving_snap: np.ndarray,
    *,
    max_shift: int = 160,
    highpass_sigma: float = 30.0,
) -> RegistrationResult:
    """Estimate the translation mapping a moving snap to an anchor snap."""
    return _estimate_prepared_translation(
        _prepare_snap(anchor_snap, highpass_sigma=highpass_sigma),
        _prepare_snap(moving_snap, highpass_sigma=highpass_sigma),
        max_shift=max_shift,
        method="snap_phase_correlation_log_highpass",
    )


def image_correlation_for_shift(
    anchor_image: np.ndarray,
    moving_image: np.ndarray,
    shift_y: int,
    shift_x: int,
) -> float:
    """Compute registered mean-image correlation over the valid overlap."""
    anchor = _standardize_image(anchor_image)
    moving = _standardize_image(moving_image)
    aligned = shift_image(moving, shift_y, shift_x)
    height, width = anchor.shape
    y0 = max(0, int(shift_y))
    y1 = min(height, height + int(shift_y))
    x0 = max(0, int(shift_x))
    x1 = min(width, width + int(shift_x))
    a = anchor[y0:y1, x0:x1].ravel()
    b = aligned[y0:y1, x0:x1].ravel()
    if a.size < 2 or a.std() <= 1e-8 or b.std() <= 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _stat_union_image(stat: Iterable, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    image = np.zeros((height, width), dtype=np.float32)
    for entry in stat:
        ypix = np.asarray(entry.get("ypix", []), dtype=int)
        xpix = np.asarray(entry.get("xpix", []), dtype=int)
        valid = (
            (ypix >= 0)
            & (ypix < height)
            & (xpix >= 0)
            & (xpix < width)
        )
        image[ypix[valid], xpix[valid]] = 1.0
    return image


def estimate_mask_translation(
    anchor_stat: Iterable,
    moving_stat: Iterable,
    image_shape: tuple[int, int],
    *,
    anchor_image: np.ndarray | None = None,
    moving_image: np.ndarray | None = None,
    max_shift: int = 80,
) -> RegistrationResult:
    """Align aggregate Suite2p mask fields by maximum pixel overlap.

    Unlike intensity phase correlation, this objective directly optimizes the
    spatial objects later used for cell identity matching. ``correlation`` is
    still the registered mean-image correlation when images are provided.
    """
    anchor_union = _stat_union_image(anchor_stat, image_shape)
    moving_union = _stat_union_image(moving_stat, image_shape)
    overlap = fftconvolve(anchor_union, moving_union[::-1, ::-1], mode="same")
    center_y, center_x = np.asarray(image_shape, dtype=int) // 2
    y0 = max(0, center_y - int(max_shift))
    y1 = min(image_shape[0], center_y + int(max_shift) + 1)
    x0 = max(0, center_x - int(max_shift))
    x1 = min(image_shape[1], center_x + int(max_shift) + 1)
    window = overlap[y0:y1, x0:x1]
    peak_y, peak_x = np.unravel_index(int(np.argmax(window)), window.shape)
    shift_y = int((y0 + peak_y) - center_y)
    shift_x = int((x0 + peak_x) - center_x)
    if anchor_image is not None and moving_image is not None:
        correlation = image_correlation_for_shift(
            anchor_image, moving_image, shift_y, shift_x
        )
    else:
        correlation = float("nan")
    return RegistrationResult(
        shift_y=shift_y,
        shift_x=shift_x,
        correlation=correlation,
        method="suite2p_mask_overlap",
    )


def stat_to_masks(
    stat: Iterable,
    image_shape: tuple[int, int],
    *,
    shift_y: int = 0,
    shift_x: int = 0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Convert Suite2p stat entries to sorted linear mask indices and centroids."""
    height, width = image_shape
    masks: list[np.ndarray] = []
    centroids: list[tuple[float, float]] = []
    for entry in stat:
        ypix = np.asarray(entry.get("ypix", []), dtype=int) + int(shift_y)
        xpix = np.asarray(entry.get("xpix", []), dtype=int) + int(shift_x)
        valid = (
            (ypix >= 0)
            & (ypix < height)
            & (xpix >= 0)
            & (xpix < width)
        )
        ypix = ypix[valid]
        xpix = xpix[valid]
        if ypix.size:
            linear = np.unique(ypix * width + xpix)
            centroids.append((float(ypix.mean()), float(xpix.mean())))
        else:
            linear = np.asarray([], dtype=np.int64)
            centroids.append((float("nan"), float("nan")))
        masks.append(linear.astype(np.int64, copy=False))
    return masks, np.asarray(centroids, dtype=float)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    intersection = int(np.intersect1d(left, right, assume_unique=True).size)
    union = int(left.size + right.size - intersection)
    return float(intersection / union) if union else 0.0


def match_rois_to_anchor(
    anchor_stat: Iterable,
    moving_stat: Iterable,
    image_shape: tuple[int, int],
    registration: RegistrationResult,
    *,
    max_centroid_distance: float = 10.0,
    min_iou: float = 0.05,
    min_score: float = 0.24,
    ambiguity_margin: float = 0.08,
) -> tuple[list[CellMatch], list[np.ndarray]]:
    """Match moving-day masks to anchor masks with global one-to-one assignment.

    Candidate score is 75% mask IoU and 25% Gaussian centroid proximity.
    Matches must have some mask overlap and pass *min_score*. ``ambiguous`` is
    true when the assigned score is within *ambiguity_margin* of another
    candidate for the same anchor cell.
    """
    anchor_masks, anchor_centroids = stat_to_masks(anchor_stat, image_shape)
    moving_masks, moving_centroids = stat_to_masks(
        moving_stat,
        image_shape,
        shift_y=registration.shift_y,
        shift_x=registration.shift_x,
    )
    n_anchor = len(anchor_masks)
    n_moving = len(moving_masks)
    if n_anchor == 0 or n_moving == 0:
        return [], moving_masks

    valid_moving = np.isfinite(moving_centroids).all(axis=1)
    valid_anchor = np.isfinite(anchor_centroids).all(axis=1)
    tree = cKDTree(moving_centroids[valid_moving])
    moving_lookup = np.flatnonzero(valid_moving)
    cost = np.full((n_anchor, n_moving), 1e3, dtype=np.float32)
    candidate_scores: dict[int, list[float]] = {}
    candidate_ious: dict[tuple[int, int], float] = {}
    candidate_distances: dict[tuple[int, int], float] = {}

    sigma = max(float(max_centroid_distance) / 2.0, 1e-6)
    for anchor_idx in np.flatnonzero(valid_anchor):
        local_candidates = tree.query_ball_point(
            anchor_centroids[anchor_idx],
            r=float(max_centroid_distance),
        )
        scores: list[float] = []
        for local_idx in local_candidates:
            moving_idx = int(moving_lookup[local_idx])
            distance = float(
                np.linalg.norm(
                    anchor_centroids[anchor_idx] - moving_centroids[moving_idx]
                )
            )
            iou = _mask_iou(anchor_masks[anchor_idx], moving_masks[moving_idx])
            if iou < min_iou:
                continue
            proximity = float(np.exp(-0.5 * (distance / sigma) ** 2))
            score = 0.75 * iou + 0.25 * proximity
            cost[anchor_idx, moving_idx] = 1.0 - score
            scores.append(score)
            candidate_ious[(anchor_idx, moving_idx)] = iou
            candidate_distances[(anchor_idx, moving_idx)] = distance
        candidate_scores[anchor_idx] = sorted(scores, reverse=True)

    anchor_rows, moving_cols = linear_sum_assignment(cost)
    matches: list[CellMatch] = []
    for anchor_idx, moving_idx in zip(anchor_rows.tolist(), moving_cols.tolist()):
        if cost[anchor_idx, moving_idx] >= 1e2:
            continue
        score = float(1.0 - cost[anchor_idx, moving_idx])
        if score < min_score:
            continue
        ranked = candidate_scores.get(anchor_idx, [])
        second = ranked[1] if len(ranked) > 1 else float("-inf")
        matches.append(
            CellMatch(
                anchor_roi=int(anchor_idx),
                moving_roi=int(moving_idx),
                score=score,
                iou=float(candidate_ious[(anchor_idx, moving_idx)]),
                centroid_distance=float(
                    candidate_distances[(anchor_idx, moving_idx)]
                ),
                ambiguous=bool(score - second < ambiguity_margin),
            )
        )
    return matches, moving_masks
