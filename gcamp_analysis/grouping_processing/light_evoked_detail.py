"""Post-hoc detail analysis for light-evoked neuron groups.

Classifies every spike in a light-evoked group as **light-evoked** or
**spontaneous**, computes per-spike kinetics rows, and derives
response-stereotypy metrics (CV) for neurons with ≥ 2 light-evoked
spikes.

This module sits *downstream* of grouping — it depends on the
``GroupingResult`` produced by ``LightEvokedStrategy`` but does not
modify or re-run the detection logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gcamp_analysis.data_classes.neuron import Neuron
    from gcamp_analysis.data_classes.neuron_group import NeuronGroup
    from gcamp_analysis.grouping_processing.service import GroupingResult


# ── Spike classification ─────────────────────────────────────────────


def _classify_spikes_for_neuron(
    neuron: "Neuron",
    schedule: List[int],
    fs: float,
    response_window_frames: int = 10,
) -> List[Dict[str, Any]]:
    """Classify each spike of *neuron* as light-evoked or spontaneous.

    For every pulse frame in *schedule*, the closest spike whose
    ``sm_f_idx`` falls in ``[pulse_frame, pulse_frame + response_window_frames]``
    is labelled **light-evoked** (greedy, one-to-one matching).  All
    remaining spikes are **spontaneous**.

    Parameters
    ----------
    neuron : Neuron
    schedule : list[int]
        Pulse onset frames.
    fs : float
        Sampling rate (Hz) — used for latency conversion.
    response_window_frames : int
        Maximum number of frames *after* the pulse onset in which a
        spike can be considered light-evoked.

    Returns
    -------
    list[dict]
        One dict per spike with classification and kinetics columns.
    """
    matched_pulse: Dict[int, int] = {}  # spike_idx -> pulse_frame
    used_spikes: set[int] = set()

    # Greedy match: for each pulse, pick the closest eligible spike
    for pulse_frame in schedule:
        best_idx: int | None = None
        best_dist = float("inf")
        for si, spike in enumerate(neuron.spikes):
            if si in used_spikes:
                continue
            dist = spike.sm_f_idx - pulse_frame
            if 0 <= dist <= response_window_frames and dist < best_dist:
                best_dist = dist
                best_idx = si
        if best_idx is not None:
            matched_pulse[best_idx] = pulse_frame
            used_spikes.add(best_idx)

    rows: List[Dict[str, Any]] = []
    neuron_idx = getattr(neuron, "index", getattr(neuron, "filtered_index", -1))
    for si, spike in enumerate(neuron.spikes):
        is_evoked = si in matched_pulse
        pulse_frame = matched_pulse.get(si)
        latency_ms = (
            (spike.sm_f_idx - pulse_frame) / fs * 1000.0
            if pulse_frame is not None
            else None
        )

        rows.append(
            {
                "neuron_idx": neuron_idx,
                "spike_idx": si,
                "peak_frame": spike.sm_f_idx,
                "classification": "light-evoked" if is_evoked else "spontaneous",
                "pulse_frame": pulse_frame,
                "amplitude": spike.prominence,
                "f_value": spike.f_value,
                "rise_slope_hz": spike.stats.get("rise_slope_hz"),
                "decay_tau_seconds": spike.stats.get("decay_tau_seconds"),
                "half_max_width_seconds": spike.stats.get("half_max_width_seconds"),
                "latency_to_pulse_ms": latency_ms,
            }
        )
    return rows


# ── Response stereotypy (CV) ─────────────────────────────────────────


_STEREOTYPY_COLS = [
    "amplitude",
    "rise_slope_hz",
    "decay_tau_seconds",
    "half_max_width_seconds",
]


def _compute_stereotypy(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Per-neuron coefficient of variation across light-evoked spikes.

    Only neurons with ≥ 2 light-evoked spikes produce a row.

    Returns
    -------
    pd.DataFrame
        Columns: ``neuron_idx``, ``n_light_evoked_spikes``, and for
        each kinetic stat: ``{stat}_mean``, ``{stat}_std``, ``{stat}_cv``.
    """
    evoked = detail_df.loc[detail_df["classification"] == "light-evoked"]
    if evoked.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for neuron_idx, grp in evoked.groupby("neuron_idx"):
        if len(grp) < 2:
            continue
        row: Dict[str, Any] = {
            "neuron_idx": neuron_idx,
            "n_light_evoked_spikes": len(grp),
        }
        for col in _STEREOTYPY_COLS:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(vals) >= 2:
                mu = float(vals.mean())
                sd = float(vals.std(ddof=1))
                row[f"{col}_mean"] = mu
                row[f"{col}_std"] = sd
                row[f"{col}_cv"] = sd / mu if mu != 0 else float("nan")
            else:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
                row[f"{col}_cv"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


# ── Public entry point ────────────────────────────────────────────────


def build_light_evoked_detail(
    result: "GroupingResult",
    fs: float,
    response_window_frames: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Build per-spike detail tables for every group in a light-evoked result.

    Parameters
    ----------
    result : GroupingResult
        The result from ``LightEvokedStrategy``.  Must contain
        ``metadata["schedule"]``.
    fs : float
        Sampling rate used for latency conversion.
    response_window_frames : int
        Frames after pulse onset within which a spike counts as
        light-evoked.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by ``group_id`` (e.g. ``"ON_2_response(s)"``).  Each
        DataFrame has one row per spike per neuron, plus appended
        stereotypy rows (marked with ``_stereotypy`` suffix key).
    """
    schedule: List[int] = result.metadata.get("schedule", [])
    if not schedule:
        return {}

    detail_tables: Dict[str, pd.DataFrame] = {}
    for group in result.groups:
        spike_rows: List[Dict[str, Any]] = []
        for neuron in group.neurons:
            spike_rows.extend(
                _classify_spikes_for_neuron(
                    neuron,
                    schedule=schedule,
                    fs=fs,
                    response_window_frames=response_window_frames,
                )
            )

        if not spike_rows:
            continue

        detail_df = pd.DataFrame(spike_rows)
        detail_tables[str(group.group_id)] = detail_df

        # Stereotypy analysis (separate key so it can be written as its
        # own sheet or appended as a second block).
        stereo_df = _compute_stereotypy(detail_df)
        if not stereo_df.empty:
            detail_tables[f"{group.group_id}_stereotypy"] = stereo_df

    return detail_tables
