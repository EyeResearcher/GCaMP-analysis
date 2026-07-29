"""Match WaveMiner-compatible components across activity thresholds.

Threshold persistence is especially important here because WaveMiner was
developed for ex-vivo retina, whereas this data set is an in-vitro calcium
movie.  This script treats 3-D flood-fill components as proposals and asks
whether the same x/y/t object survives increasingly stringent thresholds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _best_parent(
    child_labels: np.ndarray,
    child_id: int,
    parent_labels: np.ndarray,
) -> tuple[int, float]:
    """Return the parent label containing the largest fraction of child voxels."""
    parent_ids = parent_labels[child_labels == child_id]
    parent_ids = parent_ids[parent_ids > 0]
    child_voxels = int(np.count_nonzero(child_labels == child_id))
    if parent_ids.size == 0 or child_voxels == 0:
        return 0, 0.0
    counts = np.bincount(parent_ids)
    parent_id = int(np.argmax(counts))
    return parent_id, float(counts[parent_id] / child_voxels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--mode", default="level")
    parser.add_argument("--thresholds", type=float, nargs=3, default=[2.0, 2.5, 3.0])
    parser.add_argument("--minimum-overlap", type=float, default=0.50)
    parser.add_argument("--minimum-duration-frames", type=int, default=8)
    parser.add_argument("--minimum-spatial-blocks", type=int, default=64)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds)
    tables = {
        z: pd.read_csv(
            args.results / f"components_{args.mode}_z{z:.1f}.csv"
        ).set_index("component_id")
        for z in thresholds
    }
    labels = {
        z: np.load(
            args.results / f"labels_{args.mode}_z{z:.1f}.npy",
            mmap_mode="r",
        )
        for z in thresholds
    }
    low, middle, high = thresholds
    rows: list[dict] = []
    for high_id, high_row in tables[high].iterrows():
        if (
            high_row["duration_frames"] < args.minimum_duration_frames
            or high_row["spatial_block_count"] < args.minimum_spatial_blocks
            or not bool(high_row["waveminer_propagating_component"])
        ):
            continue
        middle_id, middle_overlap = _best_parent(
            labels[high], int(high_id), labels[middle]
        )
        low_id, low_overlap = _best_parent(
            labels[high], int(high_id), labels[low]
        )
        if (
            middle_id == 0
            or low_id == 0
            or middle_overlap < args.minimum_overlap
            or low_overlap < args.minimum_overlap
        ):
            continue
        middle_row = tables[middle].loc[middle_id]
        low_row = tables[low].loc[low_id]
        rows.append(
            {
                "event_id": len(rows) + 1,
                f"component_z{low:.1f}": low_id,
                f"component_z{middle:.1f}": middle_id,
                f"component_z{high:.1f}": int(high_id),
                f"high_voxel_overlap_z{low:.1f}": low_overlap,
                f"high_voxel_overlap_z{middle:.1f}": middle_overlap,
                "consensus_start_frame": int(
                    np.median(
                        [
                            low_row["start_frame"],
                            middle_row["start_frame"],
                            high_row["start_frame"],
                        ]
                    )
                ),
                "consensus_end_frame": int(
                    np.median(
                        [
                            low_row["end_frame"],
                            middle_row["end_frame"],
                            high_row["end_frame"],
                        ]
                    )
                ),
                "start_frame_range": (
                    f"{int(min(low_row['start_frame'], middle_row['start_frame'], high_row['start_frame']))}"
                    f"-{int(max(low_row['start_frame'], middle_row['start_frame'], high_row['start_frame']))}"
                ),
                "end_frame_range": (
                    f"{int(min(low_row['end_frame'], middle_row['end_frame'], high_row['end_frame']))}"
                    f"-{int(max(low_row['end_frame'], middle_row['end_frame'], high_row['end_frame']))}"
                ),
                "area_mm2_low": low_row["area_um2"] / 1e6,
                "area_mm2_middle": middle_row["area_um2"] / 1e6,
                "area_mm2_high": high_row["area_um2"] / 1e6,
                "duration_s_low": low_row["duration_seconds"],
                "duration_s_middle": middle_row["duration_seconds"],
                "duration_s_high": high_row["duration_seconds"],
                "leading_speed_um_s_low": low_row["leading_edge_speed_um_s"],
                "leading_speed_um_s_middle": middle_row["leading_edge_speed_um_s"],
                "leading_speed_um_s_high": high_row["leading_edge_speed_um_s"],
                "single_front_fraction_middle": middle_row[
                    "single_front_fraction"
                ],
                "arrival_model_middle": middle_row.get("arrival_model", np.nan),
                "arrival_r2_middle": middle_row.get(
                    "arrival_propagation_r2", np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        # A stringent threshold can split one lower-threshold event into
        # several islands.  Treat islands with the same low/middle parent as
        # one threshold-stable event and retain the largest high-threshold
        # island as its representative.
        high_size_column = "area_mm2_high"
        result["high_component_fragments"] = result.groupby(
            [f"component_z{low:.1f}", f"component_z{middle:.1f}"]
        )[f"component_z{high:.1f}"].transform(
            lambda values: ",".join(str(int(value)) for value in values)
        )
        result = (
            result.sort_values(high_size_column, ascending=False)
            .drop_duplicates(
                [f"component_z{low:.1f}", f"component_z{middle:.1f}"]
            )
        )
        result = result.sort_values(
            ["consensus_start_frame", "consensus_end_frame"]
        ).reset_index(drop=True)
        result["event_id"] = np.arange(1, len(result) + 1)
    output = args.results / f"threshold_stable_{args.mode}_components.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved {len(result)} threshold-stable components to {output}")


if __name__ == "__main__":
    main()
