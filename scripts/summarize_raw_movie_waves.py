"""Consolidate movie-first wave results across days and treatments."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = []
    for directory in args.directories:
        files.extend(directory.glob("*/movie_wave_candidates.csv"))
    combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    combined["day"] = (
        combined["recording"].str.extract(r"Day(\d+)")[0].astype(int)
    )
    summary = (
        combined.groupby(["day", "treatment", "recording"], as_index=False)
        .agg(
            candidate_peaks=("center_frame", "size"),
            movie_positive_peaks=("movie_significant_wave", "sum"),
            maximum_movie_r2=("movie_propagation_r2", "max"),
        )
        .sort_values(["day", "treatment", "recording"])
    )
    positives = combined[combined["movie_significant_wave"]].copy()
    confirmation_files = [
        path
        for directory in args.directories
        for path in directory.glob("*/movie_neighbor_confirmation.csv")
    ]
    if confirmation_files:
        confirmations = pd.concat(
            [pd.read_csv(path) for path in confirmation_files],
            ignore_index=True,
        )
        confirmation_columns = [
            "treatment",
            "recording",
            "center_frame",
            "xcorr_model",
            "xcorr_propagation_r2",
            "xcorr_propagation_p",
            "xcorr_propagation_q",
            "xcorr_significant_wave",
            "movie_xcorr_same_model",
            "movie_xcorr_angle_difference_degrees",
        ]
        positives = positives.merge(
            confirmations[confirmation_columns],
            on=["treatment", "recording", "center_frame"],
            how="left",
        )
    positives = positives.sort_values(
        ["day", "treatment", "recording", "center_frame"]
    )
    positives["overlapping_window_cluster"] = -1
    cluster = 0
    for _, group in positives.groupby(["treatment", "recording"]):
        previous = None
        for index in group.index:
            center = int(positives.loc[index, "center_frame"])
            if previous is None or center - previous > 24:
                cluster += 1
            positives.loc[index, "overlapping_window_cluster"] = cluster
            previous = center
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "recording_summary.csv", index=False)
    positives.to_csv(args.output / "positive_movie_peaks.csv", index=False)
    combined.to_csv(args.output / "all_movie_candidates.csv", index=False)
    print(summary.to_string(index=False))
    print(
        f"\n{len(combined)} candidates in {len(summary)} recordings; "
        f"{len(positives)} positive peaks in "
        f"{positives['overlapping_window_cluster'].nunique()} "
        "overlapping-window clusters."
    )


if __name__ == "__main__":
    main()
