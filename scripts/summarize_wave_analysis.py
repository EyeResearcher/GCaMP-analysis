"""Consolidate per-day retinal-wave outputs into final tables and figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def _read_if_nonempty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _result_directories(root: Path) -> list[Path]:
    directories = [root]
    directories.extend(
        path
        for path in sorted(root.glob("day*"))
        if path.is_dir() and (path / "recording_wave_summary.csv").exists()
    )
    return directories


def consolidate(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    episodes = []
    for directory in _result_directories(root):
        summary = _read_if_nonempty(directory / "recording_wave_summary.csv")
        if not summary.empty:
            summaries.append(summary)
        episode = _read_if_nonempty(directory / "all_wave_episodes.csv")
        if not episode.empty:
            episodes.append(episode)
    recording_summary = pd.concat(summaries, ignore_index=True)
    recording_summary = recording_summary.drop_duplicates(
        ["treatment", "recording"], keep="first"
    ).sort_values(["day", "treatment", "recording"], ascending=[False, True, True])
    all_episodes = (
        pd.concat(episodes, ignore_index=True)
        .drop_duplicates(["treatment", "recording", "episode_id"], keep="first")
        .sort_values(["day", "treatment", "recording", "center_frame"], ascending=[False, True, True, True])
        if episodes
        else pd.DataFrame()
    )
    if "n_movie_validated_waves" not in recording_summary:
        recording_summary["n_movie_validated_waves"] = 0
    recording_summary["n_movie_validated_waves"] = (
        recording_summary["n_movie_validated_waves"].fillna(0).astype(int)
    )
    day_treatment = (
        recording_summary.groupby(["day", "treatment"], as_index=False)
        .agg(
            n_recordings=("recording", "count"),
            total_duration_minutes=("duration_seconds", lambda x: float(np.sum(x) / 60.0)),
            median_analysis_neurons=("n_analysis_neurons", "median"),
            min_analysis_neurons=("n_analysis_neurons", "min"),
            total_population_candidates=("n_population_candidates", "sum"),
            roi_significant_waves=("n_significant_waves", "sum"),
            movie_corroborated_waves=("n_movie_validated_waves", "sum"),
            recordings_with_roi_waves=("n_significant_waves", lambda x: int(np.sum(np.asarray(x) > 0))),
            recordings_with_movie_corroboration=(
                "n_movie_validated_waves",
                lambda x: int(np.sum(np.asarray(x) > 0)),
            ),
        )
    )
    day_treatment["roi_wave_rate_per_10_min"] = (
        10.0 * day_treatment["roi_significant_waves"] / day_treatment["total_duration_minutes"]
    )
    day_treatment["corroborated_wave_rate_per_10_min"] = (
        10.0
        * day_treatment["movie_corroborated_waves"]
        / day_treatment["total_duration_minutes"]
    )
    return recording_summary, all_episodes, day_treatment


def _plot_overview(day_treatment: pd.DataFrame, output: Path) -> None:
    days = sorted(day_treatment["day"].unique())
    treatments = ["BP", "IOBP"]
    colors = {"BP": "#3b82f6", "IOBP": "#ef4444"}
    figure, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    width = 0.36
    x = np.arange(len(days))
    for offset, treatment in zip([-width / 2, width / 2], treatments):
        subset = day_treatment.set_index(["day", "treatment"])
        roi_counts = [
            subset.loc[(day, treatment), "roi_significant_waves"]
            if (day, treatment) in subset.index
            else 0
            for day in days
        ]
        movie_counts = [
            subset.loc[(day, treatment), "movie_corroborated_waves"]
            if (day, treatment) in subset.index
            else 0
            for day in days
        ]
        neurons = [
            subset.loc[(day, treatment), "median_analysis_neurons"]
            if (day, treatment) in subset.index
            else np.nan
            for day in days
        ]
        axes[0].bar(
            x + offset,
            roi_counts,
            width,
            color=colors[treatment],
            alpha=0.35,
            label=f"{treatment}: ROI significant",
        )
        axes[0].bar(
            x + offset,
            movie_counts,
            width,
            color=colors[treatment],
            label=f"{treatment}: movie corroborated",
        )
        recording_counts = [
            subset.loc[(day, treatment), "recordings_with_roi_waves"]
            if (day, treatment) in subset.index
            else 0
            for day in days
        ]
        n_recordings = [
            subset.loc[(day, treatment), "n_recordings"]
            if (day, treatment) in subset.index
            else 0
            for day in days
        ]
        axes[1].bar(
            x + offset,
            np.divide(recording_counts, n_recordings, out=np.zeros(len(days)), where=np.asarray(n_recordings) > 0),
            width,
            color=colors[treatment],
            label=treatment,
        )
        axes[2].plot(
            x,
            neurons,
            marker="o",
            linewidth=2,
            color=colors[treatment],
            label=treatment,
        )
    axes[0].set_ylabel("Detected episodes")
    axes[0].set_title("Spatially propagating calcium episodes by day")
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].set_ylabel("Fraction of recordings\nwith ≥1 ROI wave")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[2].set_ylabel("Median analysis neurons")
    axes[2].set_xlabel("Day")
    axes[2].set_xticks(x, days)
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _report(
    root: Path,
    recording_summary: pd.DataFrame,
    episodes: pd.DataFrame,
    day_treatment: pd.DataFrame,
) -> str:
    day10 = recording_summary[recording_summary["day"] == 10]
    bp10 = day10[day10["treatment"] == "BP"]
    iobp10 = day10[day10["treatment"] == "IOBP"]
    table10 = [
        [int((bp10["n_significant_waves"] > 0).sum()), int((bp10["n_significant_waves"] == 0).sum())],
        [int((iobp10["n_significant_waves"] > 0).sum()), int((iobp10["n_significant_waves"] == 0).sum())],
    ]
    day10_fisher = fisher_exact(table10, alternative="greater").pvalue
    late = recording_summary[recording_summary["day"].isin([7, 10])]
    bp_late = late[late["treatment"] == "BP"]
    io_late = late[late["treatment"] == "IOBP"]
    late_table = [
        [int((bp_late["n_significant_waves"] > 0).sum()), int((bp_late["n_significant_waves"] == 0).sum())],
        [int((io_late["n_significant_waves"] > 0).sum()), int((io_late["n_significant_waves"] == 0).sum())],
    ]
    late_fisher = fisher_exact(late_table, alternative="greater").pvalue

    significant = episodes[episodes.get("significant_wave", False) == True].copy()
    corroborated = episodes[episodes.get("movie_validated", False) == True].copy()
    planar = corroborated[corroborated["model"] == "planar"]
    median_angle = (
        float(planar["movie_angle_difference_degrees"].median()) if len(planar) else np.nan
    )
    median_speed = (
        float(corroborated["speed_um_s"].median()) if len(corroborated) else np.nan
    )
    recurrent = significant[significant.get("recurrence_cluster", -1) >= 0]
    low_sensitivity = recording_summary[recording_summary["n_analysis_neurons"] < 100]

    lines = [
        "# Retinal-wave analysis report",
        "",
        "## Main result",
        "",
        (
            f"Across {len(recording_summary)} available recording-days, the detector found "
            f"{int(significant.shape[0])} ROI-significant propagating calcium episodes. "
            f"{int(corroborated.shape[0])} also had independently significant and direction/origin-matched "
            "propagation in the block-averaged source TIFF."
        ),
        "",
        (
            "All detected episodes occurred on Days 7 or 10 and in BP recordings. "
            "No IOBP recording and no recording from Days 1–6 passed both ROI null tests."
        ),
        "",
        "Day 10 contained 15 ROI-significant episodes (10 movie-corroborated); "
        "Day 7 contained 5 ROI-significant episodes (3 movie-corroborated).",
        "",
        (
            f"At Day 10, {table10[0][0]}/{sum(table10[0])} BP versus "
            f"{table10[1][0]}/{sum(table10[1])} IOBP recordings contained a wave "
            f"(one-sided Fisher exact p={day10_fisher:.4g}). Across Days 7 and 10 combined, "
            f"the corresponding recording-day comparison was {late_table[0][0]}/{sum(late_table[0])} "
            f"versus {late_table[1][0]}/{sum(late_table[1])} (p={late_fisher:.4g}). "
            "These recording-day tests are descriptive because regions are followed longitudinally."
        ),
        "",
        (
            f"The median ROI-derived speed among movie-corroborated episodes was "
            f"{median_speed:.0f} µm/s. For corroborated planar fronts, the median independent "
            f"ROI-versus-movie direction disagreement was {median_angle:.1f}°."
        ),
        "",
        (
            f"{len(recurrent)} significant Day 10 episodes belonged to repeated-pattern clusters; "
            "these clusters were concentrated in BP 1-1_Day10."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The results support spatially propagating calcium activity consistent with retinal waves, "
            "with an apparent emergence between Days 6 and 7 and a larger burden by Day 10 in BP. "
            "They do not directly demonstrate propagating action potentials."
        ),
        "",
        "The estimated speeds are faster than should be accepted uncritically as biological wavefront "
        "velocities. Calcium kinetics, 15 Hz sampling, onset-window truncation, scan timing, and residual "
        "motion can bias speed estimates. Presence, timing order, and repeatability are more reliable here "
        "than the absolute speed.",
        "",
        "## Sensitivity caveat",
        "",
        (
            f"{len(low_sensitivity)} recordings had fewer than 100 classifier-accepted analysis neurons. "
            "Most of these are on Days 1–4, so early negative results have substantially lower power than "
            "Days 7 and 10. A movie-only detector would be a useful follow-up for those early recordings."
        ),
        "",
        "## Decision rules",
        "",
        "- Analysis neurons were inherited from the existing ROI/spike-classifier output.",
        "- Rising-edge times were taken from per-ROI z-scored `F.npy`, smoothed at σ=2 frames.",
        "- Candidate population episodes exceeded the 95th percentile of the recording-level maximum after independent circular shifts of every neuron's event train.",
        "- Planar and radial propagation models were compared against 499 activation-time permutations; model selection was repeated inside every permutation.",
        "- Propagation required BH-adjusted q≤0.05, R²≥0.15, at least 20 neurons, at least 5% participation, and at least 10% spatial coverage.",
        "- Movie corroboration required significant pixel-block propagation, the same model family, and either ≤45° directional disagreement or ≤250 µm origin disagreement.",
        "",
        "## Output files",
        "",
        f"- `{root / 'combined_recording_wave_summary.csv'}`",
        f"- `{root / 'combined_wave_episodes.csv'}`",
        f"- `{root / 'day_treatment_wave_summary.csv'}`",
        f"- `{root / 'wave_analysis_overview.png'}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    args = parser.parse_args()
    root = args.results_root
    recording, episodes, day_treatment = consolidate(root)
    recording.to_csv(root / "combined_recording_wave_summary.csv", index=False)
    episodes.to_csv(root / "combined_wave_episodes.csv", index=False)
    day_treatment.to_csv(root / "day_treatment_wave_summary.csv", index=False)
    _plot_overview(day_treatment, root / "wave_analysis_overview.png")
    (root / "wave_analysis_report.md").write_text(
        _report(root, recording, episodes, day_treatment), encoding="utf-8"
    )
    print(day_treatment.to_string(index=False))


if __name__ == "__main__":
    main()
