import argparse
import numpy as np
from typing import List, Tuple
from data_structures import Peak, Valley
from preprocess import find_peaks_and_valleys, assign_normalized_values, couple_peaks_to_valleys
from depth_ranking import compute_valley_depths, assign_valley_ranks, sort_valleys_by_metrics, sort_and_rank_valleys
from rank_scoring import compute_all_peak_rank_scores
from visualize import plot_all_styles
def main(cascade_trace = None, rawf = None):
    parser = argparse.ArgumentParser(description="Spike ranking tool")
    parser.add_argument("--raw_f", default = rawf, help="Raw fluorescence file")
    parser.add_argument("--cascade", default = cascade_trace, help="Cascade file")
    parser.add_argument("--viz", action="store_true", help="Visualize the results")
    parser.add_argument("--batch_viz", action="store_true", help="Visualize random traces with scores")
    parser.add_argument("--cascade_2d", help="If cascade is 2D array")
    args = parser.parse_args()
    if args.batch_viz:
        cascade = np.load(args.cascade_2d) if args.cascade_2d else cascade
        from batch_viz import visualize_random_traces_per_style


# Plot 10 random rows; one figure per ranking style, per row
        visualize_random_traces_per_style(
    cascade_array=cascade,
    n=10,
    sigma=2.0,
    seed=42,
    # styles=["prev_depth","next_depth","sum_depth","prev_sharpness","next_sharpness","sum_sharpness"],  # optional
    save_dir="figs_random_per_style",   # or None to display
    dpi=150,
    linewidth=0.8,
    fontsize=10,
    label_every=1
)
        return
    cascade = np.load(args.cascade) if type(args.cascade) == str else cascade_trace
    raw_f = np.load(args.raw_f) if type(args.raw_f) == str else rawf
    peaks_valleys: Tuple[List[Peak], List[Valley]] = find_peaks_and_valleys(cascade) #basically just finding spikes, easy to implement; might need valley class for all rois
    normal_peaks_valleys: Tuple[List[Peak], List[Valley], float, float] = assign_normalized_values(*peaks_valleys) #can be valley and peak method with appropriate input
    normal_peaks = normal_peaks_valleys[0]
    normal_valleys = normal_peaks_valleys[1]
    coupled_peaks_valleys :Tuple[List[Peak], List[Valley]] = couple_peaks_to_valleys(normal_peaks, normal_valleys)#easy; requires list comprehension within roi class probably
    valleys = compute_valley_depths(coupled_peaks_valleys[1], cascade) #easy, can be a valley method
    ranked_valleys, __ = sort_and_rank_valleys(valleys) #sorts valleys by different metrics; roi method probably
    valleys, peaks, styles = compute_all_peak_rank_scores(ranked_valleys, normal_peaks, 2) #spike method probably with valleys as input; completes rank score dictionary for each peak
    if args.viz:
        plot_all_styles(cascade, peaks, styles, sigma_used=2, save_dir="figs")
        from score_viz import (
            make_score_matrix, save_scores_csv,
            plot_score_heatmap, plot_scores_over_time, plot_score_distributions,
            plot_trace_per_style
        )

        # peaks: list of Peak objects with .index, .rank_score filled
        # trace: your smoothed cascade (1D np.ndarray)


        # 1) Matrix / CSV
        df = make_score_matrix(peaks, styles)
        print(df.head())
        save_scores_csv(peaks, styles, out_csv="figs/peak_rank_scores.csv")

        # 2) Heatmap
        plot_score_heatmap(peaks, styles, title="Row 1611 — Rank Scores Heatmap",
                        save_path="figs/rank_scores_heatmap.png")

        # 3) Scores vs time
        plot_scores_over_time(peaks, styles, title="Row 1611 — Scores vs Time",
                            save_path="figs/scores_vs_time.png")

        # 4) Distributions
        plot_score_distributions(peaks, styles, kind="hist", save_path="figs/score_hists.png")

        # 5) Per-style trace overlays (one PNG per style)
        plot_trace_per_style(cascade, peaks, styles, title_prefix="Row 1611",
                            save_dir="figs/trace_per_style", dpi=150, linewidth=0.8, fontsize=10)

if __name__ == '__main__':
    main()
