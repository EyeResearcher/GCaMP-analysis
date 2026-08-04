# `scripts`

Standalone command-line scripts that drive, corroborate, and visualize the
retinal-wave analysis in `gcamp_analysis.waves`. They are research/operational
utilities, not part of the importable library, and each is run directly with
`python scripts/<name>.py` (use `--help` where available).

## Run and batch

| Script | Purpose |
|---|---|
| `run_neighbor_xcorr_waves.py` | Neighbor-graph cross-correlation wave detection over a dataset. |
| `run_raw_movie_waves.py` / `run_raw_movie_day10_batch.py` | Fit propagation directly on block-averaged raw movies (single run / day-10 batch). |
| `run_waveminer_compatible.py` | Run the WaveMiner-compatible detector for cross-method comparison. |
| `waveminer_invitro_nulls.py` | Build in-vitro null distributions for WaveMiner-style statistics. |

## Corroborate and summarize

| Script | Purpose |
|---|---|
| `confirm_movie_waves_with_neighbors.py` / `confirm_waveminer_with_neighbors.py` | Cross-check ROI/movie episodes against neighbor-graph evidence. |
| `analyze_waveminer_stability.py` | Assess stability of WaveMiner results across parameters. |
| `summarize_wave_analysis.py` | Consolidate per-day wave-analysis output folders into combined tables. |
| `summarize_raw_movie_waves.py` | Aggregate raw-movie wave results. |

## Plot and report

| Script | Purpose |
|---|---|
| `plot_wave_episodes.py`, `plot_wave_recurrence.py` | Episode and recurrence-cluster figures. |
| `plot_raw_movie_waves.py`, `plot_waveminer_candidate.py`, `plot_waveminer_timeline.py` | Movie-front, candidate, and timeline figures. |
| `build_wave_gallery.py` | Assemble a gallery image from diagnostic PNGs. |
| `build_visualization_figure_report.py` | Build the visualization figure report. |
| `copy_minimal_visualization_data.ps1` | Copy the minimal dataset needed to reproduce visualization outputs. |

## Notes

- These scripts consume outputs of the main pipeline (per-video metrics) and of
  `gcamp_analysis.waves`; run those first.
- They are intentionally decoupled from the library so experimental analyses can
  evolve without affecting the published `gcamp_analysis` API.
