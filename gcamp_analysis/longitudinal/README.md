# Longitudinal cell and group tracking

This module registers repeated recordings of the same named region, matches Suite2p ROI masks one-to-one, selects the largest groups on an anchor day, and tracks their members across all available days.

The folder base name is the region identity. Thus `1-1`, `1-1_Day2`, and `1-1_Day10` are repeated observations of region `1-1`; `1-2` is a different region and is never matched to `1-1`. Treatments are also processed independently.

## Method

1. Choose an anchor day (latest by default).
2. Phase-register every snap to the next available timepoint using a log/high-pass representation, then compose the adjacent translations into the requested anchor-day coordinate system.
3. Scale the composed snap translation to Suite2p resolution and translate that day's `stat.npy` masks into anchor coordinates. The current snaps are 2048×2048 and masks are 1024×1024, so the shift scale is 0.5.
4. Generate mask candidates within a centroid-distance limit.
5. Score candidates using 75% mask intersection-over-union and 25% centroid proximity.
6. Use global Hungarian assignment so one ROI cannot match multiple anchor cells.
7. Flag a match as ambiguous when its score is close to another candidate.
8. Rank anchor-day groups by member count and retain the requested top number or fraction.

The anchor should usually be the day whose largest groups are the scientific target. Selecting the latest day answers “when did the cells in the final large groups first become active or grouped?” Selecting an earlier day instead answers “what happened later to the cells in the early large groups?”

## Command

```powershell
python -m gcamp_analysis.longitudinal `
  C:\path\to\experiment `
  --region 1-1 `
  --strategy combined `
  --top-fraction 0.10
```

Omit `--treatment` to process the requested region separately for every treatment. Use `--anchor-day 7` or `--top-n 5` to override the defaults.

## Outputs

Each treatment/region receives:

- `*_registrations.csv`: image shifts, image correlation, match counts, ambiguity counts, and mean mask IoU by day.
- `*_snap_pairwise_registrations.csv`: every adjacent snap-to-snap phase-correlation shift before composition into anchor coordinates.
- `*_skipped_recordings.csv`: recordings excluded because no snap TIFF was available; their montage panel is left blank rather than using a different registration source silently.
- `*_cell_matches.csv`: every anchor-to-day ROI assignment and its quality measures.
- `*_anchor_groups.csv`: selected largest groups and anchor neuron indices.
- `*_cell_history.csv`: one selected anchor cell per day, including detected, active, grouped, daily group ID, and first detected/active/grouped days.
- `*_group_day_summary.csv`: recovery of each selected anchor group, its dominant daily group, member counts, Jaccard overlap, and fragmentation across daily groups.
- `*_membership_overlays.tif`: registered multipage RGB TIFF, one page per day in chronological order.
- `*_overlay_montage.png`: labeled chronological preview of the TIFF pages.
- `*_overlay_legend.csv`: anchor-day group colors and fill-style definitions.

In overlays, color identifies the cell's anchor-day group. A solid mask means the matched cell belongs to a group on that day. A colored outline means the cell was detected and confidently matched, but did not belong to any group on that day. No mark means there was no accepted match. Thus, reading the montage toward day 10 shows when members of the selected day-10 groups first become grouped.

## Quality control and limitations

- Inspect registration correlation, mask IoU, ambiguous flags, and overlays before interpreting membership changes.
- The current transform is translation-only. Regions with appreciable rotation, scaling, shear, or non-rigid tissue deformation need a stronger registration model before cell identities are reliable.
- `first_grouped_day` means first membership in any retained group under the chosen strategy. Group IDs themselves are day-local; the group-day summary uses shared matched cells rather than assuming equal IDs mean equal groups.
- A missing match means “not confidently matched,” not necessarily that the biological cell disappeared.
- The method tracks cells detected in the anchor day. It cannot report cells that never appear in the anchor recording.
