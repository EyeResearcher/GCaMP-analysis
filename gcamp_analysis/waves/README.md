# Retinal-wave analysis

`gcamp_analysis.waves` detects spatially propagating calcium episodes from the
existing pipeline's analysis neurons, Suite2p fluorescence, ROI coordinates,
and source TIFFs.

The analysis makes two independently shuffled comparisons:

1. Candidate population episodes must exceed the recording-level maximum
   expected after circularly shifting each neuron's rising-edge events.
2. Within a candidate episode, planar and radial activation-time fits are
   compared with activation times permuted over the fixed ROI positions. Model
   selection is repeated inside each permutation.

Significant ROI episodes are then fit independently in 16 × 16-pixel blocks of
the source TIFF. Cross-modal corroboration requires a significant movie fit,
the same model family, and a matched direction or origin.

Run selected days:

```bash
python -m gcamp_analysis.waves /path/to/dataset --output-dir wave_results --days 10
```

Consolidate separately run day folders:

```bash
python scripts/summarize_wave_analysis.py wave_results
```

These outputs support claims about propagating calcium activity. They do not
directly establish action-potential propagation.
