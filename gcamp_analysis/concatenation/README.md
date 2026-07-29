# Concatenation metadata

This subpackage describes the sections of a Suite2p video that was formed by concatenating multiple source videos. It does not concatenate image data itself; `preprocessing/deconcat_videos.py` and the upstream acquisition/preprocessing workflow handle image files.

## Input

When `concatenated.enabled` is true, each video directory must contain exactly one `*_concat_order.csv`. Its first five columns, in order, are:

| Column | Meaning |
|---|---|
| `index` | Source section order/identifier. |
| `source file name` | Original source-video name. |
| `section type` | `baseline`, `treatment`, or `recovery`. |
| `start frame` | First included frame, zero-based. |
| `end frame` | First excluded frame. Section length is `end frame - start frame`. |

Rows must be ordered, non-overlapping, within the length of `F.npy`, and contain exactly one baseline. Multiple treatment or recovery sections are allowed.

## Output

`load_concat_metadata` returns a `ConcatMetadata` object containing the source CSV, its DataFrame, and validated `ConcatSection` records. Stable section keys are:

- `baseline` for the unique baseline;
- `treatment_1`, `treatment_2`, ...;
- `recovery_1`, `recovery_2`, ...

These keys become prefixes in per-neuron columns such as `treatment_1_spike_frequency` and suffixes in section-comparison filenames.

This module writes no files. Downstream ROI, spike, grouping, and reporting services use its section boundaries.

## Important interpretation

Sections are temporal regions within the **same Suite2p trace matrix**. This supports baseline-to-section comparisons for the same ROI indices. Separate day folders are not treated as concatenated sections and are not neuron-identity matched by this module.

