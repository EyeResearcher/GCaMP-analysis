# Independent recording analysis

Activate the `gcamp` environment, then run:

```bash
python -m gcamp_analysis analyze /path/to/recordings --config /path/to/analysis.yaml
```

The configuration uses the existing pipeline model, trace, event, and grouping
settings. `--sensor` overrides the configured sensor. `--dry-run` computes
recording results without writing files; `--quiet` reduces progress output.
This command does not aggregate folders or run experiment comparisons.

Each recording receives existing reports under `metrics/` and a portable
`analysis_results/manifest.json` with generation files below `analysis_results/`.
Copy the entire `analysis_results` directory when moving results. The manifest
contains relative file references, SHA-256 checksums, a stable recording ID,
schema version, and available capabilities. A new manifest is published only
after a generation is complete; previous generations are retained.

The bundle includes summary values, neuron metrics with original neuron IDs,
group memberships, light-evoked details, configuration/model provenance, and
input hashes. Available Suite2p ROI coordinates and reference images plus a
recording snap image are exported for future spatial alignment. Missing spatial
assets mark alignment unavailable without preventing summary comparisons.
Original recording paths are provenance only. Model file hashes are supplied
automatically by the command; programmatic callers must supply their analysis
metadata when constructing `RecordingProcessor`.

The CLI and recording notebook call `gcamp_analysis.api.analyze_recordings`.
Experiment analysis runs separately with `python -m experiment_analysis run`.
Shared contracts live in `recording_results`; combined execution is removed.
