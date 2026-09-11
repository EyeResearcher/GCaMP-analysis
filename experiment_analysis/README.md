# Experiment analysis

Run after independent recording analysis:

```bash
python -m experiment_analysis run --config experiment.yaml
```

```yaml
schema_version: 1
recordings: recordings.csv
output_dir: experiment_results
comparisons:
  - name: CA1_treatments
    type: treatment
    filter: {region: CA1}
    replicate: animal
    longitudinal: true
    series: field_of_view
    timepoint: day
    alignment: {enabled: false}
  - name: CA1_tracking
    type: longitudinal
    filter: {region: CA1}
    series: field_of_view
    timepoint: day
    alignment:
      enabled: true
      strategy: combined
      top_fraction: 0.1
  - name: hierarchy
    type: hierarchical
    levels: [region, treatment, day]
```

The recording table accepts CSV, Excel, or Parquet. Each row identifies one
bundle using `bundle_path` (a manifest or its directory). An optional
`recording_id` must match the bundle. Other columns supply comparison metadata,
for example `animal`, `region`, `treatment`, `field_of_view`, and `day`.
Configuration paths resolve relative to the YAML file; bundle paths resolve
relative to the recording table. Copy whole bundles when relocating results.

Series IDs must distinguish animals and fields of view. A temporal series has
one recording per integer timepoint and at least two timepoints. Multiple
fields of view may contribute to the same animal/timepoint: treatment results
aggregate recordings within each configured replicate before combining
replicates. Alignment always operates within the separate series ID.

Filters match exact metadata values (a scalar or list of accepted values).
CSV and Excel identifiers are read as strings; temporal comparisons convert
their configured timepoint column to integers. Hierarchical levels define a
metadata tree without depending on acquisition folders.

Every selected bundle is checked for completeness and file integrity. Each
comparison requires matching, nonmissing analysis fingerprints. Alignment also
requires spatial assets and anchor groups of the selected strategy. Validation
occurs before output creation. Missing observations are never manufactured.

Outputs include Excel and CSV descriptive summaries, a resolved configuration
with recording assignments and manifest identities, and optional alignment
outputs. Numbered alignment directories are mapped back to metadata in
`alignment_series.json`. This workflow does not load classifiers, read Excel
analysis reports, or require original Suite2p files. It does not add inferential
statistical tests. Both notebooks use the same APIs as the two entry points.
