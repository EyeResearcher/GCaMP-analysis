# Analysis data objects

This subpackage defines the mutable in-memory objects passed between processing stages. Most objects are populated progressively; a field being present does not mean it is already scientifically valid until its responsible stage has run.

## Object roles

| Object | Main input | Main populated outputs |
|---|---|---|
| `Video` | Video path, `suite2p/plane0` path, loaded Suite2p arrays | Processed trace arrays, ROIs, neurons, per-neuron summaries, and grouping results |
| `ROI` | Original Suite2p row index, raw `F` trace, optional `stat` and `Fneu` | ROI features, candidate peaks, `is_good`, and per-section activity flags |
| `Neuron` | An ROI accepted by the ROI classifier | Accepted spikes, per-spike statistics, and a per-neuron summary row |
| `Spike` | Peak frame and position among a neuron's candidates | Prominence, local window, classifier validity, section label, and kinetic statistics |
| `NeuronGroup` | A clustering ID, member neurons, strategy name, and similarity-matrix row indices | Group size, member indices, metadata, and within-group matrix summaries |

## Suite2p inputs held by `Video`

The required item is `F`, shaped `(n_rois, n_frames)`. The loader may also provide `Fneu`, `stat`, `ops`, and `iscell`. `ops.fs` supplies the sampling rate when no explicit `fs` value exists; the fallback is 15 Hz.

`ROI.index` and `Neuron.index` refer to the original Suite2p row. `Neuron.filtered_index` is a compact, changing position among retained neurons. Saved `neuron_idx` values use the stable Suite2p row index.

## Per-neuron summary fields

After spike processing, each summary row contains:

- `neuron_idx` and `filtered_index`;
- `spike_frequency` and `number_of_spikes`;
- accepted `spike_indices`, normalized peak values, and raw peak values;
- `mean_<kinetic>` and `var_<kinetic>` for each numeric per-spike metric.

Current kinetic bases are `rise_slope_hz`, `decay_tau_seconds`, and `half_max_width_seconds`. Pandas variance uses its sample-variance default (`ddof=1`), so a neuron with only one valid value generally has `NaN` variance.

## Group meaning

`NeuronGroup` is a container for a result from one clustering strategy. It does not independently prove biological connectivity. `size` is the member-neuron count, `neuron_indices` contains stable Suite2p indices, and `row_indices` locates those neurons in the strategy matrix.

