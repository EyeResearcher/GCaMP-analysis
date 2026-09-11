"""Validated experiment configuration and results-only comparison execution."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

import pandas as pd
import yaml

from recording_results.bundle import load_recording_bundle, file_identity
from recording_results.serialization import summary_from_dict
from recording_results.summaries import aggregate_node_summaries
from experiment_analysis.comparison_utils import summary_to_comparison_row


@dataclass
class Recording:
    path: Path
    manifest: dict
    metadata: dict
    summary: object


def _fields(value, allowed, context):
    if not isinstance(value, dict):
        raise ValueError(f'{context} must be a mapping')
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown {context} fields: {sorted(unknown)}')


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError(f'Use letters, digits, underscores or hyphens for output names: {value!r}')
    return value


def _required(records, columns):
    for record in records:
        for column in columns:
            value = record.metadata.get(column)
            if value is None or pd.isna(value) or str(value).strip() == '':
                raise ValueError(f"Recording {record.manifest['recording_id']} lacks {column!r}")


def _partition(records, columns):
    result = {}
    for record in records:
        key = tuple(record.metadata[c] for c in columns)
        result.setdefault(key, []).append(record)
    return result


def _aggregate(records, columns):
    return {key: aggregate_node_summaries([r.summary for r in group], children_are_videos=True)
            for key, group in _partition(records, columns).items()}


def _table(groups, columns):
    return pd.DataFrame([{**dict(zip(columns, key)),
                          **summary_to_comparison_row('/'.join(map(str, key)), value)}
                         for key, value in sorted(groups.items(), key=lambda item: item[0])])


def _load_recordings(config_path, config):
    metadata_path = (config_path.parent / config['recordings']).resolve()
    # Preserve identifiers such as 001 instead of inferring numeric IDs.
    if metadata_path.suffix.lower() == '.csv':
        frame = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    elif metadata_path.suffix.lower() == '.parquet':
        frame = pd.read_parquet(metadata_path)
    elif metadata_path.suffix.lower() in ('.xlsx', '.xls'):
        frame = pd.read_excel(metadata_path, dtype=str, keep_default_na=False)
    else:
        raise ValueError('Recording metadata must be CSV, Excel, or Parquet')
    if 'bundle_path' not in frame.columns or frame.empty:
        raise ValueError('Recording table requires bundle_path and at least one row')

    records, seen = [], set()
    for row in frame.to_dict(orient='records'):
        bundle_path = (metadata_path.parent / str(row['bundle_path'])).resolve()
        if bundle_path.is_dir():
            bundle_path /= 'manifest.json'
        manifest = load_recording_bundle(bundle_path)
        identity = manifest['recording_id']
        if identity in seen:
            raise ValueError(f'Duplicate recording ID: {identity}')
        if row.get('recording_id') and row['recording_id'] != identity:
            raise ValueError(f'Recording ID does not match bundle: {bundle_path}')
        seen.add(identity)
        row.update(recording_id=identity, bundle_path=str(bundle_path))
        summary = json.loads((bundle_path.parent / manifest['files']['summary']['path']).read_text())
        records.append(Recording(bundle_path, manifest, row, summary_from_dict(summary)))
    return frame, records


def _validate_temporal_settings(spec, subset, kind, columns):
    time = spec.setdefault('timepoint', 'day')
    series = spec.get('series')
    if not isinstance(series, str) or not isinstance(time, str):
        raise ValueError('Longitudinal comparisons require series and timepoint column names')
    columns += [series, time]
    _required(subset, columns)
    for record in subset:
        try:
            day = float(record.metadata[time])
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid numeric timepoint: {record.metadata[time]}') from exc
        if not day.is_integer():
            raise ValueError('Timepoints must be finite integers in a consistent unit')
        record.metadata[time] = int(day)

    partitions = _partition(
        subset,
        (['treatment'] if kind == 'treatment' else []) + [series],
    )
    for key, items in partitions.items():
        days = [record.metadata[time] for record in items]
        if len(days) < 2 or len(days) != len(set(days)):
            raise ValueError(f'Series {key} needs at least two unique timepoints, one recording per timepoint')
        for identity_col in ('animal', 'subject', 'region', 'treatment', spec.get('replicate')):
            if identity_col and len({record.metadata.get(identity_col) for record in items}) > 1:
                raise ValueError(f'Series {key} mixes {identity_col} identities')
    return partitions, time


def _validate_alignment(spec, subset, kind, temporal, partitions, time, name):
    alignment = spec.get('alignment', {})
    _fields(alignment, ('enabled', 'strategy', 'anchor_day', 'top_fraction', 'top_n'), 'alignment')
    if not isinstance(alignment.get('enabled', False), bool):
        raise ValueError('alignment.enabled must be true or false')
    if not alignment.get('enabled'):
        return

    if not temporal or kind == 'hierarchical':
        raise ValueError('Alignment requires a longitudinal comparison')
    if any(not record.manifest['capabilities'].get('alignment') for record in subset):
        raise ValueError(f'{name} requires complete spatial assets for every recording')
    fraction = alignment.get('top_fraction', .1)
    if type(fraction) not in (float, int) or not 0 < fraction <= 1:
        raise ValueError('top_fraction must be greater than zero and at most one')
    if 'anchor_day' in alignment and type(alignment['anchor_day']) is not int:
        raise ValueError('anchor_day must be an integer')
    if not isinstance(alignment.get('strategy', 'combined'), str):
        raise ValueError('Alignment strategy must be a string')
    if 'top_n' in alignment and (type(alignment['top_n']) is not int or alignment['top_n'] < 1):
        raise ValueError('top_n must be a positive integer')

    for items in partitions.values():
        anchor = alignment.get('anchor_day', max(record.metadata[time] for record in items))
        anchors = [record for record in items if record.metadata[time] == anchor]
        if len(anchors) != 1:
            raise ValueError(f'Anchor timepoint {anchor} is absent in a selected series')
        groups = json.loads((anchors[0].path.parent / anchors[0].manifest['files']['groups']['path']).read_text())
        if not any(group.get('method') == alignment.get('strategy', 'combined') for group in groups):
            raise ValueError(f'Anchor has no groups for the configured alignment strategy in {name}')


def _validate_comparison(spec, frame, records, names):
    _fields(spec, ('name', 'type', 'filter', 'replicate', 'series', 'timepoint',
                   'longitudinal', 'alignment', 'levels'), 'comparison')
    name = _name(spec.get('name'))
    if name in names:
        raise ValueError(f'Duplicate comparison name: {name}')
    names.add(name)
    kind = spec.get('type')
    if kind not in ('treatment', 'longitudinal', 'hierarchical'):
        raise ValueError(f'Unknown comparison type: {kind}')
    filters = spec.get('filter', {})
    if not isinstance(filters, dict) or any(key not in frame.columns for key in filters):
        raise ValueError(f'Invalid metadata filter in {name}')
    subset = [
        Recording(record.path, record.manifest, dict(record.metadata), record.summary)
        for record in records
        if all(record.metadata.get(key) in (value if isinstance(value, list) else [value])
               for key, value in filters.items())
    ]
    if not subset:
        raise ValueError(f'Comparison {name} selects no recordings')
    fingerprints = {record.manifest.get('config_fingerprint') for record in subset}
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError(f'Comparison {name} has missing or incompatible analysis fingerprints')
    if 'longitudinal' in spec and not isinstance(spec['longitudinal'], bool):
        raise ValueError('longitudinal must be true or false')

    temporal = kind == 'longitudinal' or spec.get('longitudinal', False)
    if kind == 'hierarchical' and (
        spec.get('longitudinal') or any(key in spec for key in ('series', 'timepoint', 'replicate'))
    ):
        raise ValueError('Hierarchical comparisons use levels, not replicate/series/timepoint settings')
    if kind == 'longitudinal' and 'replicate' in spec:
        raise ValueError('Use treatment comparisons to aggregate biological replicates')

    columns = []
    if kind == 'treatment':
        if not isinstance(spec.get('replicate'), str):
            raise ValueError('Treatment comparisons require a replicate column')
        columns += ['treatment', spec['replicate']]
        _required(subset, columns)
        if len({record.metadata['treatment'] for record in subset}) < 2:
            raise ValueError(f'{name} requires at least two treatments')
    if kind == 'hierarchical':
        levels = spec.get('levels')
        if not isinstance(levels, list) or not levels or any(not isinstance(column, str) for column in levels):
            raise ValueError('Hierarchical comparisons require a list of metadata levels')
        columns += levels

    partitions, time = None, None
    if temporal:
        partitions, time = _validate_temporal_settings(spec, subset, kind, columns)
    _required(subset, columns)
    _validate_alignment(spec, subset, kind, temporal, partitions, time, name)
    return spec, subset


def load_experiment(path):
    """Validate all comparisons and bundles before creating any outputs."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    _fields(config, ('schema_version', 'recordings', 'output_dir', 'comparisons'), 'experiment')
    if config.get('schema_version') != 1:
        raise ValueError('Experiment schema_version must be 1')
    for field in ('recordings', 'output_dir'):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ValueError(f'{field} must be a nonempty path')

    frame, records = _load_recordings(path, config)
    comparisons = config.get('comparisons')
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError('comparisons must be a nonempty list')
    names, selected = set(), []
    for spec in comparisons:
        selected.append(_validate_comparison(spec, frame, records, names))

    output = (path.parent / config['output_dir']).resolve()
    for record in records:
        if output.is_relative_to(record.path.parent):
            raise ValueError('Experiment outputs must be outside recording bundles')
    return config, selected, output

def comparison_tables(spec, records):
    tables = {'recordings': pd.DataFrame([{**r.metadata, **summary_to_comparison_row(
        r.manifest['recording_name'], r.summary)} for r in records])}
    kind = spec['type']
    if kind == 'longitudinal':
        keys = [spec['series'], spec['timepoint']]
        tables['series_timepoint'] = _table(_aggregate(records, keys), keys)
    elif kind == 'treatment':
        keys = ['treatment', spec['replicate']]
        replicate = _aggregate(records, keys)
        tables['replicate_summary'] = _table(replicate, keys)
        treatments = {}
        for key, summary in replicate.items():
            treatments.setdefault((key[0],), []).append(summary)
        tables['treatment_summary'] = _table({key: aggregate_node_summaries(values, children_are_videos=False)
                                             for key, values in treatments.items()}, ['treatment'])
        if spec.get('longitudinal'):
            time = spec['timepoint']
            replicate_day = _aggregate(records, keys + [time])
            tables['replicate_timepoint'] = _table(replicate_day, keys + [time])
            treatment_day = {}
            for key, value in replicate_day.items():
                treatment_day.setdefault((key[0], key[2]), []).append(value)
            tables['treatment_timepoint'] = _table({key: aggregate_node_summaries(values, children_are_videos=False)
                for key, values in treatment_day.items()}, ['treatment', time])
    else:
        levels = spec['levels']
        groups = _aggregate(records, levels)
        tables[f'level_{len(levels)}'] = _table(groups, levels)
        for depth in range(len(levels) - 1, 0, -1):
            parents = {}
            for key, value in groups.items():
                parents.setdefault(key[:depth], []).append(value)
            groups = {key: aggregate_node_summaries(values, children_are_videos=False) for key, values in parents.items()}
            tables[f'level_{depth}'] = _table(groups, levels[:depth])
    return tables


def run_experiment(config_path):
    config, comparisons, output = load_experiment(config_path)
    # Heavy spatial dependencies are imported only when alignment is requested.
    output.mkdir(parents=True, exist_ok=True)
    resolved = {'config': config, 'comparisons': {spec['name']: [
        {'metadata': r.metadata, 'manifest_identity': file_identity(r.path)} for r in records]
        for spec, records in comparisons}}
    (output / 'resolved_experiment.json').write_text(json.dumps(resolved, indent=2), encoding='utf-8')
    for spec, records in comparisons:
        destination = output / spec['name']
        destination.mkdir(exist_ok=True)
        tables = comparison_tables(spec, records)
        warnings = []
        if spec['type'] == 'treatment':
            for (treatment,), items in _partition(records, ['treatment']).items():
                count = len({r.metadata[spec['replicate']] for r in items})
                if count < 2:
                    warnings.append({'severity': 'warning', 'code': 'low_replicate_count',
                                     'message': f'{treatment} has only {count} independent replicate'})
        tables['validation'] = pd.DataFrame(warnings, columns=['severity', 'code', 'message'])
        with pd.ExcelWriter(destination / 'comparisons.xlsx', engine='openpyxl') as writer:
            for name, table in tables.items():
                table.to_excel(writer, sheet_name=name, index=False)
                table.to_csv(destination / f'{name}.csv', index=False)
        alignment = spec.get('alignment', {})
        if alignment.get('enabled'):
            from experiment_analysis.longitudinal.tracking import LongitudinalTracker
            from experiment_analysis.longitudinal.models import RecordingRef
            keys = (['treatment'] if spec['type'] == 'treatment' else []) + [spec['series']]
            mapping = []
            for index, (key, items) in enumerate(_partition(records, keys).items(), start=1):
                label = f'series_{index:04d}'
                mapping.append({'output_series': label, 'metadata': dict(zip(keys, key))})
                refs = [RecordingRef('aligned', label, r.metadata[spec['timepoint']],
                        r.manifest['recording_name'], r.path.parent, r.path.parent, r.path,
                        bundle_path=r.path) for r in items]
                tracker = LongitudinalTracker(destination, strategy=alignment.get('strategy', 'combined'))
                tracker.run(treatment='aligned', region=label, output_dir=destination,
                            recordings=refs, **{k: alignment[k] for k in ('anchor_day', 'top_fraction', 'top_n') if k in alignment})
            (destination / 'alignment_series.json').write_text(json.dumps(mapping, indent=2))
    return output

