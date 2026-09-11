"""Portable, versioned recording results. No model loading or analysis execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

import numpy as np

from recording_results.serialization import config_fingerprint, json_safe, summary_to_dict
from recording_results.summaries import summary_from_video_record

BUNDLE_VERSION = 1


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _write_json(path, value):
    def normalize(item):
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, dict):
            return {str(k): normalize(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(v) for v in item]
        return json_safe(item)
    path.write_text(json.dumps(normalize(value), indent=2, allow_nan=False), encoding='utf-8')


def _code_identity():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    digest = hashlib.sha256()
    for folder in ('gcamp_analysis', 'utils', 'recording_results'):
        for path in sorted((root / folder).rglob('*.py')):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return {"commit": commit, "dirty": dirty, "source_sha256": digest.hexdigest()}


def write_recording_bundle(record, stats, *, analysis_metadata=None):
    """Publish a complete generation by atomically replacing its manifest last.

    Older generations remain intact, so a failed write cannot corrupt the last
    completed bundle. All data references are relative to the bundle directory.
    """
    root = record.video_dir / 'analysis_results'
    manifest_path = root / 'manifest.json'
    previous = load_recording_bundle(manifest_path) if manifest_path.exists() else None
    recording_id = previous['recording_id'] if previous else str(uuid4())
    generation = root / 'generations' / uuid4().hex
    generation.mkdir(parents=True)
    metadata = dict(analysis_metadata or {})
    files = {}

    def add(name, filename, value):
        path = generation / filename
        _write_json(path, value)
        files[name] = path.relative_to(root).as_posix()

    summary = summary_from_video_record(record, source=record.video_dir.name)
    add('summary', 'summary.json', summary_to_dict(summary))
    neurons = stats.per_neuron_spike_summaries.copy()
    if 'neuron_idx' not in neurons.columns:
        neurons.insert(0, 'neuron_idx', neurons.index)
    add('neurons', 'neurons.json', neurons.to_dict(orient='records'))
    add('groups', 'groups.json', stats.grouping_stats.to_dict(orient='records'))
    add('light_evoked', 'light_evoked.json', {
        str(key): value.to_dict(orient='records') for key, value in stats.light_evoked_details.items()
    })
    plane = record.video_dir / 'suite2p' / 'plane0'
    inputs = {path.name: file_identity(path) for path in sorted(plane.glob('*.npy'))}
    spatial = {}
    if (plane / 'stat.npy').is_file():
        rois = np.load(plane / 'stat.npy', allow_pickle=True)
        if all('ypix' in roi and 'xpix' in roi for roi in rois):
            add('roi_coordinates', 'roi_coordinates.json', [
                {'roi_id': i, 'ypix': np.asarray(roi['ypix']).tolist(),
                 'xpix': np.asarray(roi['xpix']).tolist()} for i, roi in enumerate(rois)
            ])
            spatial['roi_coordinates'] = True
    if (plane / 'ops.npy').is_file():
        ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
        reference = ops.get('meanImg')
        if reference is None:
            reference = ops.get('refImg')
        add('acquisition', 'acquisition.json', {key: ops.get(key) for key in ('fs', 'Ly', 'Lx')})
        if reference is not None:
            path = generation / 'reference.npy'
            np.save(path, np.asarray(reference), allow_pickle=False)
            files['reference_image'] = path.relative_to(root).as_posix()
            spatial['reference_image'] = True
    snaps = sorted(path for path in record.video_dir.iterdir()
                   if path.is_file() and path.stem.endswith('_snap')
                   and path.suffix.lower() in ('.tif', '.tiff'))
    preferred = [path for path in snaps if path.stem == f'{record.video_dir.name}_snap']
    snap = preferred[0] if len(preferred) == 1 else snaps[0] if len(snaps) == 1 else None
    if snap:
        path = generation / 'snap.tif'
        shutil.copyfile(snap, path)
        files['snap_image'] = path.relative_to(root).as_posix()
        inputs['snap_image'] = file_identity(snap)
        spatial['snap_image'] = True
    add('provenance', 'provenance.json', {
        'analysis_metadata': metadata, 'code': _code_identity(),
        'inputs': inputs, 'original_recording_path': str(record.video_dir.resolve()),
    })
    manifest = {
        'artifact_kind': 'gcamp-recording-results', 'schema_version': BUNDLE_VERSION,
        'status': 'complete', 'recording_id': recording_id, 'recording_name': record.video_dir.name,
        'provenance_quality': 'recorded',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config_fingerprint': config_fingerprint(metadata),
        'capabilities': {'summary': True, 'alignment': all(spatial.get(k) for k in
                        ('roi_coordinates', 'reference_image', 'snap_image'))},
        'files': {key: {'path': value, **file_identity(root / value)} for key, value in files.items()},
    }
    temporary = root / f'.manifest-{uuid4().hex}.json'
    _write_json(temporary, manifest)
    temporary.replace(manifest_path)
    return manifest_path


def load_recording_bundle(path):
    """Validate a bundle without consulting original inputs or human reports."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('artifact_kind') != 'gcamp-recording-results' or
            manifest.get('schema_version') != BUNDLE_VERSION or manifest.get('status') != 'complete'):
        raise ValueError(f'Unsupported or incomplete recording bundle: {path}')
    required = {'summary', 'neurons', 'groups', 'provenance'}
    if manifest.get('capabilities', {}).get('alignment'):
        required.update({'roi_coordinates', 'reference_image', 'snap_image'})
    if not required.issubset(manifest.get('files', {})):
        raise ValueError(f'Missing required bundle entries: {path}')
    root = path.parent.resolve()
    for entry in manifest['files'].values():
        relative = Path(entry['path'])
        target = (root / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root):
            raise ValueError(f'Bundle reference escapes its directory: {relative}')
        if not target.is_file() or file_identity(target) != {k: entry[k] for k in ('sha256', 'size_bytes')}:
            raise ValueError(f'Missing or modified bundle file: {target}')
    return manifest
