import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from recording_results.models import VideoRunRecord
from recording_results.bundle import write_recording_bundle
from experiment_analysis.workflow import load_experiment, run_experiment


def make_experiment(tmp_path, *, spatial=False):
    rows = []
    for treatment in ('control', 'drug'):
        for field in ('a', 'b'):
            for day in (1, 2):
                video = tmp_path / 'sources' / f'{treatment}_{field}_{day}'
                plane = video / 'suite2p/plane0'
                plane.mkdir(parents=True)
                if spatial:
                    import tifffile
                    image = np.random.default_rng(1).normal(size=(32, 32)).astype('float32')
                    np.save(plane / 'ops.npy', {'meanImg': image})
                    np.save(plane / 'stat.npy', np.array([{'ypix': [10, 10, 11], 'xpix': [10, 11, 10]}], dtype=object))
                    tifffile.imwrite(video / f'{video.name}_snap.tif', image)
                record = VideoRunRecord(video, video / 'metrics', 1, 1, 1, 1)
                stats = SimpleNamespace(per_neuron_spike_summaries=pd.DataFrame({'neuron_idx': [0]}),
                    grouping_stats=pd.DataFrame({'method': ['combined'], 'group_id': ['g1'], 'neuron_indices': [[0]]}),
                    light_evoked_details={})
                manifest = write_recording_bundle(record, stats, analysis_metadata={'config': {'same': True}})
                moved = tmp_path / 'bundles' / video.name
                shutil.copytree(manifest.parent, moved)
                rows.append({'bundle_path': str(moved.relative_to(tmp_path)), 'treatment': treatment,
                             'animal': treatment + '_animal', 'series': treatment + '_' + field, 'day': day})
    shutil.rmtree(tmp_path / 'sources')
    pd.DataFrame(rows).to_csv(tmp_path / 'recordings.csv', index=False)
    config = {'schema_version': 1, 'recordings': 'recordings.csv', 'output_dir': 'outputs',
              'comparisons': [{'name': 'treatments', 'type': 'treatment', 'replicate': 'animal',
                               'series': 'series', 'timepoint': 'day', 'longitudinal': True}]}
    path = tmp_path / 'experiment.yaml'
    path.write_text(yaml.safe_dump(config))
    return path, config


def test_relocated_results_multiple_fields_per_replicate(tmp_path):
    path, _ = make_experiment(tmp_path)
    output = run_experiment(path)
    table = pd.read_csv(output / 'treatments/replicate_timepoint.csv')
    assert len(table) == 4
    assert set(table.n_videos) == {2}
    before = {p: p.read_bytes() for p in (tmp_path / 'bundles').rglob('*') if p.is_file()}
    run_experiment(path)
    assert all(p.read_bytes() == value for p, value in before.items())


def test_invalid_config_writes_nothing(tmp_path):
    path, config = make_experiment(tmp_path)
    config['comparisons'][0]['replicat'] = 'animal'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='Unknown'):
        run_experiment(path)
    assert not (tmp_path / 'outputs').exists()


def test_duplicate_series_timepoint_rejected(tmp_path):
    path, config = make_experiment(tmp_path)
    config['comparisons'][0]['series'] = 'animal'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='unique timepoints'):
        load_experiment(path)


def test_cli_does_not_import_classifiers(tmp_path):
    path, _ = make_experiment(tmp_path)
    script = '''
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname in ('gcamp_analysis.video_runner', 'utils.io_utils', 'sklearn', 'joblib'):
            raise RuntimeError('Forbidden analysis dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from experiment_analysis.__main__ import main
main(['run', '--config', sys.argv[1]])
'''
    result = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_alignment_reads_only_portable_assets(tmp_path, monkeypatch):
    path, config = make_experiment(tmp_path, spatial=True)
    config['comparisons'][0]['alignment'] = {'enabled': True, 'strategy': 'combined', 'top_n': 1}
    path.write_text(yaml.safe_dump(config))
    def forbidden(*args, **kwargs):
        pytest.fail('Alignment read an Excel report')
    monkeypatch.setattr(pd, 'read_excel', forbidden)
    output = run_experiment(path)
    mapping = json.loads((output / 'treatments/alignment_series.json').read_text())
    assert len(mapping) == 4
    assert len(list(output.rglob('*membership_overlays.tif'))) == 4


def test_longitudinal_and_hierarchy_outputs(tmp_path):
    path, config = make_experiment(tmp_path)
    config['comparisons'] = [
        {'name': 'days', 'type': 'longitudinal', 'series': 'series', 'timepoint': 'day'},
        {'name': 'tree', 'type': 'hierarchical', 'levels': ['treatment', 'animal']},
    ]
    path.write_text(yaml.safe_dump(config))
    output = run_experiment(path)
    days = pd.read_csv(output / 'days/series_timepoint.csv')
    assert len(days) == 8
    hierarchy = pd.read_csv(output / 'tree/level_1.csv')
    assert set(hierarchy.n_videos) == {4}


def test_alignment_missing_assets_rejected_before_output(tmp_path):
    path, config = make_experiment(tmp_path)
    config['comparisons'][0]['alignment'] = {'enabled': True}
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='spatial assets'):
        run_experiment(path)
    assert not (tmp_path / 'outputs').exists()
