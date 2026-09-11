from pathlib import Path
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from recording_results.models import VideoRunRecord
from recording_results.bundle import load_recording_bundle, write_recording_bundle


def fixture_record(tmp_path):
    video = tmp_path / 'recording'
    plane = video / 'suite2p' / 'plane0'
    plane.mkdir(parents=True)
    np.save(plane / 'F.npy', np.ones((2, 4)))
    record = VideoRunRecord(video, video / 'metrics', 2, 2, 1, 3)
    stats = SimpleNamespace(
        per_neuron_spike_summaries=pd.DataFrame({'number_of_spikes': [3]}, index=[1]),
        grouping_stats=pd.DataFrame({'method': ['corr'], 'group_id': ['g1'],
                                     'neuron_indices': [np.array([1])]}),
        light_evoked_details={},
    )
    return record, stats, plane


def test_bundle_relocates_without_source_or_workbook(tmp_path):
    record, stats, _ = fixture_record(tmp_path)
    path = write_recording_bundle(record, stats)
    first = load_recording_bundle(path)
    assert not first['capabilities']['alignment']
    moved = tmp_path / 'moved'
    shutil.copytree(path.parent, moved)
    shutil.rmtree(record.video_dir)
    manifest = load_recording_bundle(moved / 'manifest.json')
    neurons = json.loads((moved / manifest['files']['neurons']['path']).read_text())
    groups = json.loads((moved / manifest['files']['groups']['path']).read_text())
    assert neurons[0]['neuron_idx'] == 1
    assert groups[0]['neuron_indices'] == [1]


def test_bundle_identity_integrity_and_failed_publication(tmp_path, monkeypatch):
    record, stats, _ = fixture_record(tmp_path)
    path = write_recording_bundle(record, stats)
    first = load_recording_bundle(path)
    write_recording_bundle(record, stats)
    second = load_recording_bundle(path)
    assert first['recording_id'] == second['recording_id']
    import recording_results.bundle as results
    def fail(*args):
        raise RuntimeError('interrupted')
    monkeypatch.setattr(results, '_code_identity', fail)
    with pytest.raises(RuntimeError):
        write_recording_bundle(record, stats)
    assert load_recording_bundle(path) == second
    target = path.parent / second['files']['neurons']['path']
    target.write_text('[]')
    with pytest.raises(ValueError, match='modified'):
        load_recording_bundle(path)


@pytest.mark.parametrize('extension', ['tif', 'tiff'])
def test_alignment_assets_are_portable(tmp_path, extension):
    record, stats, plane = fixture_record(tmp_path)
    np.save(plane / 'stat.npy', np.array([{'ypix': [1], 'xpix': [2]}], dtype=object))
    np.save(plane / 'ops.npy', {'meanImg': np.ones((4, 4)), 'fs': 15, 'Ly': 4, 'Lx': 4})
    import tifffile
    tifffile.imwrite(record.video_dir / f'recording_snap.{extension}', np.ones((4, 4), dtype='uint16'))
    path = write_recording_bundle(record, stats)
    assert load_recording_bundle(path)['capabilities']['alignment']


def test_recording_processor_dry_run(tmp_path, monkeypatch):
    import gcamp_analysis.recording_processor as module
    record, stats, plane = fixture_record(tmp_path)
    np.save(plane / 'iscell.npy', np.ones((2, 2)))
    video = SimpleNamespace(n_rois=2, n_good_rois=2, neurons=[], grouping_results={},
                            summary_df=pd.DataFrame(), suite2p_data={}, clear_results=lambda: None)
    calls = []
    monkeypatch.setattr(module.Video, 'from_suite2p', lambda **kw: video)
    monkeypatch.setattr(module.VideoStatistics, 'from_video', lambda v: stats)
    def forbidden(*args, **kwargs):
        pytest.fail('Dry run wrote results')
    monkeypatch.setattr(module, 'write_recording_bundle', forbidden)
    monkeypatch.setattr(module, 'VideoStatisticsWriter', forbidden)
    processor = module.RecordingProcessor(SimpleNamespace(run=lambda *a, **k: calls.append(1)), dry_run=True)
    assert len(processor.process_directory(record.video_dir)) == 1
    assert calls == [1]
    assert not (record.video_dir / 'analysis_results').exists()


def test_recording_processing_publishes_bundle_without_comparisons(tmp_path, monkeypatch):
    import gcamp_analysis.recording_processor as module
    record, stats, plane = fixture_record(tmp_path)
    np.save(plane / 'iscell.npy', np.ones((2, 2)))
    video = SimpleNamespace(n_rois=2, n_good_rois=2, neurons=[], grouping_results={},
                            summary_df=pd.DataFrame(), suite2p_data={}, clear_results=lambda: None)
    monkeypatch.setattr(module.Video, 'from_suite2p', lambda **kw: video)
    monkeypatch.setattr(module.VideoStatistics, 'from_video', lambda v: stats)
    monkeypatch.setattr(module, 'VideoStatisticsWriter', lambda: SimpleNamespace(write=lambda *a, **k: None))
    monkeypatch.setattr(module, 'VideoFiguresWriter', lambda: SimpleNamespace(write=lambda *a: None))
    processor = module.RecordingProcessor(SimpleNamespace(run=lambda *a, **k: None))
    processor.process_directory(tmp_path)
    assert load_recording_bundle(record.video_dir / 'analysis_results/manifest.json')['status'] == 'complete'
    assert not list(tmp_path.rglob('sibling_comparisons.xlsx'))
