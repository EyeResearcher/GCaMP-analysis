"""Common recording-analysis API for the CLI and notebook."""
from pathlib import Path
from copy import deepcopy


def analyze_recordings(root, config, *, sensor=None, dry_run=False, verbose=True):
    from utils.io_utils import load_config, load_model_bundle
    from gcamp_analysis.video_runner import VideoPipelineRunner
    from gcamp_analysis.recording_processor import RecordingProcessor

    config = load_config(Path(config)) if isinstance(config, (str, Path)) else deepcopy(config)
    bundle = load_model_bundle(config['models'])
    models = {'roi': bundle['roi'][0], 'roi_config': bundle['roi'][1],
              'spike': bundle['spike'][0], 'spike_config': bundle['spike'][1]}
    runner = VideoPipelineRunner.build(config, models, sensor)
    processor = RecordingProcessor(runner, dry_run=dry_run, analysis_metadata={
        'config': config, 'sensor_type': sensor or config.get('traces', {}).get('sensor_type'),
        'models': bundle.provenance,
    })
    return processor.process_directory(Path(root), verbose=verbose)
