from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pandas as pd
import pytest

# The default test interpreter used by lightweight CI may not include numba.
# These tests do not execute accelerated similarity functions, so a decorator
# passthrough is sufficient to import the orchestration layer under test.
try:
    import numba  # noqa: F401
except ImportError:
    numba_stub = ModuleType("numba")
    numba_stub.njit = lambda function=None, **kwargs: (
        function if function is not None else lambda decorated: decorated
    )
    numba_typed_stub = ModuleType("numba.typed")
    numba_typed_stub.List = list
    sys.modules["numba"] = numba_stub
    sys.modules["numba.typed"] = numba_typed_stub

import gcamp_analysis.recording_processor as processor_module
import main as main_module
from gcamp_analysis.recording_processor import RecordingProcessor


def test_processor_dry_run_analyzes_without_invoking_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVideo:
        n_rois = 4
        n_good_rois = 3
        neurons = []
        grouping_results = {}
        summary_df = pd.DataFrame()
        suite2p_data = {}

        def __init__(self) -> None:
            self.cleared = False

        def clear_results(self) -> None:
            self.cleared = True

    class FakeRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, video: FakeVideo, verbose: bool = True) -> None:
            self.calls += 1

    class ForbiddenWriter:
        def __init__(self, *args, **kwargs) -> None:
            pytest.fail("A filesystem writer was invoked during a dry run")

    video = FakeVideo()
    runner = FakeRunner()
    monkeypatch.setattr(
        processor_module.Video,
        "from_suite2p",
        lambda **kwargs: video,
    )
    monkeypatch.setattr(
        processor_module.VideoStatistics,
        "from_video",
        lambda video: SimpleNamespace(light_evoked_details={}),
    )
    monkeypatch.setattr(processor_module, "VideoStatisticsWriter", ForbiddenWriter)
    monkeypatch.setattr(processor_module, "VideoFiguresWriter", ForbiddenWriter)

    video_dir = tmp_path / "video"
    processor = RecordingProcessor(
        runner=runner,
        dry_run=True,
    )
    record = processor.process_recording(video_dir, verbose=False)

    assert runner.calls == 1
    assert record.video_dir == video_dir
    assert record.n_rois_total == 4
    assert video.cleared
    assert not (video_dir / "metrics").exists()


def test_combined_workflow_is_removed():
    with pytest.raises(SystemExit, match='Combined execution has been removed'):
        main_module.main()
