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
except ModuleNotFoundError:
    numba_stub = ModuleType("numba")
    numba_stub.njit = lambda function=None, **kwargs: (
        function if function is not None else lambda decorated: decorated
    )
    numba_typed_stub = ModuleType("numba.typed")
    numba_typed_stub.List = list
    sys.modules["numba"] = numba_stub
    sys.modules["numba.typed"] = numba_typed_stub

import gcamp_analysis.experiments.processor as processor_module
import main as main_module
from gcamp_analysis.experiments.processor import ExperimentProcessor


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
    processor = ExperimentProcessor(
        runner=runner,
        output_root=tmp_path,
        dry_run=True,
    )
    record = processor._process_one_video(video_dir, verbose=False)

    assert runner.calls == 1
    assert record.video_dir == video_dir
    assert record.n_rois_total == 4
    assert video.cleared
    assert not (video_dir / "metrics").exists()


def test_main_dry_run_skips_experiment_comparison_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, bool] = {}
    fake_tree = object()

    class FakeBuilder:
        def __init__(self, is_video_dir) -> None:
            pass

        def build(self, root: Path) -> object:
            return fake_tree

    class FakeProcessor:
        def __init__(
            self,
            runner,
            output_root: Path,
            dry_run: bool,
            analysis_metadata: dict | None = None,
        ) -> None:
            captured["dry_run"] = dry_run
            captured["has_analysis_metadata"] = analysis_metadata is not None

        def process_tree(self, tree: object, verbose: bool) -> None:
            captured["processed"] = True

        def compare_siblings(self, tree: object) -> dict:
            captured["compared"] = True
            return {}

    monkeypatch.setattr(main_module, "load_config", lambda path: {"models": {}})
    monkeypatch.setattr(
        main_module,
        "load_model_bundle",
        lambda config: {
            "roi": (object(), {}),
            "spike": (object(), {}),
        },
    )
    monkeypatch.setattr(
        main_module.VideoPipelineRunner,
        "build",
        lambda config, models, sensor_type: object(),
    )
    monkeypatch.setattr(main_module, "ExperimentTreeBuilder", FakeBuilder)
    monkeypatch.setattr(main_module, "ExperimentProcessor", FakeProcessor)
    monkeypatch.setattr(
        main_module,
        "save_comparisons",
        lambda **kwargs: pytest.fail(
            "Experiment comparisons were written during a dry run"
        ),
    )

    main_module.main(
        experiment_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        verbose=False,
        dry_run=True,
    )

    assert captured == {
        "dry_run": True,
        "has_analysis_metadata": True,
        "processed": True,
        "compared": True,
    }
