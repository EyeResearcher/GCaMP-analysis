import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import sklearn

from utils.io_utils import load_model, load_model_bundle, load_suite2p_data


def _write_model_bundle(root: Path) -> dict:
    model_specs = {
        "roi": (["peak_density", "range_trace"], "sqrt"),
        "spike": (["spike_prom", "distance"], "raw"),
    }
    manifest = {
        "scikit_learn_version": sklearn.__version__,
        "models": {},
    }
    iteration = "iteration-001"
    for name, (features, transform) in model_specs.items():
        folder = root / name / iteration
        folder.mkdir(parents=True)
        model_path = folder / "model.joblib"
        config_path = folder / "results.json"
        joblib.dump(SimpleNamespace(feature_names_in_=np.array(features)), model_path)
        config_path.write_text(
            json.dumps({"features": features, "transform": transform}),
            encoding="utf-8",
        )
        manifest["models"][name] = {
            "model": f"{name}/{iteration}/model.joblib",
            "config": f"{name}/{iteration}/results.json",
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "features": features,
            "transform": transform,
        }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_load_suite2p_data_memory_maps_arrays_and_slims_ops(tmp_path: Path) -> None:
    fluorescence = np.arange(12, dtype=np.float32).reshape(3, 4)
    iscell = np.ones((3, 2), dtype=np.float64)
    np.save(tmp_path / "F.npy", fluorescence)
    np.save(tmp_path / "Fneu.npy", fluorescence + 1)
    np.save(tmp_path / "spks.npy", fluorescence + 2)
    np.save(tmp_path / "iscell.npy", iscell)
    np.save(
        tmp_path / "ops.npy",
        {
            "fs": 30.0,
            "Ly": 512,
            "Lx": 256,
            "regPC": np.ones((2, 4, 8, 8), dtype=np.float32),
            "meanImg": np.ones((8, 8), dtype=np.float32),
        },
        allow_pickle=True,
    )

    data = load_suite2p_data(tmp_path)

    assert isinstance(data["F"], np.memmap)
    assert isinstance(data["Fneu"], np.memmap)
    assert isinstance(data["spks"], np.memmap)
    assert isinstance(data["iscell"], np.memmap)
    np.testing.assert_array_equal(data["F"], fluorescence)
    np.testing.assert_array_equal(data["Fneu"], fluorescence + 1)
    np.testing.assert_array_equal(data["spks"], fluorescence + 2)
    assert data["ops"] == {"fs": 30.0, "Ly": 512, "Lx": 256}
    assert data["fs"] == 30.0


def test_load_suite2p_data_supplies_empty_ops_when_file_is_absent(
    tmp_path: Path,
) -> None:
    fluorescence = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.save(tmp_path / "F.npy", fluorescence)
    np.save(tmp_path / "iscell.npy", np.ones((2, 2), dtype=np.float64))

    data = load_suite2p_data(tmp_path)

    assert data["ops"] == {}
    assert data["fs"] == 15.0
    np.testing.assert_array_equal(data["Fneu"], np.zeros_like(fluorescence))


def test_load_model_preserves_single_local_model_loading(tmp_path: Path) -> None:
    model_path = tmp_path / "roi.joblib"
    joblib.dump(SimpleNamespace(name="roi"), model_path)

    model, config = load_model({"roi_model_path": str(model_path)}, "roi")

    assert model.name == "roi"
    assert config is None


def test_huggingface_bundle_downloads_one_snapshot_for_both_models(
    tmp_path: Path, monkeypatch
) -> None:
    _write_model_bundle(tmp_path)
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    bundle = load_model_bundle(
        {
            "source": "huggingface",
            "repo_id": "example/gcamp-analysis-models",
            "revision": "v1.0.0",
        }
    )

    assert list(bundle) == ["roi", "spike"]
    assert list(bundle["roi"][0].feature_names_in_) == ["peak_density", "range_trace"]
    assert bundle["spike"][1]["transform"] == "raw"
    assert calls == [
        {
            "repo_id": "example/gcamp-analysis-models",
            "revision": "v1.0.0",
            "allow_patterns": ["manifest.json", "roi/**", "spike/**"],
        }
    ]


def test_huggingface_bundle_rejects_checksum_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = _write_model_bundle(tmp_path)
    manifest["models"]["roi"]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda **_kwargs: str(tmp_path)
    )

    try:
        load_model_bundle(
            {
                "source": "huggingface",
                "repo_id": "example/gcamp-analysis-models",
                "revision": "abc123",
            }
        )
    except ValueError as exc:
        assert "SHA-256 mismatch for roi model" in str(exc)
    else:
        raise AssertionError("Expected a checksum mismatch")


def test_huggingface_bundle_requires_iteration_folders(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = _write_model_bundle(tmp_path)
    manifest["models"]["roi"]["model"] = "roi/model.joblib"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda **_kwargs: str(tmp_path)
    )

    with pytest.raises(ValueError, match="inside an iteration folder"):
        load_model_bundle(
            {
                "source": "huggingface",
                "repo_id": "example/gcamp-analysis-models",
                "revision": "v1.0.0",
            }
        )
