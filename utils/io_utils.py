"""
I/O utilities for loading experiment data.
"""
from __future__ import annotations
import hashlib
import json
import shutil
from datetime import datetime

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
from joblib import load
import yaml
if TYPE_CHECKING:
    from gcamp_analysis.data_classes import Video

def load_suite2p_data(suite2p_path: Path) -> dict:
    """
    Load the Suite2p inputs required by the analysis pipeline.

    Large numeric arrays are opened read-only with NumPy memory mapping so
    processing a video does not eagerly copy every Suite2p input into RAM.
    Suite2p ``ops.npy`` files may contain very large registration payloads
    (notably ``regPC``); the analysis pipeline only consumes ``fs``, ``Ly``,
    and ``Lx``, so only those fields are retained.

    Parameters
    ----------
    suite2p_path : Path
        Path to suite2p/plane0 directory

    Returns
    -------
    dict
        Dictionary containing Suite2p data arrays.
        Keys: F, iscell (required); Fneu, spks, stat, ops, fs (optional).
    """
    suite2p_path = Path(suite2p_path)
    data = {}

    required = ['F.npy', 'iscell.npy', ]
    for file in required:
        if not (suite2p_path / file).exists():
            raise FileNotFoundError(f"Required file {file} not found in {suite2p_path}")

    data['F'] = np.load(suite2p_path / 'F.npy', mmap_mode='r')
    data['iscell'] = np.load(suite2p_path / 'iscell.npy', mmap_mode='r')

    if (suite2p_path / 'Fneu.npy').exists():
        data['Fneu'] = np.load(suite2p_path / 'Fneu.npy', mmap_mode='r')
    else:
        data['Fneu'] = np.zeros_like(data['F'])

    if (suite2p_path / 'spks.npy').exists():
        data['spks'] = np.load(suite2p_path / 'spks.npy', mmap_mode='r')

    if (suite2p_path / 'stat.npy').exists():
        data['stat'] = np.load(suite2p_path / 'stat.npy', allow_pickle=True)

    if (suite2p_path / 'ops.npy').exists():
        full_ops = np.load(
            suite2p_path / 'ops.npy',
            allow_pickle=True,
        ).item()
        data['ops'] = {
            key: full_ops[key]
            for key in ('fs', 'Ly', 'Lx')
            if key in full_ops
        }
        del full_ops
        data['fs'] = data['ops'].get('fs', 15.0)
    else:
        data['ops'] = {}
        data['fs'] = 15.0

    return data
    
def load_config(config_path: Path = Path("config.yaml")) -> Dict:
    """Load configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def _require_file(path: Path, description: str) -> Path:
    """Return *path* when it is a file, otherwise raise a useful error."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} cannot be found at {path}.")
    return path


def _load_json(path: Path, description: str) -> Dict:
    """Load a required JSON object."""
    _require_file(path, description)
    try:
        with path.open("r", encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} at {path} must contain a JSON object.")
    return value


def _path_in_snapshot(snapshot: Path, relative_path: str, description: str) -> Path:
    """Resolve a manifest path without allowing it to escape the snapshot."""
    candidate = (snapshot / relative_path).resolve()
    try:
        candidate.relative_to(snapshot.resolve())
    except ValueError as exc:
        raise ValueError(f"{description} path escapes the downloaded snapshot: {relative_path}") from exc
    return _require_file(candidate, description)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_local_model_paths(models_config: Dict) -> Dict[str, Dict[str, Path | None]]:
    paths: Dict[str, Dict[str, Path | None]] = {}
    for which in ("roi", "spike"):
        model_path = Path(models_config.get(f"{which}_model_path", ""))
        _require_file(model_path, f"{which} model")
        config_value = models_config.get(f"{which}_config_path")
        config_path = Path(config_value) if config_value else None
        if config_path is not None:
            _require_file(config_path, f"{which} model config")
        paths[which] = {"model": model_path, "config": config_path}
    return paths


def _resolve_huggingface_model_paths(
    models_config: Dict,
) -> tuple[Dict[str, Dict[str, Path]], Dict]:
    """Download one pinned Hub snapshot and resolve both model folders."""
    repo_id = models_config.get("repo_id")
    revision = models_config.get("revision")
    if not repo_id or not revision:
        raise ValueError(
            "Hugging Face model configuration requires both 'repo_id' and a "
            "pinned 'revision'."
        )
    if revision in {"main", "master"}:
        raise ValueError(
            "Hugging Face model revision must be a release tag or commit hash, "
            "not a mutable default branch."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face model loading requires the 'huggingface_hub' package."
        ) from exc

    print(f"Resolving model bundle {repo_id}@{revision} (downloads are cached)...")
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=["manifest.json", "roi/**", "spike/**"],
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not obtain pinned model bundle {repo_id}@{revision}. If this "
            "bundle is not already cached, connect to the network or configure "
            "models.source: local with explicit ROI and spike paths."
        ) from exc

    manifest = _load_json(snapshot / "manifest.json", "model bundle manifest")
    manifest_models = manifest.get("models")
    if not isinstance(manifest_models, dict):
        raise ValueError("Model bundle manifest is missing the 'models' object.")
    expected_version = manifest.get("scikit_learn_version")
    if not expected_version:
        raise ValueError("Model bundle manifest is missing 'scikit_learn_version'.")
    from sklearn import __version__ as sklearn_version
    if sklearn_version != expected_version:
        raise ValueError(
            f"scikit-learn version mismatch: bundle requires {expected_version}, "
            f"but the application is using {sklearn_version}."
        )

    paths: Dict[str, Dict[str, Path]] = {}
    for which in ("roi", "spike"):
        entry = manifest_models.get(which)
        if not isinstance(entry, dict):
            raise ValueError(f"Model bundle manifest is missing models.{which}.")
        for key in ("model", "config", "sha256", "features", "transform"):
            if key not in entry:
                raise ValueError(f"Model bundle manifest is missing models.{which}.{key}.")
        model_relative = Path(entry["model"])
        config_relative = Path(entry["config"])
        for key, relative_path in (
            ("model", model_relative),
            ("config", config_relative),
        ):
            if len(relative_path.parts) < 3 or relative_path.parts[0] != which:
                raise ValueError(
                    f"models.{which}.{key} must point inside an iteration folder "
                    f"under {which}/ (for example, "
                    f"{which}/iteration-001/model.joblib)."
                )
        if model_relative.parent != config_relative.parent:
            raise ValueError(
                f"The {which} model and results must be in the same iteration folder."
            )
        model_path = _path_in_snapshot(snapshot, entry["model"], f"{which} model")
        config_path = _path_in_snapshot(snapshot, entry["config"], f"{which} model config")
        actual_hash = _sha256(model_path)
        if actual_hash != entry["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {which} model: expected {entry['sha256']}, "
                f"got {actual_hash}."
            )

        paths[which] = {"model": model_path, "config": config_path}
    return paths, manifest


def _validate_huggingface_model(
    which: str,
    model: Any,
    model_cfg: Dict,
    manifest: Dict,
) -> None:
    """Cross-check a deserialized model and sidecar against the manifest."""
    from utils.inference import get_model_feature_names

    entry = manifest["models"][which]
    expected_features = list(entry["features"])
    sidecar_features = model_cfg.get("features")
    model_features = get_model_feature_names(model)
    if sidecar_features != expected_features:
        raise ValueError(
            f"Feature mismatch for {which} config: expected {expected_features}, "
            f"got {sidecar_features}."
        )
    if model_features != expected_features:
        raise ValueError(
            f"Feature mismatch for {which} model: expected {expected_features}, "
            f"got {model_features}."
        )
    if model_cfg.get("transform") != entry["transform"]:
        raise ValueError(
            f"Transform mismatch for {which}: expected {entry['transform']!r}, "
            f"got {model_cfg.get('transform')!r}."
        )


def load_model_bundle(models_config: Dict) -> Dict[str, tuple[Any, Optional[Dict]]]:
    """Resolve and load the ROI and spike models from local paths or one Hub repo.

    Hub repositories use ``roi/`` and ``spike/`` folders described by a root
    ``manifest.json``. The snapshot is requested once, so both classifiers come
    from the same pinned release and share the standard Hugging Face cache.
    """
    source = models_config.get("source", "local")
    manifest: Optional[Dict] = None
    if source == "local":
        paths = _resolve_local_model_paths(models_config)
    elif source == "huggingface":
        paths, manifest = _resolve_huggingface_model_paths(models_config)
    else:
        raise ValueError("models.source must be either 'local' or 'huggingface'.")

    loaded: Dict[str, tuple[Any, Optional[Dict]]] = {}
    for which in ("roi", "spike"):
        config_path = paths[which]["config"]
        model_cfg = (
            _load_json(config_path, f"{which} model config")
            if config_path is not None
            else None
        )
        if manifest is not None:
            assert model_cfg is not None
            entry = manifest["models"][which]
            if model_cfg.get("features") != entry["features"]:
                raise ValueError(
                    f"Feature mismatch for {which} config: expected "
                    f"{entry['features']}, got {model_cfg.get('features')}."
                )
            if model_cfg.get("transform") != entry["transform"]:
                raise ValueError(
                    f"Transform mismatch for {which}: expected "
                    f"{entry['transform']!r}, got {model_cfg.get('transform')!r}."
                )
        model = load(paths[which]["model"])
        if manifest is not None:
            # Hub sidecars and model metadata are required and validated before inference.
            _validate_huggingface_model(which, model, model_cfg, manifest)
        loaded[which] = (model, model_cfg)
    return loaded


def load_model(models_config: Dict, which: str) -> tuple:
    """Load a model and its JSON config sidecar.

    Parameters
    ----------
    models_config : dict
        The ``models`` section of the pipeline YAML.  Expected keys:
        ``<which>_model_path`` and (optionally) ``<which>_config_path``.
    which : str
        ``"roi"`` or ``"spike"``.

    Returns
    -------
    tuple[Any, dict | None]
        ``(sklearn_model, config_dict)`` where *config_dict* is ``None``
        when no config path is given or the file does not exist.
    """
    if which not in {"roi", "spike"}:
        raise ValueError("which must be either 'roi' or 'spike'.")
    if models_config.get("source", "local") == "local":
        model_path = Path(models_config.get(f"{which}_model_path", ""))
        _require_file(model_path, f"{which} model")
        model = load(model_path)

        config_value = models_config.get(f"{which}_config_path")
        if not config_value or not Path(config_value).is_file():
            return model, None
        return model, _load_json(Path(config_value), f"{which} model config")
    return load_model_bundle(models_config)[which]


def create_backup(input_path: Path) -> Path:
    """
    Create timestamped backup of a file.

    Parameters
    ----------
    input_path : Path
        Path to file to backup

    Returns
    -------
    backup_path : Path
        Path to created backup
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = input_path.with_suffix(f'.backup_{timestamp}.npy')
    shutil.copy(input_path, backup_path)
    return backup_path
