from pathlib import Path

import numpy as np
import pytest
import tifffile

from preprocessing import batch_s2p


def test_is_already_processed_requires_suite2p_outputs(tmp_path: Path):
    plane0 = tmp_path / "suite2p" / "plane0"
    plane0.mkdir(parents=True)

    assert not batch_s2p.is_already_processed(tmp_path)

    for filename in ("F.npy", "iscell.npy", "ops.npy"):
        np.save(plane0 / filename, np.array([]))

    assert batch_s2p.is_already_processed(tmp_path)


def test_remove_incomplete_cache_removes_reusable_binary_state(tmp_path: Path):
    plane0 = tmp_path / "suite2p" / "plane0"
    plane0.mkdir(parents=True)
    cached_paths = [
        plane0 / "data.bin",
        plane0 / "data_raw.bin",
        plane0 / "ops.npy",
    ]
    for path in cached_paths:
        path.write_bytes(b"stale")

    removed = batch_s2p.remove_incomplete_cache(tmp_path)

    assert removed == cached_paths
    assert not any(path.exists() for path in cached_paths)


def test_remove_incomplete_cache_preserves_completed_output(tmp_path: Path):
    plane0 = tmp_path / "suite2p" / "plane0"
    plane0.mkdir(parents=True)
    data_bin = plane0 / "data.bin"
    data_bin.write_bytes(b"valid")
    for filename in ("F.npy", "iscell.npy", "ops.npy"):
        np.save(plane0 / filename, np.array([]))

    assert batch_s2p.remove_incomplete_cache(tmp_path) == []
    assert data_bin.exists()


def test_validate_tiff_accepts_readable_image(tmp_path: Path):
    path = tmp_path / "valid.tiff"
    tifffile.imwrite(
        path,
        np.zeros((4, 8, 8), dtype=np.uint16),
        photometric="minisblack",
    )

    batch_s2p.validate_tiff(path)


def test_validate_tiff_rejects_header_only_bigtiff(tmp_path: Path):
    path = tmp_path / "invalid.tiff"
    path.write_bytes(b"II+\x00\x08\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00")

    with pytest.raises(ValueError, match="only 16 bytes"):
        batch_s2p.validate_tiff(path)
