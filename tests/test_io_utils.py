from pathlib import Path

import numpy as np

from utils.io_utils import load_suite2p_data


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
