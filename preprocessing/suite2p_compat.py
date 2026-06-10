from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import tifffile


class ContiguousImageJStack:
    def __init__(self, path: str):
        self.array = tifffile.memmap(path, mode="r")


def patch_suite2p_tiff_reader() -> None:
    from suite2p.io import tiff as suite2p_tiff

    original_open_tiff = suite2p_tiff.open_tiff
    original_read_tiff = suite2p_tiff.read_tiff

    def open_tiff(path: str, sktiff: bool):
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            is_contiguous_imagej_stack = (
                len(tif.pages) == 1
                and len(series.shape) == 3
                and series.axes.endswith("YX")
                and series.shape[0] > 1
                and tif.pages[0].is_contiguous
            )

        if is_contiguous_imagej_stack:
            stack = ContiguousImageJStack(path)
            print(
                "Reading contiguous ImageJ TIFF series "
                f"as {stack.array.shape[0]} frames"
            )
            return stack, stack.array.shape[0]

        return original_open_tiff(path, sktiff)

    def read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff):
        if not isinstance(tif, ContiguousImageJStack):
            return original_read_tiff(
                file, tif, Ltif, ix, batch_size, use_sktiff
            )

        if ix >= Ltif:
            return None

        nfr = min(Ltif - ix, batch_size)
        images = np.asarray(tif.array[ix:ix + nfr])
        if images.dtype.type in (np.uint16, np.int32):
            images = (images // 2).astype(np.int16)
        elif images.dtype.type != np.int16:
            images = images.astype(np.int16)
        return images

    suite2p_tiff.open_tiff = open_tiff
    suite2p_tiff.read_tiff = read_tiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", required=True)
    parser.add_argument("--db", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ops = np.load(Path(args.ops), allow_pickle=True).item()
    db = np.load(Path(args.db), allow_pickle=True).item()

    patch_suite2p_tiff_reader()

    from suite2p import run_s2p

    run_s2p(ops, db)


if __name__ == "__main__":
    main()
