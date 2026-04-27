from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import subprocess
# EXAMPLE USAGE: python preprocessing/batch_s2p.py --fs 15 --pretrained_model invitro_rgcs_max
# ---------------------------------------------------------------------
# WHERE YOUR REAL GUI OPS LIVE (we'll read from here only)
# ---------------------------------------------------------------------
GUI_OPS_PATH = Path(r"C:\Users\mzinn1\.suite2p\ops\ops_user.npy")
GUI_DB_PATH  = Path(r"C:\Users\mzinn1\.suite2p\ops\db.npy")

# folder where we will write per-run temp ops/db
# (separate from .suite2p so we know EXACTLY what’s used)
TMP_DIR = Path(r"C:\Users\mzinn1\Desktop\s2p_temp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

ROOTS = [
r"G:\Calcium Imaging\GCaMP6s_EX328",
r"G:\Calcium Imaging\GCaMP6s_EX330",
r"G:\Calcium Imaging\GCaMP6s_EX329",
r"E:\EX345",
r"E:\EX344",
r"E:\GCaMP6s_EX357"
]

SKIP_IF_PROCESSED = False
FORCE_GPU = True

# ----- defaults (same as before) -----
GUI_DEFAULT_OPS = {
    "nplanes": 1,
    "nchannels": 1,
    "functional_chan": 1,
    "tau": 1.0,
    "fs": 15,
    "do_bidiphase": 0,
    "bidiphase": 0.0,
    "multiplane_parallel": 0,
    "ignore_flyback": -1,
    "aspect": 1.0,

    "preclassify": 0.0,
    "save_mat": 0,
    "save_NWB": 0.0,
    "combined": 1.0,
    "reg_tif": 0,
    "reg_tif_chan2": 0,
    "delete_bin": 1,
    "move_bin": 0,

    "do_registration": 1,
    "align_by_chan": 1,
    "nimg_init": 300,
    "batch_size": 400,
    "smooth_sigma": 4.0,
    "smooth_sigma_time": 0.0,
    "maxregshift": 0.1,
    "th_badframes": 1.0,
    "keep_movie_raw": 0,
    "two_step_registration": 0.0,

    "nonrigid": 1,
    "block_size": [128, 128],
    "snr_thresh": 1.2,
    "maxregshiftNR": 5.0,
    "spatial_hp_reg": 42.0,
    "pre_smooth": 1.0,
    "spatial_taper": 40.0,

    "roidetect": 1,
    "sparse_mode": 1,
    "denoise": 1.0,
    "spatial_scale": 4,
    "connected": 0,
    "threshold_scaling": 0.7,
    "max_overlap": 1.0,
    "max_iterations": 20,
    "high_pass": 5.0,
    "spatial_hp_detect": 30.0,

    "anatomical_only": 1,
    "cellprob_threshold": 0.0,
    "flow_threshold": 0.5,
    "pretrained_model": "invitro_rgcs_max",
    "spatial_hp_cp": 0.0,

    "neuropil_extract": 1,
    "allow_overlap": 0,
    "inner_neuropil_radius": 2,
    "min_neuropil_pixels": 350,

    "soma_crop": 1.0,
    "spikedetect": 1,
    "win_baseline": 60.0,
    "sig_baseline": 10.0,
    "neucoeff": 0.7,
}

def folder_has_tifs(folder: Path) -> bool:
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        if any(folder.glob(ext)):
            return True
    return False

def is_already_processed(folder: Path) -> bool:
    return (folder / "suite2p").exists()

def is_experiment_leaf(folder: Path) -> bool:
    return len(folder.parts) <= 6

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run Suite2p on folders of TIFFs."
    )
    parser.add_argument(
        "--fs", type=float, default=None,
        help="Sampling rate (Hz). Overrides the value in GUI ops.",
    )
    parser.add_argument(
        "--pretrained_model", type=str, default=None,
        help="Pretrained model name for cellpose. Overrides the value in GUI ops.",
    )
    return parser.parse_args()


def load_base_ops(cli_overrides: dict | None = None) -> dict:
    if GUI_OPS_PATH.exists():
        try:
            ops = np.load(GUI_OPS_PATH, allow_pickle=True).item()
        except Exception as e:
            print(f"!! failed to load GUI ops, using defaults: {e}")
            ops = GUI_DEFAULT_OPS.copy()
    else:
        ops = GUI_DEFAULT_OPS.copy()

    # Apply any CLI overrides
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                print(f"  CLI override: {key} = {value}")
                ops[key] = value

    return ops

def run_suite2p_on_folder(folder: Path, cli_overrides: dict | None = None):
    print(f"\n=== Running Suite2p on: {folder} ===")

    ops = load_base_ops(cli_overrides)
    ops["reg_tif"] = 0
    ops["delete_bin"] = 1
    # per-folder
    ops["save_path0"] = str(folder)
    ops["fast_disk"]  = str(folder)

    if FORCE_GPU:
        ops["useGPU"] = True
        ops["do_registration"] = 1
    np.save(GUI_OPS_PATH, ops)  # overwrite GUI ops to ensure consistency
    # WRITE TO TEMP DIR — NOT .suite2p
    ops_temp = TMP_DIR / "ops_temp.npy"
    np.save(ops_temp, ops)

    # also write a temp db
    db_temp = TMP_DIR / "db_temp.npy"
    db = {
        "data_path": [str(folder)],
        "subfolders": [],
        "look_one_level_down": False,
        "save_path0": str(folder),
        "fast_disk": str(folder),
        "input_format": "tif",
    }
    np.save(db_temp, db)

    # print to verify
    #print("→ using ops:")``
    #for k, v in ops.items():
    #    #print(f"   {k}: {v}")

    cmd = [
        "python", "-m", "suite2p",
        "--ops", str(ops_temp),
        "--db",  str(db_temp),
    ]
    subprocess.run(cmd, check=True)

def batch_s2p(roots, cli_overrides: dict | None = None, skip_if_processed: bool = SKIP_IF_PROCESSED):
    for root_str in roots:
        root = Path(root_str)
        if not root.exists():
            print(f"!! root not found: {root}")
            continue

        for folder in root.rglob("*"):
            if not folder.is_dir():
                continue
            if not folder_has_tifs(folder):
                continue
            if skip_if_processed and is_already_processed(folder):
                print(f"↳ Skipping (already processed): {folder}")
                continue
            try:
                run_suite2p_on_folder(folder, cli_overrides)
            except Exception as e:
                print(f"!! Error processing {folder}: {e}")
def main():
    args = parse_args()

    # Build override dict from CLI args
    cli_overrides = {
        "fs": args.fs,
        "pretrained_model": args.pretrained_model,
    }

    batch_s2p(ROOTS, cli_overrides=cli_overrides, skip_if_processed=SKIP_IF_PROCESSED)

    print("\n✅ done.")

if __name__ == "__main__":
    main()
