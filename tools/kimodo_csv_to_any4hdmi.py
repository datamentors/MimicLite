"""Convert kimodo G1 CSVs (kimodo/exports/mujoco.py save_csv: raw MuJoCo qpos per row =
root xyz, root quat wxyz, 29 joints in MuJoCo order, absolute angles) into one any4hdmi dataset.

usage: python kimodo_csv_to_any4hdmi.py --out-dir DIR [--fps 30] CSV [CSV ...]
Run with the sim2real venv. Names: kimodo_kimodo_g1_rp_v1_<name>.csv -> motions/<name>.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from mjhub import resolve_asset_reference
except ImportError:
    from mjhub import resolve_mjcf_reference as resolve_asset_reference

from any4hdmi.core.format import MOTION_DTYPE, MOTIONS_SUBDIR, save_motion, write_manifest
from any4hdmi.core.model import G1_JOINT_ORDER, base_qpos_adr, joint_qpos_adrs, load_model
from any4hdmi.utils.mjcf import (
    DEFAULT_MJCF_PATH,
    DEFAULT_MJCF_REPO_ID,
    DEFAULT_MJCF_REVISION,
    build_hf_mjcf_reference,
    qpos_names_from_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=30.0, help="kimodo G1 output rate.")
    args = parser.parse_args()

    mjcf_reference = build_hf_mjcf_reference(
        repo_id=DEFAULT_MJCF_REPO_ID, path=DEFAULT_MJCF_PATH, revision=DEFAULT_MJCF_REVISION
    )
    model = load_model(resolve_asset_reference(mjcf_reference))
    base_adr = base_qpos_adr(model)
    hinge_adrs = joint_qpos_adrs(model, G1_JOINT_ORDER)
    out_dir = Path(args.out_dir).expanduser().resolve()

    total_frames = 0
    for csv in args.csvs:
        csv_path = Path(csv).expanduser().resolve()
        rows = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
        rows = rows[None, :] if rows.ndim == 1 else rows
        if rows.shape[1] != 7 + len(G1_JOINT_ORDER):
            raise ValueError(f"{csv_path}: expected {7 + len(G1_JOINT_ORDER)} columns, got {rows.shape[1]}")
        quat = rows[:, 3:7]
        norms = np.linalg.norm(quat, axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-2):
            raise ValueError(f"{csv_path}: root quaternion not unit (norm {norms.min():.3f}..{norms.max():.3f})")

        qpos = np.zeros((len(rows), model.nq), dtype=MOTION_DTYPE)
        qpos[:, base_adr : base_adr + 3] = rows[:, 0:3]
        qpos[:, base_adr + 3 : base_adr + 7] = quat / norms[:, None]
        qpos[:, hinge_adrs] = rows[:, 7:]
        name = csv_path.stem.removeprefix("kimodo_kimodo_g1_rp_v1_")
        out = save_motion(out_dir / MOTIONS_SUBDIR / f"{name}.npz", qpos)
        total_frames += len(qpos)
        print(f"{name}: {len(qpos)} frames ({len(qpos) / args.fps:.1f} s) -> {out}")

    write_manifest(
        out_dir,
        dataset_name="kimodo",
        mjcf=mjcf_reference,
        timestep=1.0 / args.fps,
        qpos_names=qpos_names_from_model(model),
        num_motions=len(args.csvs),
        total_hours=total_frames / args.fps / 3600.0,
        source={"csvs": [str(Path(c).expanduser().resolve()) for c in args.csvs], "fps": args.fps,
                "root_representation": "xyz + qw qx qy qz", "joint_order": "mujoco"},
    )


if __name__ == "__main__":
    main()
