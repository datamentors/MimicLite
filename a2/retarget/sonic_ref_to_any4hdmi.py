"""Convert gear_sonic_deploy reference folders (joint_pos.csv, body_pos.csv, body_quat.csv @ 50 Hz,
IsaacLab joint order, body 0 = pelvis, quat wxyz) into one any4hdmi dataset.

usage: python sonic_ref_to_any4hdmi.py --out-dir DIR REF_DIR [REF_DIR ...]
Run with the sim2real venv. Mirrors any4hdmi/scripts/preprocess/lafan.py.
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

# IsaacLab / SONIC joint order (same as MimicLite policy_joint_names).
ISAACLAB_JOINT_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
SONIC_FPS = 50.0


def _load(ref_dir: Path, name: str) -> np.ndarray:
    path = ref_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    return data[None, :] if data.ndim == 1 else data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("refs", nargs="+")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    mjcf_reference = build_hf_mjcf_reference(
        repo_id=DEFAULT_MJCF_REPO_ID, path=DEFAULT_MJCF_PATH, revision=DEFAULT_MJCF_REVISION
    )
    model = load_model(resolve_asset_reference(mjcf_reference))
    base_adr = base_qpos_adr(model)
    hinge_adrs = joint_qpos_adrs(model, G1_JOINT_ORDER)
    reorder = [ISAACLAB_JOINT_ORDER.index(n) for n in G1_JOINT_ORDER]
    out_dir = Path(args.out_dir).expanduser().resolve()

    total_frames = 0
    for ref in args.refs:
        ref_dir = Path(ref).expanduser().resolve()
        joint_pos = _load(ref_dir, "joint_pos.csv")
        body_pos = _load(ref_dir, "body_pos.csv")
        body_quat = _load(ref_dir, "body_quat.csv")
        if joint_pos.shape[1] != 29 or not (len(joint_pos) == len(body_pos) == len(body_quat)):
            raise ValueError(
                f"{ref_dir}: unexpected shapes joint_pos={joint_pos.shape} "
                f"body_pos={body_pos.shape} body_quat={body_quat.shape}"
            )
        quat = body_quat[:, 0:4]
        quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

        qpos = np.zeros((len(joint_pos), model.nq), dtype=MOTION_DTYPE)
        qpos[:, base_adr : base_adr + 3] = body_pos[:, 0:3]
        qpos[:, base_adr + 3 : base_adr + 7] = quat
        qpos[:, hinge_adrs] = joint_pos[:, reorder]
        out = save_motion(out_dir / MOTIONS_SUBDIR / f"{ref_dir.name}.npz", qpos)
        total_frames += len(qpos)
        print(f"{ref_dir.name}: {len(qpos)} frames ({len(qpos) / SONIC_FPS:.1f} s) -> {out}")

    write_manifest(
        out_dir,
        dataset_name="sonic_ref",
        mjcf=mjcf_reference,
        timestep=1.0 / SONIC_FPS,
        qpos_names=qpos_names_from_model(model),
        num_motions=len(args.refs),
        total_hours=total_frames / SONIC_FPS / 3600.0,
        source={"refs": [str(Path(r).expanduser().resolve()) for r in args.refs], "fps": SONIC_FPS,
                "root_representation": "body 0 (pelvis) pos + quat wxyz", "joint_order": "isaaclab"},
    )


if __name__ == "__main__":
    main()
