"""Offline check of the PICO -> A2 GMR config without a headset.

Synthesizes PICO/XRoboToolkit-convention human frames from a G1 motion by inverting GMR's own
xrobot_to_g1 mapping (q_human = q_G1body * off^-1, positions un-scaled about the pelvis), retargets them to
A2 with the active T-pose calibration (a2_sim2real.GMR_CONFIG), and compares A2 against G1 frame by frame. --out DIR also writes the A2 motion
as an any4hdmi dataset clip (DIR/manifest.json + DIR/motions/<clip>_a2.npz, 50 fps) that sim2real's
integrated_sim2sim / npz_pub play like any A2 training clip.

Run in the sim2real teleop venv (from the MimicLite checkout that holds sim2real/ and any4hdmi/):
  sim2real/venv/teleop/.venv/bin/python <a2>/teleop/tools/check_retarget.py \
      --motion any4hdmi/output/g1/lafan/motions/walk1_subject5.npz --manifest any4hdmi/output/g1/lafan/manifest.json \
      --out <a2>/teleop/out/lafan_a2 [--start 0 --frames 600]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817 - conventional rotation notation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a2_sim2real  # noqa: E402

a2_sim2real.register()
from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402, N817 - project-wide GMR name
from general_motion_retargeting.params import IK_CONFIG_DICT  # noqa: E402
from general_motion_retargeting.params import ROBOT_XML_DICT  # noqa: E402

# human joint -> (G1 body it is read back from, G1 config entry whose offset/scale inverts it).
# Ankle: G1 ankle_roll_link with Left_Foot's offset. Foot (toes) is not used by the A2 config but is the
# lowest joint, which pico_retarget_pub aligns to the floor. Spine3 (chest) drives A2's torso orientation.
G1_SOURCE = {"Pelvis": ("pelvis", "Pelvis"), "Spine3": ("torso_link", "Spine3")}
for S, s in (("Left", "left"), ("Right", "right")):
    G1_SOURCE.update(
        {
            f"{S}_Hip": (f"{s}_hip_yaw_link", f"{S}_Hip"),
            f"{S}_Knee": (f"{s}_knee_link", f"{S}_Knee"),
            f"{S}_Ankle": (f"{s}_ankle_roll_link", f"{S}_Foot"),
            f"{S}_Foot": (f"{s}_toe_link", f"{S}_Foot"),
            f"{S}_Shoulder": (f"{s}_shoulder_yaw_link", f"{S}_Shoulder"),
            f"{S}_Elbow": (f"{s}_elbow_link", f"{S}_Elbow"),
            f"{S}_Wrist": (f"{s}_wrist_yaw_link", f"{S}_Wrist"),
        }
    )
A2_SOLE_BELOW_ANKLE = 0.059  # toe_roll origin to foot contact spheres (a2_serial.xml)
# (G1 joint, A2 joint, sign) pairs compared as angle traces
JOINT_PAIRS = [
    ("left_knee_joint", "idx04_left_tarsus", 1.0),
    ("right_knee_joint", "idx10_right_tarsus", 1.0),
    ("left_hip_pitch_joint", "idx03_left_hip_pitch", 1.0),
    ("right_hip_pitch_joint", "idx09_right_hip_pitch", 1.0),
]
# (G1 from, G1 to, A2 from, A2 to) segment directions compared as unit vectors
SEGMENTS = [
    ("left_shoulder_pitch_link", "left_elbow_link", "left-arm-link2", "left-arm-link4"),
    ("left_elbow_link", "left_wrist_yaw_link", "left-arm-link4", "left-arm-link7"),
    ("right_shoulder_pitch_link", "right_elbow_link", "right-arm-link2", "right-arm-link4"),
    ("right_elbow_link", "right_wrist_yaw_link", "right-arm-link4", "right-arm-link7"),
    ("left_knee_link", "left_ankle_roll_link", "left_tarsus", "left_toe_roll"),
    ("right_knee_link", "right_ankle_roll_link", "right_tarsus", "right_toe_roll"),
]


def qadr(model: mujoco.MjModel, joint: str) -> int:
    """Return the qpos address of a named joint.

    Args:
        model: MuJoCo model.
        joint: joint name.

    Returns:
        int: index of the joint's first qpos entry.

    Raises:
        KeyError: if the joint is not in the model.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jid < 0:
        raise KeyError(f"joint {joint} not in model")
    return int(model.jnt_qposadr[jid])


def bid(model: mujoco.MjModel, name: str) -> int:
    """Return the id of a named body.

    Args:
        model: MuJoCo model.
        name: body name.

    Returns:
        int: body id.

    Raises:
        KeyError: if the body is not in the model.
    """
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise KeyError(f"body {name} not in model")
    return i


class G1ToHuman:
    """Convert G1 motion frames to PICO-convention human joint poses.

    Output is in a right-handed z-up world with quaternions wxyz, obtained by inverting GMR's xrobot_to_g1
    offsets and pelvis-relative scales.
    """

    def __init__(self, manifest_path: Path):
        """Load the G1 model and GMR's xrobot -> unitree_g1 config.

        Args:
            manifest_path: any4hdmi manifest.json whose qpos_names (after the 7 root entries) give the joint order.
        """
        names = json.loads(Path(manifest_path).read_text())["qpos_names"][7:]
        cfg = json.loads(Path(IK_CONFIG_DICT["xrobot"]["unitree_g1"]).read_text())
        self.off = {e[0]: R.from_quat(e[4], scalar_first=True) for e in cfg["ik_match_table1"].values()}
        self.scale = cfg["human_scale_table"]
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
        self.data = mujoco.MjData(self.model)
        self.adr = [qadr(self.model, n) for n in names]
        self.body = {joint: bid(self.model, g1_body) for joint, (g1_body, _) in G1_SOURCE.items()}

    def __call__(self, frame: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Run G1 forward kinematics for one frame and return the human joint poses.

        Also leaves the G1 FK result in self.data.

        Args:
            frame: G1 qpos (root xyz + quat wxyz + joints in manifest order).

        Returns:
            dict: human joint name -> (position, quaternion wxyz).
        """
        m, d = self.model, self.data
        d.qpos[:7] = frame[:7]
        d.qpos[self.adr] = frame[7:]
        mujoco.mj_forward(m, d)
        root_h = d.xpos[self.body["Pelvis"]] / self.scale["Pelvis"]
        human = {}
        for joint, (_, key) in G1_SOURCE.items():
            b = self.body[joint]
            quat = (R.from_quat(d.xquat[b], scalar_first=True) * self.off[key].inv()).as_quat(scalar_first=True)
            pos = (
                root_h if joint == "Pelvis" else (d.xpos[b] - self.scale["Pelvis"] * root_h) / self.scale[key] + root_h
            )
            human[joint] = (pos.astype(np.float64), quat)
        return human


def main() -> None:
    """Retarget a G1 clip to A2 through synthetic PICO frames, print comparison metrics, optionally save."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--motion", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    qpos_g1 = np.load(args.motion)["qpos"][args.start : args.start + args.frames]
    if len(qpos_g1) == 0:
        raise SystemExit("no frames in the selected range")

    to_human = G1ToHuman(args.manifest)
    g1, gd = to_human.model, to_human.data

    retarget = GMR(src_human="xrobot", tgt_robot=a2_sim2real.GMR_ROBOT, actual_human_height=1.8)
    a2 = retarget.configuration.model
    ad = mujoco.MjData(a2)
    jlo, jhi = a2.jnt_range[1:, 0], a2.jnt_range[1:, 1]

    a2_qpos, g1_tr, a2_tr, cos, sole = [], [], [], [], []
    for frame in qpos_g1:
        human = to_human(frame)  # also runs G1 FK into gd
        retarget.retarget(human)
        q = retarget.configuration.data.qpos.copy()
        a2_qpos.append(q)
        ad.qpos[:] = q
        mujoco.mj_forward(a2, ad)
        g1_tr.append([gd.qpos[qadr(g1, gj)] for gj, _, _ in JOINT_PAIRS])
        a2_tr.append([s * ad.qpos[qadr(a2, aj)] for _, aj, s in JOINT_PAIRS])
        row = []
        for g0, g1b, a0, a1 in SEGMENTS:
            vg = gd.xpos[bid(g1, g1b)] - gd.xpos[bid(g1, g0)]
            va = ad.xpos[bid(a2, a1)] - ad.xpos[bid(a2, a0)]
            row.append(float(vg @ va / (np.linalg.norm(vg) * np.linalg.norm(va) + 1e-9)))
        cos.append(row)
        sole.append(
            min(ad.xpos[bid(a2, "left_toe_roll")][2], ad.xpos[bid(a2, "right_toe_roll")][2]) - A2_SOLE_BELOW_ANKLE
        )

    a2_qpos = np.asarray(a2_qpos, dtype=np.float32)
    g1_tr, a2_tr, cos, sole = map(np.asarray, (g1_tr, a2_tr, cos, sole))
    joints = a2_qpos[:, 7:]
    viol = np.maximum(jlo - joints, joints - jhi).max()
    print(f"frames {len(a2_qpos)}  max joint-limit violation {max(viol, 0.0):.4f} rad")
    for k, (gj, aj, _) in enumerate(JOINT_PAIRS):
        corr = np.corrcoef(g1_tr[:, k], a2_tr[:, k])[0, 1]
        mad = np.abs(g1_tr[:, k] - a2_tr[:, k]).mean()
        print(f"{gj:>22} vs {aj:<22} corr {corr:+.3f}  mean|diff| {mad:.3f} rad")
    for k, (g0, g1b, _, _) in enumerate(SEGMENTS):
        print(f"segment {g0}->{g1b}: direction cos mean {cos[:, k].mean():+.3f}  min {cos[:, k].min():+.3f}")
    print(
        f"A2 torso z mean {a2_qpos[:, 2].mean():.3f} m;  lowest sole z min {sole.min():+.3f} / "
        f"mean of per-frame lowest {sole.mean():+.3f} m"
    )
    if args.out is not None:
        clip = write_dataset_clip(args.out, args.motion.stem + "_a2", a2_qpos, float(manifest["timestep"]))
        print(f"wrote {clip} ({1.0 / OUT_TIMESTEP:.0f} fps, {args.out}/manifest.json)")


OUT_TIMESTEP = 0.02  # 50 fps, same as the A2 training datasets


def write_dataset_clip(out_dir: Path, name: str, qpos: np.ndarray, src_timestep: float) -> Path:
    """Write qpos as an any4hdmi dataset clip resampled to 50 fps.

    Writes manifest + motions/<name>.npz so sim2real's integrated_sim2sim / npz_pub can play it like the
    A2 training clips.

    Args:
        out_dir: dataset directory (created if missing).
        name: clip name (file stem).
        qpos: (T, D) A2 qpos at src_timestep.
        src_timestep: input frame period [s].

    Returns:
        Path: the written clip npz.
    """
    t_src = np.arange(len(qpos)) * src_timestep
    t_out = np.arange(0.0, t_src[-1] + 1e-9, OUT_TIMESTEP)
    out = np.empty((len(t_out), qpos.shape[1]), dtype=np.float32)
    for k in list(range(3)) + list(range(7, qpos.shape[1])):
        out[:, k] = np.interp(t_out, t_src, qpos[:, k])
    if len(qpos) > 1:
        from scipy.spatial.transform import Slerp  # noqa: PLC0415 - only needed when there is more than one frame

        rots = R.from_quat(qpos[:, 3:7], scalar_first=True)
        out[:, 3:7] = Slerp(t_src, rots)(t_out).as_quat(scalar_first=True)
    else:
        out[:, 3:7] = qpos[:, 3:7]
    (out_dir / "motions").mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 2,
        "dataset_name": out_dir.name,
        "mjcf": str(a2_sim2real.A2_SRC_MJCF.with_name("a2_kin.xml")),
        "motions_subdir": "motions",
        "timestep": OUT_TIMESTEP,
        "qpos_dim": int(out.shape[1]),
        "qpos_names": [
            "root_tx",
            "root_ty",
            "root_tz",
            "root_qw",
            "root_qx",
            "root_qy",
            "root_qz",
            *a2_sim2real.JOINT_NAMES,
        ],
        "source": {"tool": "a2/teleop/tools/check_retarget.py", "src_timestep": src_timestep},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    clip = out_dir / "motions" / f"{name}.npz"
    np.savez(clip, qpos=out)
    return clip


if __name__ == "__main__":
    main()
