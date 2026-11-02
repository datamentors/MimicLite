"""Per-operator T-pose calibration -> GMR IK config (PICO/XRoboToolkit body -> AgiBot A2). This is the only
way the A2 teleop GMR config is made; `teleop.sh calibrate <height>` records the T-pose and runs it.

Run in the sim2real teleop venv (has GMR + mujoco + scipy):
  sim2real/venv/teleop/.venv/bin/python a2/teleop/tools/make_xrobot_to_a2.py --tpose REC.npz --out CFG.json \
      [--height 1.75]

Derivation
----------
GMR sets each robot body target to  q_target = q_human * rot_offset  (wxyz, scipy, offsets keyed by human
joint; table2 reuses table1's offsets). A2's zero pose is a T-pose, so with the operator's recorded T-pose
(preprocessed like pico_retarget_pub, straightened to ideal segment directions) every rotation offset is
  off = q_human(T)^-1 * q_A2(zero)
and every position offset puts the scaled human joint exactly on the A2 zero-pose body. Scales are the
operator's measured hip->ankle and shoulder->wrist lengths against A2's.

Mapping (A2 has no waist: torso is the root, placed by the pelvis; its orientation follows the chest,
Spine3 -> head_yaw, welded to the torso):
  torso <- Pelvis, *_hip_pitch <- *_Hip, *_tarsus (knee) <- *_Knee, *_toe_roll <- *_Ankle,
  *-arm-link3 <- *_Shoulder, *-arm-link4 <- *_Elbow, *-arm-link7 <- *_Wrist, head_yaw <- Spine3
IK weights (only) are copied from the analog entry of GMR's PICO-tuned xrobot_to_g1.json; A2's ankle uses
G1's Left_Foot weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from a2_sim2real import ensure_sim_mjcf  # noqa: E402

from general_motion_retargeting.params import IK_CONFIG_DICT  # noqa: E402

A2_SOLE_BELOW_ANKLE = 0.059  # toe_roll origin to foot contact spheres (a2_serial.xml)
# (A2 body, human joint, xrobot_to_g1 human joint whose IK weights are reused)
MAPPING = [("torso", "Pelvis", "Pelvis")]
for side, S in (("left", "Left"), ("right", "Right")):
    MAPPING += [
        (f"{side}_hip_pitch", f"{S}_Hip", f"{S}_Hip"),
        (f"{side}_tarsus", f"{S}_Knee", f"{S}_Knee"),
        (f"{side}_toe_roll", f"{S}_Ankle", f"{S}_Foot"),
        (f"{side}-arm-link3", f"{S}_Shoulder", f"{S}_Shoulder"),
        (f"{side}-arm-link4", f"{S}_Elbow", f"{S}_Elbow"),
        (f"{side}-arm-link7", f"{S}_Wrist", f"{S}_Wrist"),
    ]


def forward(model: mujoco.MjModel, qpos: dict[str, float]) -> mujoco.MjData:
    data = mujoco.MjData(model)
    for name, value in qpos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint {name} not in model")
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    return data


def body(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[np.ndarray, R]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"body {name} not in model")
    return data.xpos[bid].copy(), R.from_quat(data.xquat[bid], scalar_first=True)


PICO_JOINTS = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]
# T-pose straightening: (joint, child joint, ideal world direction of joint->child); A2 zero pose is a T-pose
# (legs straight down, arms straight out along +-y) facing +x
TPOSE_SEGMENTS = [("Pelvis", "Spine1", (0, 0, 1))]
for _S, _y in (("Left", 1), ("Right", -1)):
    TPOSE_SEGMENTS += [
        (f"{_S}_Hip", f"{_S}_Knee", (0, 0, -1)), (f"{_S}_Knee", f"{_S}_Ankle", (0, 0, -1)),
        (f"{_S}_Shoulder", f"{_S}_Elbow", (0, _y, 0)), (f"{_S}_Elbow", f"{_S}_Wrist", (0, _y, 0)),
        (f"{_S}_Wrist", f"{_S}_Hand", (0, _y, 0)),
    ]


def swing(d_from: np.ndarray, d_to: np.ndarray) -> R:
    """Smallest rotation taking direction d_from onto d_to."""
    a, b = d_from / np.linalg.norm(d_from), np.asarray(d_to, float) / np.linalg.norm(d_to)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return R.identity()
    return R.from_rotvec(axis / s * np.arctan2(s, float(a @ b)))


def tpose_human(rec_path: Path, min_link_height: float) -> dict[str, tuple[np.ndarray, R]]:
    """Mean T-pose from a raw PICO recording, preprocessed exactly like pico_retarget_pub does before GMR
    (Unity -> z-up, pelvis heading removed, lowest joint at min_link_height), then straightened to an ideal
    T-pose (segment directions only; measured segment lengths and twists kept)."""
    import general_motion_retargeting as gmr
    from sim2real.teleop import pico_retarget_pub as prp

    rec = np.load(rec_path)
    poses = rec["poses"]  # (T, 24, 7) Unity x y z qx qy qz qw
    if poses.ndim != 3 or poses.shape[1:] != (24, 7):
        raise ValueError(f"{rec_path}: expected poses (T, 24, 7), got {poses.shape}")
    # steadiest half of the recording
    speed = np.linalg.norm(np.diff(poses[:, :, :3], axis=0), axis=2).max(1)
    win = max(len(poses) // 2, 1)
    start = int(np.argmin(np.convolve(speed, np.ones(win), "valid"))) if len(speed) >= win else 0
    frames = []
    for raw in poses[start: start + win]:
        d = {n: [raw[i, :3].astype(np.float64), np.array([raw[i, 6], raw[i, 3], raw[i, 4], raw[i, 5]])]
             for i, n in enumerate(PICO_JOINTS)}
        d = gmr.XRobotStreamer.coordinate_transform_unity_data(None, d)
        d = {k: [np.asarray(v[0], np.float32), np.asarray(v[1], np.float32)] for k, v in d.items()}
        src_pos, src_yaw = prp._pelvis_pose_no_z_yaw_from_body_pose_dict(d)
        d = prp._transform_body_pose_dict(
            d, src_pelvis_pos_no_z=src_pos, src_pelvis_quat_yaw=src_yaw,
            dst_pelvis_pos_no_z=np.zeros(3, np.float32), dst_pelvis_quat_yaw=np.array([1, 0, 0, 0], np.float32))
        frames.append(d)
    pos = {n: np.mean([f[n][0] for f in frames], axis=0).astype(np.float64) for n in PICO_JOINTS}
    rot = {n: R.from_quat(np.array([f[n][1] for f in frames]), scalar_first=True).mean() for n in PICO_JOINTS}

    # operator heading in GMR's input frame: the publisher removes the pelvis frame's yaw, which is not the
    # operator's facing direction (measured: operator faces +y). A2's T-pose is turned to the same heading.
    left = pos["Left_Hip"] - pos["Right_Hip"]
    left[2] = 0.0
    if np.linalg.norm(left) < 0.05:
        raise RuntimeError(f"hip width {np.linalg.norm(left):.3f} m too small to find the operator's heading")
    facing = np.cross(left / np.linalg.norm(left), [0.0, 0.0, 1.0])
    heading = R.from_rotvec([0.0, 0.0, np.arctan2(facing[1], facing[0])])

    # straighten: rebuild chains with ideal directions, rotate each joint frame by the same swing
    ideal_pos = dict(pos)
    for joint, child, direction in TPOSE_SEGMENTS:
        seg = pos[child] - pos[joint]
        ideal_dir = heading.apply(np.asarray(direction, float))
        rot[joint] = swing(seg, ideal_dir) * rot[joint]
        if joint != "Pelvis":
            ideal_pos[child] = ideal_pos[joint] + np.linalg.norm(seg) * ideal_dir
    for S in ("Left", "Right"):  # feet follow their ankle
        ideal_pos[f"{S}_Foot"] = ideal_pos[f"{S}_Ankle"] + (pos[f"{S}_Foot"] - pos[f"{S}_Ankle"])
    lowest = min(p[2] for p in ideal_pos.values())
    for n in ideal_pos:
        ideal_pos[n] = ideal_pos[n] + np.array([0.0, 0.0, min_link_height - lowest])
    print(f"T-pose: frames {start}..{start + win} of {len(poses)}, max joint speed in window "
          f"{speed[start: start + win - 1].max() / (1 / 60):.2f} m/s (at ~60 Hz); operator faces "
          f"{np.degrees(heading.magnitude() * np.sign(heading.as_rotvec()[2])):+.0f} deg from +x")
    return {n: (ideal_pos[n], rot[n]) for n in PICO_JOINTS}, heading


def calibrate_from_tpose(args) -> None:
    """A2 zero pose == T-pose: offsets that map the operator's recorded T-pose exactly onto A2's zero pose."""
    g1_cfg = json.loads(Path(IK_CONFIG_DICT["xrobot"]["unitree_g1"]).read_text())
    g1_weights = {t: {e[0]: e[1:3] for e in g1_cfg[t].values()} for t in ("ik_match_table1", "ik_match_table2")}
    a2 = mujoco.MjModel.from_xml_path(str(ensure_sim_mjcf()))
    ad = forward(a2, {})
    human, heading = tpose_human(args.tpose, args.min_link_height)

    torso_pos = body(a2, ad, "torso")[0]
    a2_root_height = torso_pos[2] - body(a2, ad, "left_toe_roll")[0][2] + A2_SOLE_BELOW_ANKLE
    # A2 zero pose, turned to the operator's heading, standing on the floor
    want = {b: heading.apply(body(a2, ad, b)[0] - torso_pos) + np.array([0.0, 0.0, a2_root_height])
            for b, *_ in MAPPING}
    seg = lambda a, b: np.mean([np.linalg.norm(human[f"{S}_{b}"][0] - human[f"{S}_{a}"][0])  # noqa: E731
                                for S in ("Left", "Right")])
    a2_leg = body(a2, ad, "left_hip_pitch")[0][2] - body(a2, ad, "left_toe_roll")[0][2]
    a2_arm = np.linalg.norm(body(a2, ad, "left-arm-link7")[0] - body(a2, ad, "left-arm-link2")[0])
    s_leg = a2_leg / seg("Hip", "Ankle")
    s_arm = a2_arm / seg("Shoulder", "Wrist")
    scale = {"Pelvis": s_leg}  # root (walking) displacement scaled like the legs
    root = human["Pelvis"][0]

    out = {
        "robot_root_name": "torso",
        "human_root_name": "Pelvis",
        "ground_height": 0.0,
        "human_height_assumption": args.height,
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "human_scale_table": scale,
        "ik_match_table1": {},
        "ik_match_table2": {},
        "_calibration": {"tpose": Path(args.tpose).name, "height_m": args.height},
    }
    for a2_body, joint, g1_human in MAPPING:
        s = s_leg if joint == "Pelvis" else (s_arm if any(k in joint for k in ("Shoulder", "Elbow", "Wrist")) else s_leg)
        scale[joint] = s
        p_h, q_h = human[joint]
        q_a2 = heading * body(a2, ad, a2_body)[1]
        off = q_h.inv() * q_a2
        predicted = s * root if joint == "Pelvis" else s * (p_h - root) + s_leg * root
        local = q_a2.inv().apply(want[a2_body] - predicted)
        quat = off.as_quat(scalar_first=True)
        quat = quat if quat[0] >= 0 else -quat
        for table in ("ik_match_table1", "ik_match_table2"):
            pos_w, rot_w = g1_weights[table][g1_human]
            if a2_body == "torso":
                # torso orientation comes from the chest (below), the pelvis only places the root. GMR keeps
                # offsets only for table1 entries with a non-zero weight, so table1 gets a weak position weight
                rot_w = 0
                pos_w = max(pos_w, 1) if table == "ik_match_table1" else pos_w
            out[table][a2_body] = [joint, pos_w, rot_w, [round(float(v), 4) for v in local],
                                   [round(float(v), 6) for v in quat]]

    # A2 has no waist: its rigid torso (with the welded head) follows the operator's chest (Spine3), like
    # G1's torso_link does. The PICO pelvis estimate tips back when standing normally, which with a
    # pelvis-driven torso showed as a head bent backward. head_yaw is welded to torso (same frame at zero).
    q_chest = human["Spine3"][1]
    quat = (q_chest.inv() * (heading * body(a2, ad, "head_yaw")[1])).as_quat(scalar_first=True)
    quat = quat if quat[0] >= 0 else -quat
    scale["Spine3"] = s_leg
    for table in ("ik_match_table1", "ik_match_table2"):
        pos_w, rot_w = g1_weights[table]["Spine3"]
        out[table]["head_yaw"] = ["Spine3", 0, rot_w, [0.0, 0.0, 0.0], [round(float(v), 6) for v in quat]]
    args.out.write_text(json.dumps(out, indent=4) + "\n")
    print(f"segments: hip->ankle {seg('Hip', 'Ankle'):.3f} m (A2 {a2_leg:.3f}), shoulder->wrist "
          f"{seg('Shoulder', 'Wrist'):.3f} m (A2 {a2_arm:.3f}); hip width "
          f"{np.linalg.norm(human['Left_Hip'][0] - human['Right_Hip'][0]):.3f} m")
    print(f"scales: pelvis/legs {s_leg:.3f}  arms {s_arm:.3f};  operator height for teleop.sh retarget: {args.height}")
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tpose", type=Path, required=True,
                    help="raw PICO T-pose recording (tools/record_pico.py): calibrate directly against A2's T-pose")
    ap.add_argument("--out", type=Path, required=True, help="calibrated GMR config (teleop.sh: <calib>/xrobot_to_a2_tpose_<ts>.json)")
    ap.add_argument("--height", type=float, default=1.75, help="operator height passed to 'teleop.sh retarget'")
    ap.add_argument("--min-link-height", type=float, default=0.01, help="pico_retarget_pub --min-link-height")
    calibrate_from_tpose(ap.parse_args())


if __name__ == "__main__":
    main()
