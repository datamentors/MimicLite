"""Generate a2/teleop/xrobot_to_a2.json (GMR IK config: PICO/XRoboToolkit body -> AgiBot A2) from GMR's
own xrobot_to_g1.json, which is already tuned for PICO body-tracking data.

Run in the sim2real teleop venv (has GMR + mujoco + scipy):
  sim2real/venv/teleop/.venv/bin/python a2/teleop/tools/make_xrobot_to_a2.py [--out PATH]

Derivation
----------
GMR sets each robot body target to  q_target = q_human * rot_offset  (wxyz, scipy, offsets keyed by human
joint; table2 reuses table1's offsets). For a pose P in which G1 and A2 look the same,
  q_G1(P) = q_human(P) * off_G1   and   q_A2(P) = q_human(P) * off_A2
  =>  off_A2 = off_G1 * q_G1(P)^-1 * q_A2(P)
P = G1 MJCF zero pose (legs straight, upper arms down, elbows 90 deg with forearms forward).
A2 in P: legs at zero, arms (j1..j4) = (0, +-pi/2, 0, -+pi/2) (asserted below). Forearm twist j5 in P is
A2's neutral twist from the training data (BonesSeedA2 medians left -0.06 / right -1.49, = policy default
pose -0.05 / -1.51): left 0, right -pi/2. A "thumb up" twist (+pi/2 both) put the retargeted j5 at
+2.6 / +2.1 rad, outside anything the policy was trained on.

Mapping (A2 has no waist: torso is the root, so Spine3 is not used):
  torso <- Pelvis, *_hip_pitch <- *_Hip, *_tarsus (knee) <- *_Knee, *_toe_roll <- *_Ankle,
  *-arm-link3 <- *_Shoulder, *-arm-link4 <- *_Elbow, *-arm-link7 <- *_Wrist
Weights are copied from the G1 analog. G1 targets its toe from Left_Foot; A2 targets the ankle joint
centre from Left_Ankle with Left_Foot's weights and offset (all xrobot lower-body joints share the same
offset in the G1 config, i.e. the same frame convention).

Scales (positions are scaled about the pelvis; arms have position weight 0, only rotations matter):
  legs:   0.9 * (A2 hip_pitch->ankle) / (G1 hip_pitch->ankle)
  pelvis: matches A2 standing root height, with a torso position offset along torso z that moves the
          target from the A2 torso origin (0.21 m above the hip axis) to where G1's pelvis would be.
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
from a2_sim2real import XROBOT_TO_A2_JSON, ensure_sim_mjcf  # noqa: E402

from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT  # noqa: E402

A2_SOLE_BELOW_ANKLE = 0.059  # toe_roll origin to foot contact spheres (a2_serial.xml)
HALF_PI = float(np.pi / 2)
A2_POSE_P = {
    "idx14_left_arm_joint2": HALF_PI, "idx16_left_arm_joint4": -HALF_PI, "idx17_left_arm_joint5": 0.0,
    "idx21_right_arm_joint2": -HALF_PI, "idx23_right_arm_joint4": HALF_PI, "idx24_right_arm_joint5": -HALF_PI,
}
# (A2 body, human joint, G1 body, G1 human joint whose weights/offset are reused)
MAPPING = [("torso", "Pelvis", "pelvis", "Pelvis")]
for side, S in (("left", "Left"), ("right", "Right")):
    MAPPING += [
        (f"{side}_hip_pitch", f"{S}_Hip", f"{side}_hip_yaw_link", f"{S}_Hip"),
        (f"{side}_tarsus", f"{S}_Knee", f"{side}_knee_link", f"{S}_Knee"),
        (f"{side}_toe_roll", f"{S}_Ankle", f"{side}_ankle_roll_link", f"{S}_Foot"),
        (f"{side}-arm-link3", f"{S}_Shoulder", f"{side}_shoulder_yaw_link", f"{S}_Shoulder"),
        (f"{side}-arm-link4", f"{S}_Elbow", f"{side}_elbow_link", f"{S}_Elbow"),
        (f"{side}-arm-link7", f"{S}_Wrist", f"{side}_wrist_yaw_link", f"{S}_Wrist"),
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


def check_pose_p(a2: mujoco.MjModel, d: mujoco.MjData) -> None:
    for side, S in (("left", "L"), ("right", "R")):
        shoulder, _ = body(a2, d, f"{side}-arm-link2")
        elbow, _ = body(a2, d, f"{side}-arm-link4")
        paw, _ = body(a2, d, f"{S}_paw")
        upper, fore = elbow - shoulder, paw - elbow
        if not (upper[2] < -0.2 and np.abs(upper[:2]).max() < 0.03 and fore[0] > 0.2 and abs(fore[2]) < 0.03):
            raise AssertionError(f"A2 pose P is not 'upper arm down, forearm forward' ({side})")


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
    for a2_body, joint, _, g1_human in MAPPING:
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
            out[table][a2_body] = [joint, pos_w, rot_w, [round(float(v), 4) for v in local],
                                   [round(float(v), 6) for v in quat]]
    args.out.write_text(json.dumps(out, indent=4) + "\n")
    print(f"segments: hip->ankle {seg('Hip', 'Ankle'):.3f} m (A2 {a2_leg:.3f}), shoulder->wrist "
          f"{seg('Shoulder', 'Wrist'):.3f} m (A2 {a2_arm:.3f}); hip width "
          f"{np.linalg.norm(human['Left_Hip'][0] - human['Right_Hip'][0]):.3f} m")
    print(f"scales: pelvis/legs {s_leg:.3f}  arms {s_arm:.3f};  operator height for teleop.sh retarget: {args.height}")
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=XROBOT_TO_A2_JSON)
    ap.add_argument("--tpose", type=Path, default=None,
                    help="raw PICO T-pose recording (tools/record_pico.py): calibrate directly against A2's T-pose")
    ap.add_argument("--height", type=float, default=1.75, help="operator height passed to 'teleop.sh retarget'")
    ap.add_argument("--min-link-height", type=float, default=0.01, help="pico_retarget_pub --min-link-height")
    args = ap.parse_args()
    if args.tpose is not None:
        calibrate_from_tpose(args)
        return

    g1_cfg = json.loads(Path(IK_CONFIG_DICT["xrobot"]["unitree_g1"]).read_text())
    g1 = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    a2 = mujoco.MjModel.from_xml_path(str(ensure_sim_mjcf()))
    gd = forward(g1, {})
    ad = forward(a2, A2_POSE_P)
    check_pose_p(a2, ad)

    g1_by_human = {}
    for table in ("ik_match_table1", "ik_match_table2"):
        for g1_body, entry in g1_cfg[table].items():
            g1_by_human.setdefault(table, {})[entry[0]] = (g1_body, entry)

    # geometry (left side; both robots are symmetric)
    z = lambda m, d, n: float(body(m, d, n)[0][2])  # noqa: E731
    g1_leg = z(g1, gd, "left_hip_pitch_link") - z(g1, gd, "left_ankle_roll_link")
    a2_leg = z(a2, ad, "left_hip_pitch") - z(a2, ad, "left_toe_roll")
    g1_root_drop = z(g1, gd, "pelvis") - z(g1, gd, "left_hip_pitch_link")
    a2_root_drop = z(a2, ad, "torso") - z(a2, ad, "left_hip_pitch")
    g1_root_height = z(g1, gd, "pelvis")  # G1 MJCF qpos0 stands on the floor
    a2_root_height = z(a2, ad, "torso") - z(a2, ad, "left_toe_roll") + A2_SOLE_BELOW_ANKLE
    g1_scale = g1_cfg["human_scale_table"]
    leg_ratio = a2_leg / g1_leg
    s_leg = g1_scale["Left_Knee"] * leg_ratio
    # torso target offset (torso z): A2 root sits higher above the hips than G1's pelvis does
    torso_dz = a2_root_drop - leg_ratio * g1_root_drop
    human_pelvis_height = g1_root_height / g1_scale["Pelvis"]
    s_pelvis = (a2_root_height - torso_dz) / human_pelvis_height
    g1_arm = np.linalg.norm(body(g1, gd, "left_wrist_yaw_link")[0] - body(g1, gd, "left_shoulder_yaw_link")[0])
    a2_arm = np.linalg.norm(body(a2, ad, "left-arm-link7")[0] - body(a2, ad, "left-arm-link3")[0])
    s_arm = g1_scale["Left_Elbow"] * a2_arm / g1_arm

    scale = {"Pelvis": s_pelvis}
    out = {
        "robot_root_name": "torso",
        "human_root_name": "Pelvis",
        "ground_height": 0.0,
        "human_height_assumption": g1_cfg["human_height_assumption"],
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "human_scale_table": scale,
        "ik_match_table1": {},
        "ik_match_table2": {},
    }
    for a2_body, human, g1_body_p, g1_human in MAPPING:
        if human != "Pelvis":
            scale[human] = s_arm if any(k in human for k in ("Shoulder", "Elbow", "Wrist")) else s_leg
        q_g1 = body(g1, gd, g1_body_p)[1]
        q_a2 = body(a2, ad, a2_body)[1]
        for table in ("ik_match_table1", "ik_match_table2"):
            _, (_, pos_w, rot_w, _, off_g1) = g1_by_human[table][g1_human]
            off_a2 = R.from_quat(off_g1, scalar_first=True) * q_g1.inv() * q_a2
            quat = off_a2.as_quat(scalar_first=True)
            quat = quat if quat[0] >= 0 else -quat
            pos_off = [0.0, 0.0, round(torso_dz, 4)] if a2_body == "torso" else [0.0, 0.0, 0.0]
            out[table][a2_body] = [human, pos_w, rot_w, pos_off, [round(float(v), 6) for v in quat]]

    args.out.write_text(json.dumps(out, indent=4) + "\n")
    print(f"G1 leg {g1_leg:.3f}  A2 leg {a2_leg:.3f}  ratio {leg_ratio:.3f}")
    print(f"root above hip: G1 {g1_root_drop:.3f}  A2 {a2_root_drop:.3f}  -> torso target dz {torso_dz:.3f}")
    print(f"standing root height: G1 {g1_root_height:.3f}  A2 {a2_root_height:.3f}")
    print(f"scales: pelvis {s_pelvis:.3f}  legs {s_leg:.3f}  arms {s_arm:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
