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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=XROBOT_TO_A2_JSON)
    args = ap.parse_args()

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
