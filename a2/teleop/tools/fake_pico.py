"""Run the real sim2real pico_retarget_pub.py with a fake PICO: no headset or XRoboToolkit service needed.

The fake XRobotStreamer replays a G1 motion as PICO body tracking (G1ToHuman, converted back to the Unity
frame the SDK delivers) in real time and presses the controller buttons the operator would:
  t=3 s A (policy: stand/init) -> t=6 s A+B (policy on) -> t=9 s X (publisher: start following).
Everything downstream (GMR A2 retarget, ZMQ motion stream :28701, button relay :5592, policy, sim) is the
unchanged teleop path.

Run in the sim2real teleop venv:
  sim2real/venv/teleop/.venv/bin/python <a2>/teleop/tools/fake_pico.py \
      --motion any4hdmi/output/g1/lafan/motions/walk1_subject5.npz --manifest any4hdmi/output/g1/lafan/manifest.json \
      -- --robot a2 --no-viewer [other pico_retarget_pub args]
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_retarget import G1ToHuman  # noqa: E402  (registers A2 on import)

import general_motion_retargeting as gmr  # noqa: E402

UNITY_TO_RH = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # xrobot_utils transform
UNITY_TO_RH_ROT = R.from_matrix(UNITY_TO_RH)
BUTTON_SCRIPT = [(3.0, 4.0, ("A",)), (6.0, 7.0, ("A", "B")), (9.0, 9.5, ("X",))]  # (start s, end s, pressed)


class FakeXRobotStreamer(gmr.XRobotStreamer):
    motion: np.ndarray
    to_human: G1ToHuman
    fps: float

    def __init__(self):  # no xrt.init(): nothing to connect to
        self.body_joint_names = [
            "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
            "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
            "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
            "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
        ]
        self.t0 = time.monotonic()

    def _human_frame(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        idx = int((time.monotonic() - self.t0) * self.fps) % len(self.motion)
        human = dict(self.to_human(self.motion[idx]))
        # joints the A2 config does not use: plausible placeholders above the floor
        pelvis_pos, pelvis_quat = human["Pelvis"]
        for name, dz in (("Spine1", 0.1), ("Spine2", 0.2), ("Spine3", 0.3), ("Neck", 0.5), ("Head", 0.6)):
            human[name] = (pelvis_pos + np.array([0.0, 0.0, dz]), pelvis_quat)
        for S in ("Left", "Right"):
            human[f"{S}_Collar"] = human[f"{S}_Shoulder"]
            human[f"{S}_Hand"] = human[f"{S}_Wrist"]
        return human

    def get_raw_body_data(self):
        human = self._human_frame()
        poses = np.zeros((len(self.body_joint_names), 7), dtype=np.float32)
        for i, name in enumerate(self.body_joint_names):
            pos, quat_wxyz = human[name]
            pos_u = UNITY_TO_RH.T @ pos
            q_u = (UNITY_TO_RH_ROT.inv() * R.from_quat(quat_wxyz, scalar_first=True)).as_quat()  # xyzw
            poses[i] = np.concatenate([pos_u, q_u])
        return poses, None, None, None, time.time_ns()

    def get_controller_data(self):
        t = time.monotonic() - self.t0
        pressed = set()
        for start, end, keys in BUTTON_SCRIPT:
            if start <= t < end:
                pressed.update(keys)
        side = lambda one, two: {  # noqa: E731
            "index_trig": 0.0, "grip": 0.0, "key_one": one in pressed, "key_two": two in pressed,
            "axis": [0.0, 0.0], "axis_click": False,
        }
        return {"LeftController": side("X", "Y"), "RightController": side("A", "B"), "timestamp": time.time_ns()}


def main() -> None:
    if "--" not in sys.argv:
        sys.exit(__doc__)
    split = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args(sys.argv[1:split])

    FakeXRobotStreamer.motion = np.load(args.motion)["qpos"]
    FakeXRobotStreamer.to_human = G1ToHuman(args.manifest)
    FakeXRobotStreamer.fps = 1.0 / float(json.loads(args.manifest.read_text())["timestep"])
    gmr.XRobotStreamer = FakeXRobotStreamer

    import sim2real
    script = Path(sim2real.__file__).resolve().parent / "teleop" / "pico_retarget_pub.py"
    sys.argv = [str(script)] + sys.argv[split + 1:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
