"""Show A2 standing still in a pose in the browser (no physics): `zero` = the T-pose the operator calibration
maps onto (teleop.sh calibrate), `default` = the policy's stand pose.

Run in the sim2real venv:  sim2real/.venv/bin/python <a2>/teleop/tools/view_pose.py [zero|default] [--port N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a2_sim2real  # noqa: E402

a2_sim2real.register()
from sim2real.config.robots import get_robot_cfg  # noqa: E402
from sim2real.sim_env.utils.mjcf import load_sim_model  # noqa: E402
from sim2real.utils.mjviser_viewer import MjviserMujocoViewer  # noqa: E402

A2_SOLE_BELOW_ANKLE = 0.059


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pose", nargs="?", choices=("zero", "default"), default="zero")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    cfg = get_robot_cfg("a2")
    model = load_sim_model(cfg)
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(cfg.default_qpos)
    if args.pose == "zero":
        data.qpos[7:] = 0.0
    # stand the feet on the floor
    mujoco.mj_kinematics(model, data)
    ankle_z = min(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)][2]
                  for b in ("left_toe_roll", "right_toe_roll"))
    data.qpos[2] += A2_SOLE_BELOW_ANKLE - ankle_z
    mujoco.mj_forward(model, data)

    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    viewer = MjviserMujocoViewer(model, data, label=f"a2-{args.pose}-pose", port=args.port,
                                 tracked_body_id=torso, create_gui=False)
    print(f"A2 {args.pose} pose (x = forward, y = robot's left). Ctrl+C to stop.", flush=True)
    try:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
