"""Kinematic replay of AgiBot A2 motions in the browser (MuJoCo FK + mjviser), no physics.

Inputs (any mix, switch between them in the GUI dropdown):
  *.csv  raw BonesSeedA2/soma_uniform retarget (120 fps, root cm + euler xyz deg, joints deg).
         Parsed like step 1 of a2/tools/convert_bonesseed_a2.py, nothing else
         (no height fix, no leg re-solve, no cuts) -> shows the retarget as stored on S3.
  *.npz  converted MimicLite motion (qpos: root xyz m, quat wxyz, 26 hinges in MJCF order), --npz-fps.

usage (on the workstation, from the MimicLite checkout):
  sim2real/.venv/bin/python a2/retarget/replay_a2.py <motions>/*.csv <motions>/*.npz
  then open http://<workstation-ip>:8090
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sim2real.utils.mjviser_viewer import MjviserMujocoViewer

HERE = Path(__file__).resolve().parent
CSV_FPS = 120.0


def load_csv(path: Path, m: mujoco.MjModel) -> np.ndarray:
    """Read a raw BonesSeedA2 CSV into qpos rows of the given A2 model.

    Args:
        path: CSV with root translate (cm), root rotate (euler xyz deg) and per-joint ``*_dof`` columns (deg).
        m: A2 kinematic model; its hinge order defines the qpos joint columns.

    Returns:
        (frames, nq) qpos array: root xyz in m, quat wxyz, hinge angles in rad.
    """
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    hinges = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    q = np.zeros((len(df), m.nq))
    q[:, 0:3] = df[["root_translateX", "root_translateY", "root_translateZ"]].to_numpy() / 100.0
    euler = df[["root_rotateX", "root_rotateY", "root_rotateZ"]].to_numpy()
    q[:, 3:7] = Rotation.from_euler("xyz", euler, degrees=True).as_quat()[:, [3, 0, 1, 2]]
    # idx05_left_toe_pitch -> left_toe_pitch_dof (head columns unused: head is welded)
    q[:, 7:] = np.deg2rad(df[[j.split("_", 1)[1] + "_dof" for j in hinges]].to_numpy())
    return q


def main() -> None:
    """Load the clips and serve the browser replay viewer until it is closed."""
    p = argparse.ArgumentParser()
    p.add_argument("motions", nargs="+", type=Path)
    p.add_argument("--mjcf", type=Path, default=HERE.parent / "model" / "a2_serial" / "a2_kin.xml")
    p.add_argument("--npz-fps", type=float, default=50.0)
    p.add_argument("--port", type=int, default=8090)
    args = p.parse_args()

    m = mujoco.MjModel.from_xml_path(str(args.mjcf))
    d = mujoco.MjData(m)

    clips: dict[str, tuple[np.ndarray, float]] = {}
    for path in args.motions:
        if path.suffix == ".csv":
            q, fps = load_csv(path, m), CSV_FPS
        else:
            q, fps = np.load(path)["qpos"].astype(np.float64), args.npz_fps
        assert q.shape[1] == m.nq, f"{path}: qpos dim {q.shape[1]} != model nq {m.nq}"
        label = f"{path.name}  [{fps:g} fps, {len(q)} fr, {len(q) / fps:.2f} s]"
        clips[label] = (q, fps)
        print(f"loaded {label}  root z range {q[:, 2].min():.3f}..{q[:, 2].max():.3f} m")

    names = list(clips)
    viewer = MjviserMujocoViewer(m, d, label="A2 motion replay", port=args.port, tracked_body_id=1, camera_distance=3.0)
    gui = viewer.server.gui
    with gui.add_folder("Replay"):
        clip_dd = gui.add_dropdown("Clip", options=names, initial_value=names[0])
        playing = gui.add_checkbox("Play", initial_value=True)
        loop = gui.add_checkbox("Loop", initial_value=True)
        speed = gui.add_slider("Speed", min=0.05, max=2.0, step=0.05, initial_value=1.0)
        frame = gui.add_slider("Frame", min=0, max=len(clips[names[0]][0]) - 1, step=1, initial_value=0)
        info = gui.add_markdown("")

    state = {"t": 0.0, "clip": names[0]}

    @clip_dd.on_update
    def _(_) -> None:
        state["clip"] = clip_dd.value
        state["t"] = 0.0
        frame.max = len(clips[clip_dd.value][0]) - 1
        frame.value = 0

    @frame.on_update
    def _(_) -> None:
        if not playing.value:  # scrubbing while paused
            state["t"] = frame.value / clips[state["clip"]][1]

    last = time.perf_counter()
    while viewer.is_running():
        now = time.perf_counter()
        dt, last = now - last, now
        q, fps = clips[state["clip"]]
        dur = (len(q) - 1) / fps
        if playing.value:
            state["t"] += dt * speed.value
            if state["t"] > dur:
                state["t"] = state["t"] % dur if loop.value else dur
        i = min(round(state["t"] * fps), len(q) - 1)
        d.qpos[:] = q[i]
        mujoco.mj_kinematics(m, d)
        viewer.sync()
        if playing.value and frame.value != i:
            frame.value = i
        info.content = f"t = {i / fps:.2f} / {dur:.2f} s · frame {i}/{len(q) - 1} · root z {q[i, 2]:.3f} m"
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
