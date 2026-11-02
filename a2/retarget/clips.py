"""Clip helpers for run.sh (sim2real venv, numpy only).

Commands (``python clips.py <command> <args...>``):
    pad <src.npz> <dst.npz> <n> [ramp manifest]: prepend n copies of frame 0 (GMR IK starts cold, this lets
        frame 0 converge). With ramp > 0, first ease the G1 arms from a T-pose (GMR's A2 start pose) to frame 0
        over ramp frames, so the IK follows a continuous path into frame 0.
    trim <clip.npz> <n>: drop the first n frames in place (removes the padding again).
    score <traj.npz> <clip.npz>: integrated_sim2sim trajectory -> tracking summary line.
"""

import json
from pathlib import Path
import sys

import numpy as np

G1_TPOSE_ARMS = {
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 1.57,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -1.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -1.57,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -1.0,
}


def pad(src: str, dst: str, n: str, ramp: str = "0", manifest: str = "") -> None:
    """Prepend copies of frame 0 (and optionally a T-pose arm ramp) to a G1 clip.

    Args:
        src: Input any4hdmi clip (npz with ``qpos``).
        dst: Output npz path.
        n: Number of frame-0 copies to prepend.
        ramp: Number of frames to ease the arms from the T-pose into frame 0; "0" disables the ramp.
        manifest: any4hdmi manifest.json giving ``qpos_names`` (required when ramp > 0).
    """
    q = np.load(src)["qpos"]
    parts = [np.repeat(q[:1], int(n), axis=0), q]
    if int(ramp) > 0:
        with Path(manifest).open() as f:
            names = json.load(f)["qpos_names"]
        start = q[0].copy()
        for joint, value in G1_TPOSE_ARMS.items():
            start[names.index(joint)] = value
        a = np.linspace(0.0, 1.0, int(ramp), endpoint=False)[:, None]
        parts.insert(0, (start * (1 - a) + q[:1] * a).astype(q.dtype))
    np.savez(dst, qpos=np.concatenate(parts))


def trim(clip: str, n: str) -> None:
    """Drop the first n frames of a clip in place.

    Args:
        clip: npz clip with ``qpos``; overwritten.
        n: Number of leading frames to remove.
    """
    q = np.load(clip)["qpos"]
    np.savez(clip, qpos=q[int(n) :])


def score(traj_path: str, clip_path: str) -> None:
    """Print a one-line tracking summary of an integrated_sim2sim trajectory.

    Reports mean body error (root-relative and world), the first frame where the root drops below 0.5 m (fall),
    and the largest per-frame joint step of the reference.

    Args:
        traj_path: Trajectory npz written by integrated_sim2sim ``--trajectory_output``.
        clip_path: The A2 reference clip the policy tracked.
    """
    t = np.load(traj_path, allow_pickle=True)
    q = np.load(clip_path)["qpos"]
    frame = t["motion_t"]  # reference frame index (50 fps)
    rob, ref = t["robot_body_pos_w"], t["motion_body_pos_w"]
    world = np.linalg.norm(rob - ref, axis=-1).mean(-1)
    # body positions relative to the root: tracking quality without global drift
    local = np.linalg.norm(
        (rob - t["robot_root_pos_w"][:, None]) - (ref - t["motion_root_pos_w"][:, None]), axis=-1
    ).mean(-1)
    fall = np.where(t["robot_root_pos_w"][:, 2] < 0.5)[0]
    jump = np.abs(np.diff(q[:, 7:], axis=0)).max(1)
    fall_txt = f"at frame {int(frame[fall[0]])} ({frame[fall[0]] * 0.02:.1f} s)" if len(fall) else "none"
    print(
        f"  {clip_path.rsplit('/', 1)[-1]}: {len(q)} frames ({len(q) * 0.02:.1f} s); "
        f"body err root-relative {100 * local.mean():.1f} cm (p95 {100 * np.percentile(local, 95):.1f}), "
        f"world {100 * world.mean():.1f} cm; fall {fall_txt}; "
        f"ref max joint step {jump.max():.2f} rad ({int((jump > 0.36).sum())} steps > 18 rad/s)"
    )


if __name__ == "__main__":
    {"pad": pad, "trim": trim, "score": score}[sys.argv[1]](*sys.argv[2:])
