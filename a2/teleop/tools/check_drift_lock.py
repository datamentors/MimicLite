"""Offline check of the PICO drift lock (a2_sim2real.DriftLock) without a headset.

1. Standing still: a raw PICO recording (e.g. a T-pose from tools/record_pico.py) plus an injected constant
   drift of the whole PICO world frame -> the corrected pelvis must stay put.
2. Walking: a G1 clip as PICO body data (check_retarget.G1ToHuman, like tools/fake_pico.py) -> real steps must
   pass through (corrected pelvis path == raw path); with injected drift, the drift is removed while standing.

Run in the sim2real teleop venv (from the MimicLite checkout that holds sim2real/ and any4hdmi/):
  sim2real/venv/teleop/.venv/bin/python <a2>/teleop/tools/check_drift_lock.py --tpose <a2>/../calib/tpose_<ts>.npz \
      --motion any4hdmi/output/g1/lafan/motions/walk1_subject5.npz --manifest any4hdmi/output/g1/lafan/manifest.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_retarget import G1ToHuman  # noqa: E402  (registers A2 on import)

from a2_sim2real import DriftLock  # noqa: E402
from general_motion_retargeting.xrobot_utils import XRobotStreamer  # noqa: E402  (unpatched transform)
from make_xrobot_to_a2 import PICO_JOINTS  # noqa: E402


def run(frames: list[dict], t: np.ndarray, drift_mps: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Feed frames (+ drift * t) through a fresh DriftLock: raw and corrected pelvis xy, locked fraction, max |offset|."""
    lock = DriftLock()
    raw, out, locked, max_off = [], [], 0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):  # lock/release log lines
        for frame, ts in zip(frames, t):
            shift = drift_mps * ts
            d = {n: [[p[0] + shift[0], p[1] + shift[1], p[2]], q] for n, (p, q) in frame.items()}
            corrected = lock.apply(d, float(ts))
            raw.append(d["Pelvis"][0][:2])
            out.append(corrected["Pelvis"][0][:2])
            locked += lock.locked
            max_off = max(max_off, float(np.linalg.norm(lock.offset)))
    return np.asarray(raw), np.asarray(out), locked / len(frames), max_off


def recording_frames(path: Path) -> tuple[list[dict], np.ndarray]:
    rec = np.load(path)
    frames = []
    for raw in rec["poses"]:  # (T, 24, 7) Unity x y z qx qy qz qw -> z-up, like pico_retarget_pub
        d = {n: [raw[i, :3].tolist(), [raw[i, 6], raw[i, 3], raw[i, 4], raw[i, 5]]] for i, n in enumerate(PICO_JOINTS)}
        frames.append(XRobotStreamer.coordinate_transform_unity_data(None, d))
    return frames, np.asarray(rec["t"], dtype=np.float64)


def clip_frames(motion: Path, manifest: Path) -> tuple[list[dict], np.ndarray]:
    to_human = G1ToHuman(manifest)
    dt = float(json.loads(manifest.read_text())["timestep"])
    frames = [{n: [p.tolist(), q] for n, (p, q) in to_human(f).items()} for f in np.load(motion)["qpos"]]
    return frames, np.arange(len(frames)) * dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tpose", type=Path, required=True, help="raw PICO recording of the operator standing still")
    ap.add_argument("--motion", type=Path, required=True, help="G1 clip with walking (any4hdmi npz)")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--drift", type=float, default=0.01, help="injected drift speed [m/s] (exaggerated: ~0.6 m/min)")
    args = ap.parse_args()
    drift = np.array([0.0, -args.drift])

    frames, t = recording_frames(args.tpose)
    raw, out, frac, _ = run(frames, t, drift)
    raw_shift, out_shift = np.linalg.norm(raw[-1] - raw[0]), np.linalg.norm(out[-1] - out[0])
    print(f"standing ({args.tpose.name}, {t[-1]:.1f} s, drift {args.drift} m/s): pelvis shift raw {raw_shift:.3f} m "
          f"-> corrected {out_shift:.3f} m, locked {frac:.0%}  [{'PASS' if out_shift < 0.01 else 'FAIL'} < 0.01 m]")

    frames, t = clip_frames(args.motion, args.manifest)
    raw, out, frac, max_off = run(frames, t, np.zeros(2))
    path = np.linalg.norm(np.diff(raw, axis=0), axis=1).sum()
    end_err = np.linalg.norm((out[-1] - out[0]) - (raw[-1] - raw[0]))
    print(f"walking ({args.motion.name}, {t[-1]:.1f} s, no drift): pelvis path {path:.2f} m, locked {frac:.0%}, "
          f"max |offset| {max_off:.3f} m, end displacement error {end_err:.3f} m  "
          f"[{'PASS' if max_off < 0.05 else 'FAIL'} offset < 0.05 m]")
    raw, out, frac, _ = run(frames, t, drift)
    removed = 1.0 - np.linalg.norm((out[-1] - out[0]) - (raw[-1] - raw[0] - drift * t[-1])) / (args.drift * t[-1])
    print(f"walking + drift {args.drift} m/s: injected {args.drift * t[-1]:.2f} m, removed {removed:.0%} "
          f"(only while both feet are still)")


if __name__ == "__main__":
    main()
