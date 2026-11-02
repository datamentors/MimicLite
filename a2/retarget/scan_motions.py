"""Backward-locomotion + flight/landing coverage of A2 MimicLite clips (qpos npz, 50 fps).
usage: python scan_motions.py <a2_kin.xml> <motions dir or npz> [--workers N] [--limit N]
Per 0.5 s window: root velocity in the reference heading frame (fwd/lat). Flight: both feet (toe_roll) > 3 cm above
the clip floor (median of per-frame lowest foot); a landing = flight phase >= 0.1 s that ends."""
import argparse
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

W, DT = 25, 0.02
_m = _d = _feet = _qmap = None


def _init(xml, feet, qmap):
    global _m, _d, _feet, _qmap
    _m = mujoco.MjModel.from_xml_path(xml)
    _d = mujoco.MjData(_m)
    _feet = [mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, n) for n in feet]
    _qmap = None if qmap is None else [int(_m.jnt_qposadr[mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in qmap]


def scan(path):
    try:
        q = np.load(path)["qpos"].astype(np.float64)
    except Exception as e:  # unreadable clip: report, don't crash the scan
        return {"path": str(path), "error": repr(e)}
    n = len(q)
    if n < 2 * W:
        return {"path": str(path), "frames": n, "win": np.zeros((0, 2)), "flight": 0, "landings": 0, "air_max": 0.0}
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    # per-frame velocity in that frame's heading, averaged per window (turning clips stay correct)
    d = np.diff(q[:, :2], axis=0) / DT
    c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    nw = (n - 1) // W
    fwd = (c * d[:, 0] + s * d[:, 1])[: nw * W].reshape(nw, W).mean(1)
    lat = (-s * d[:, 0] + c * d[:, 1])[: nw * W].reshape(nw, W).mean(1)
    fz = np.empty((n, 2))
    for i in range(n):
        if _qmap is None:
            _d.qpos[:] = q[i]
        else:
            _d.qpos[:7] = q[i, :7]
            _d.qpos[_qmap] = q[i, 7:]
        mujoco.mj_kinematics(_m, _d)
        fz[i] = _d.xpos[_feet, 2]
    fz -= np.median(fz.min(1))  # clip floor (robust to feet dipping below it)
    air = (fz > 0.03).all(1)
    edges = np.diff(np.concatenate([[0], air.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    landings = int(sum((e - b) >= 5 and e < n for b, e in zip(starts, ends)))
    return {"path": str(path), "frames": n, "win": np.stack([fwd, lat], 1), "flight": int(air.sum()),
            "landings": landings, "air_max": float(fz.min(1).max())}


def category(path):
    stem = Path(path).name.split("__")[0]
    return "_".join(t for t in stem.split("_") if not t.isdigit() and t not in ("L", "R", "M"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("src")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--feet", default="left_toe_roll,right_toe_roll")
    ap.add_argument("--manifest", default=None, help="any4hdmi manifest: map qpos_names joints onto the MJCF")
    a = ap.parse_args()
    src = Path(a.src)
    files = sorted(src.rglob("*.npz")) if src.is_dir() else [src]
    if a.limit:
        files = files[:: max(1, len(files) // a.limit)][: a.limit]
    import json
    qmap = json.load(open(a.manifest))["qpos_names"][7:] if a.manifest else None
    with Pool(a.workers, initializer=_init, initargs=(a.xml, a.feet.split(","), qmap)) as pool:
        res = pool.map(scan, files, chunksize=64)
    errs = [r for r in res if "error" in r]
    res = [r for r in res if "error" not in r]
    frames = sum(r["frames"] for r in res)
    win = np.concatenate([r["win"] for r in res]) if res else np.zeros((0, 2))
    fwd = win[:, 0]
    print(f"clips {len(res)} (unreadable {len(errs)}), {frames * DT / 3600:.1f} h, 0.5 s windows {len(win)}")
    for lo, hi, name in ((-9, -0.6, "backward < -0.6"), (-0.6, -0.3, "backward -0.6..-0.3"), (-0.3, 0.3, "|fwd| < 0.3"),
                         (0.3, 0.6, "forward 0.3..0.6"), (0.6, 9, "forward > 0.6")):
        m = (fwd >= lo) & (fwd < hi)
        print(f"  {name:22s} {100 * m.mean():6.2f} % of time  ({m.sum() * W * DT / 3600:.2f} h)")
    # sustained backward bouts (>= 1 s: two consecutive windows < -0.3) and starts from standstill
    bouts = starts = 0
    bclips = Counter()
    for r in res:
        f = r["win"][:, 0] if len(r["win"]) else np.zeros(0)
        b = (f[:-1] < -0.3) & (f[1:] < -0.3)
        st = (np.abs(f[:-1]) < 0.15) & (f[1:] < -0.4)
        bouts += int(b.sum())
        starts += int(st.sum())
        if b.any():
            bclips[category(r["path"])] += 1
    nb = sum(bclips.values())
    print(f"  clips with >= 1 s backward at > 0.3 m/s: {nb} ({100 * nb / max(len(res), 1):.2f} %); "
          f"1 s backward windows {bouts}; standstill -> backward >0.4 m/s transitions {starts}")
    print("  top backward categories:", ", ".join(f"{k} {v}" for k, v in bclips.most_common(12)))
    fl = sum(r["flight"] for r in res)
    lands = sum(r["landings"] for r in res)
    lclips = sum(r["landings"] > 0 for r in res)
    print(f"  flight (both feet > 3 cm) {100 * fl / max(frames, 1):.2f} % of frames; landings (flight >= 0.1 s) {lands} "
          f"in {lclips} clips ({100 * lclips / max(len(res), 1):.2f} %)")
    if errs:
        print("  unreadable:", "; ".join(f"{e['path']}: {e['error']}" for e in errs[:5]))


if __name__ == "__main__":
    main()
