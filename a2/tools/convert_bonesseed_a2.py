"""Convert BonesSeedA2 soma_uniform CSVs (S3, read-only) into one any4hdmi dataset for MimicLite.

Input CSV (Newton IK output, 120 fps): Frame, root_translate{X,Y,Z} [cm], root_rotate{X,Y,Z}
[deg, extrinsic xyz], then 28 joint columns [deg] named <joint>_dof (head, legs, 2x7 arms).
Output: <out>/manifest.json, <out>/mjcf/a2_kin.xml (kinematics-only twin of a2_serial.xml, + meshes), <out>/motions/<date>/<clip>.npz
(qpos float32 at --fps: root xyz [m], root quat wxyz, 26 hinge joints in MJCF order).

Per clip:
  1. qpos at 120 fps; head columns dropped (head is welded in the serial model)
  2. height fix: lift root so the 10th-percentile lowest ankle sits at the sole height (the
     retarget leaves stance feet ~2-3 cm in the floor); offset clamped to [-0.05, 0.10] m
  2b. re-solve leg joints to the retargeted foot poses (removes pinned-knee branch flips)
  3. resample to --fps (linear joints/position, slerp root rotation)
  4. reject clips that would fail MimicLite's dataset validator (mimic_lite/tasks/motion.py):
     non-finite, |joint| >= 3 rad, |joint vel| >= 20 rad/s, body z >= 2.5 m,
     body lin vel >= 10 m/s, body ang vel >= 30 rad/s; and clips shorter than --min-seconds
S3 is processed one date folder at a time: sync to a scratch dir, convert, delete the scratch.
Already-converted clips are skipped, so the run can be stopped and resumed.

usage: python convert_bonesseed_a2.py --out /data/motions/bonesseed_a2 --mjcf-dir /opt/a2/a2_serial
       [--s3 s3://.../BonesSeedA2/soma_uniform] [--dates 210531,...] [--limit N] [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

DEFAULT_S3 = os.environ.get("A2_S3_URI", "")  # s3://<bucket>/datasets/BonesSeedA2/soma_uniform, from docker/.env
SRC_FPS = 120.0
SOLE_BELOW_ANKLE = 0.059  # toe_roll origin to foot contact spheres, from a2_serial.xml
LIMITS = {"joint_pos": 3.0, "joint_vel": 20.0, "body_z": 2.5, "lin_vel": 10.0, "ang_vel": 30.0}
FLIP_RAD_S = 40.0  # raw 120 fps joint step above this (>19 deg in one frame) = IK solution switch -> cut
CUT_PAD = 3  # frames dropped on each side of a cut
MAX_JOINT_SPEED = 18.0  # rad/s clamp on the resampled reference (loader rejects >= 20)

_MODEL = None


def _model(xml: str) -> mujoco.MjModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = mujoco.MjModel.from_xml_path(xml)
    return _MODEL


def csv_column(joint: str) -> str:
    """idx05_left_toe_pitch -> left_toe_pitch_dof"""
    return joint.split("_", 1)[1] + "_dof"


def resample(qpos: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    t_src = np.arange(len(qpos)) / src_fps
    t_dst = np.minimum(np.arange(0.0, t_src[-1] + 1e-9, 1.0 / dst_fps), t_src[-1])
    out = np.empty((len(t_dst), qpos.shape[1]))
    for c in list(range(0, 3)) + list(range(7, qpos.shape[1])):
        out[:, c] = np.interp(t_dst, t_src, qpos[:, c])
    rot = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])  # wxyz -> xyzw
    out[:, [4, 5, 6, 3]] = Slerp(t_src, rot)(t_dst).as_quat()
    return out


def clamp_joint_speed(q: np.ndarray, fps: float, vmax: float) -> np.ndarray:
    """Limit per-step hinge change to vmax/fps (forward pass). Touches only the few frames where
    the reference outruns the loader's 20 rad/s limit; the lag recovers once the joint slows."""
    out = q.copy()
    step = vmax / fps
    for t in range(1, len(out)):
        out[t, 7:] = out[t - 1, 7:] + np.clip(q[t, 7:] - out[t - 1, 7:], -step, step)
    return out


LEG_JOINTS = {
    "left_toe_roll": ["idx01_left_hip_roll", "idx02_left_hip_yaw", "idx03_left_hip_pitch",
                      "idx04_left_tarsus", "idx05_left_toe_pitch", "idx06_left_toe_roll"],
    "right_toe_roll": ["idx07_right_hip_roll", "idx08_right_hip_yaw", "idx09_right_hip_pitch",
                       "idx10_right_tarsus", "idx11_right_toe_pitch", "idx12_right_toe_roll"],
}
W_ROT = 0.3  # m per rad in the foot pose error
IK_MAX_ITERS = 10
IK_TOL_POS = 1e-3  # m
IK_TOL_ROT = 5e-3  # rad
IK_FALLBACK_POS = 0.02  # m: foot error above this -> keep the raw leg for that frame


def resolve_legs(m: mujoco.MjModel, d: mujoco.MjData, qpos: np.ndarray) -> np.ndarray:
    """Re-solve leg joints so each foot (toe_roll body) matches its retargeted pose, warm-started
    from the previous frame. The retarget keeps the foot trajectory but its knee sometimes sits
    pinned at the -5 deg limit and flips branch; a continuous solve removes that."""
    legs = []
    for body, names in LEG_JOINTS.items():
        jids = [m.joint(n).id for n in names]
        legs.append((m.body(body).id, m.jnt_qposadr[jids], m.jnt_dofadr[jids], m.jnt_range[jids]))
    T = len(qpos)
    tpos = np.empty((T, len(legs), 3))
    tquat = np.empty((T, len(legs), 4))
    for t in range(T):
        d.qpos[:] = qpos[t]
        mujoco.mj_kinematics(m, d)
        for i, (b, *_rest) in enumerate(legs):
            tpos[t, i] = d.xpos[b]
            tquat[t, i] = d.xquat[b]
    out = qpos.copy()
    prev = [qpos[0, qa].copy() for _, qa, _, _ in legs]
    for p in prev:
        p[3] = max(p[3], 0.4)  # start from a bent knee
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    er = np.zeros(3)
    damp = 1e-4 * np.eye(6)
    for t in range(T):
        d.qpos[:] = qpos[t]
        for p, (_, qa, _, _) in zip(prev, legs):
            d.qpos[qa] = p
        for _ in range(IK_MAX_ITERS):
            mujoco.mj_kinematics(m, d)
            errs = []
            for i, (b, _, _, _) in enumerate(legs):
                ep = tpos[t, i] - d.xpos[b]
                mujoco.mju_subQuat(er, tquat[t, i], d.xquat[b])  # body frame: q_cur * quat(er) = q_tgt
                ew = d.xmat[b].reshape(3, 3) @ er
                errs.append(None if np.linalg.norm(ep) <= IK_TOL_POS and np.linalg.norm(ew) <= IK_TOL_ROT
                            else np.concatenate([ep, W_ROT * ew]))
            if all(e is None for e in errs):
                break
            mujoco.mj_comPos(m, d)
            for (b, qa, va, rng), err in zip(legs, errs):
                if err is None:
                    continue
                mujoco.mj_jacBody(m, d, jp, jr, b)
                jac = np.vstack([jp[:, va], W_ROT * jr[:, va]])
                dq = jac.T @ np.linalg.solve(jac @ jac.T + damp, err)
                d.qpos[qa] = np.clip(d.qpos[qa] + dq, rng[:, 0], rng[:, 1])
        mujoco.mj_kinematics(m, d)
        for i, (b, qa, _, _) in enumerate(legs):
            # Target not reachable from the continuous branch (e.g. deep crouch): fall back to the
            # raw leg, which matches its foot exactly; any resulting jump is cut later.
            if np.linalg.norm(tpos[t, i] - d.xpos[b]) > IK_FALLBACK_POS:
                prev[i] = qpos[t, qa].copy()
            else:
                prev[i] = d.qpos[qa].copy()
            out[t, qa] = prev[i]
    return out


def clip_stats(m: mujoco.MjModel, d: mujoco.MjData, q: np.ndarray, fps: float) -> dict:
    """Same quantities MimicLite's dataset validator checks (bodies exclude world)."""
    nb = m.nbody - 1
    pos = np.empty((len(q), nb, 3))
    quat = np.empty((len(q), nb, 4))
    for t in range(len(q)):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(m, d)
        pos[t] = d.xpos[1:]
        quat[t] = d.xquat[1:]
    r0 = Rotation.from_quat(quat[:-1].reshape(-1, 4)[:, [1, 2, 3, 0]])
    r1 = Rotation.from_quat(quat[1:].reshape(-1, 4)[:, [1, 2, 3, 0]])
    return {
        "joint_pos": float(np.abs(q[:, 7:]).max()),
        "joint_vel": float(np.abs(np.diff(q[:, 7:], axis=0)).max() * fps),
        "body_z": float(pos[..., 2].max()),
        "lin_vel": float(np.linalg.norm(np.diff(pos, axis=0), axis=-1).max() * fps),
        "ang_vel": float((r1 * r0.inv()).magnitude().max() * fps),
    }


def convert_one(job: tuple[str, str, str, float, float]) -> dict:
    csv_path, out_path, xml, fps, min_seconds = job
    name = Path(csv_path).name
    try:
        m = _model(xml)
        d = mujoco.MjData(m)
        joints = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        if len(df) < 2:
            return {"name": name, "status": "reject", "reason": "too_short"}
        qpos = np.zeros((len(df), m.nq))
        qpos[:, 0:3] = df[["root_translateX", "root_translateY", "root_translateZ"]].to_numpy() / 100.0
        euler = df[["root_rotateX", "root_rotateY", "root_rotateZ"]].to_numpy()
        qpos[:, 3:7] = Rotation.from_euler("xyz", euler, degrees=True).as_quat()[:, [3, 0, 1, 2]]
        qpos[:, 7:] = np.deg2rad(df[[csv_column(j) for j in joints]].to_numpy())
        if not np.isfinite(qpos).all():
            return {"name": name, "status": "reject", "reason": "non_finite"}

        ankles = [m.body("left_toe_roll").id, m.body("right_toe_roll").id]
        low = []
        for t in range(0, len(qpos), 4):
            d.qpos[:] = qpos[t]
            mujoco.mj_kinematics(m, d)
            low.append(min(d.xpos[ankles[0], 2], d.xpos[ankles[1], 2]))
        offset = float(np.clip(SOLE_BELOW_ANKLE - np.percentile(low, 10), -0.05, 0.10))
        qpos[:, 2] += offset
        qpos = resolve_legs(m, d, qpos)

        # Remaining single-frame jumps are whole-frame glitches in the source (root, feet and arms
        # all step ~2x at once): cut there and keep the valid pieces as separate motions.
        step = np.abs(np.diff(qpos[:, 7:], axis=0)).max(axis=1) * SRC_FPS
        cuts = np.flatnonzero(step > FLIP_RAD_S) + 1
        bounds = np.concatenate([[0], cuts, [len(qpos)]])
        segments = [(a + (CUT_PAD if a else 0), b - (CUT_PAD if b < len(qpos) else 0))
                    for a, b in zip(bounds[:-1], bounds[1:])]
        segments = [(a, b) for a, b in segments if (b - a) / SRC_FPS >= min_seconds]

        stem = out_path[: -len(".npz")]
        saved, rejected = [], []
        for k, (a, b) in enumerate(segments):
            q = clamp_joint_speed(resample(qpos[a:b], SRC_FPS, fps), fps, MAX_JOINT_SPEED)
            stats = clip_stats(m, d, q, fps)
            bad = [key for key, v in stats.items() if not np.isfinite(v) or v >= LIMITS[key]]
            if bad:
                rejected.append({"seg": k, "reason": ",".join(bad), "stats": stats})
                continue
            dst = out_path if len(segments) == 1 and len(cuts) == 0 else f"{stem}__s{k}.npz"
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            tmp = dst + ".tmp.npz"
            np.savez_compressed(tmp, qpos=q.astype(np.float32))
            os.replace(tmp, dst)
            saved.append(len(q))
        if not saved:
            return {"name": name, "status": "reject", "reason": "no_valid_segment", "cuts": len(cuts),
                    "segments": len(segments), "rejected": rejected}
        return {"name": name, "status": "ok", "frames": int(sum(saved)), "segments_saved": len(saved),
                "cuts": len(cuts), "kept_frac": float(sum(saved) / (len(qpos) * fps / SRC_FPS)),
                "offset": offset, "rejected": rejected}
    except Exception as exc:  # one bad CSV must not stop the chunk
        return {"name": name, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def s3_dates(s3: str) -> list[str]:
    out = subprocess.run(["aws", "s3", "ls", s3.rstrip("/") + "/"], check=True, capture_output=True, text=True)
    return sorted(line.split()[-1].rstrip("/") for line in out.stdout.splitlines() if line.strip().startswith("PRE"))


def write_manifest(out: Path, joints: list[str], fps: float, s3: str) -> dict:
    files = sorted((out / "motions").rglob("*.npz"))
    frames = 0
    for f in files:
        with np.load(f) as z:
            frames += z["qpos"].shape[0]
    payload = {
        "format_version": 2,
        "dataset_name": "bonesseed_a2",
        "mjcf": "mjcf/a2_kin.xml",
        "motions_subdir": "motions",
        "timestep": 1.0 / fps,
        "qpos_dim": 7 + len(joints),
        "qpos_names": ["root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz", *joints],
        "num_motions": len(files),
        "source": {"s3": s3, "src_fps": SRC_FPS, "root": "cm + euler xyz deg -> m + quat wxyz",
                   "height_fix": "p10 lowest ankle -> 0.059 m", "mirrors_included": True},
        "total_hours": frames / fps / 3600.0,
    }
    (out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--mjcf-dir", required=True, help="dir with a2_kin.xml + meshes/ (model/build_a2_serial.py output)")
    p.add_argument("--s3", default=DEFAULT_S3, required=not DEFAULT_S3, help="source prefix (env A2_S3_URI)")
    p.add_argument("--dates", default="", help="comma list of date folders (default: all)")
    p.add_argument("--limit", type=int, default=0, help="max clips per date folder (testing)")
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--scratch", default=tempfile.gettempdir())
    args = p.parse_args()

    out = Path(args.out).resolve()
    (out / "motions").mkdir(parents=True, exist_ok=True)
    mjcf_dst = out / "mjcf"
    if not (mjcf_dst / "a2_kin.xml").is_file():
        shutil.copytree(args.mjcf_dir, mjcf_dst, dirs_exist_ok=True)
    xml = str(mjcf_dst / "a2_kin.xml")
    m = mujoco.MjModel.from_xml_path(xml)
    joints = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]

    report_path = out / "conversion_report.jsonl"
    dates = [s for s in args.dates.split(",") if s] or s3_dates(args.s3)
    print(f"{len(dates)} date folders, workers={args.workers}, out={out}", flush=True)
    totals = {"ok": 0, "skip": 0, "reject": 0, "error": 0}
    t0 = time.time()
    for i, date in enumerate(dates):
        chunk = Path(tempfile.mkdtemp(prefix=f"a2csv_{date}_", dir=args.scratch))
        try:
            cmd = ["aws", "s3", "sync", "--only-show-errors", f"{args.s3.rstrip('/')}/{date}/", str(chunk)]
            if args.limit:
                listing = subprocess.run(["aws", "s3", "ls", f"{args.s3.rstrip('/')}/{date}/"], check=True,
                                         capture_output=True, text=True).stdout.split("\n")
                names = [l.split()[-1] for l in listing if l.endswith(".csv")][: args.limit]
                cmd += ["--exclude", "*"] + sum((["--include", n] for n in names), [])
            subprocess.run(cmd, check=True)
            jobs = []
            for csv in sorted(chunk.glob("*.csv")):
                dst = out / "motions" / date / (csv.stem + ".npz")
                if dst.is_file() or any(dst.parent.glob(f"{csv.stem}__s*.npz")):
                    totals["skip"] += 1
                    continue
                jobs.append((str(csv), str(dst), xml, args.fps, args.min_seconds))
            with ProcessPoolExecutor(max_workers=args.workers) as pool, report_path.open("a") as rep:
                for res in pool.map(convert_one, jobs, chunksize=4):
                    res["date"] = date
                    totals[res["status"]] += 1
                    if res["status"] != "ok":
                        rep.write(json.dumps(res) + "\n")
        except subprocess.CalledProcessError as exc:
            print(f"[{date}] S3 sync failed ({exc}); stopping so the run can be resumed", file=sys.stderr, flush=True)
            raise
        finally:
            shutil.rmtree(chunk, ignore_errors=True)
        print(f"[{i + 1}/{len(dates)}] {date}: totals {totals} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    payload = write_manifest(out, joints, args.fps, args.s3)
    print(f"done: {payload['num_motions']} motions, {payload['total_hours']:.1f} h, totals {totals}", flush=True)


if __name__ == "__main__":
    main()
