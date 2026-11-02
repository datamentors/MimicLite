"""Look at the A2 self-collision shapes (build_a2_serial.py --self-collision) and find self-contacts.

Self-collision shapes are the geoms with contype bit 2 (names *_collision). A "hit" is a contact
between two of them with penetration (dist < 0) - in training this is what the self_collisions
penalty and the self_collision sensor react to.

  pose    render the INIT pose (a2.py INIT_JOINT_POS) from 4 sides, shapes visible, hits in red
  motion  replay one motion file kinematically: per-frame hits, worst pairs, optional mp4
  audit   sample motions from a dataset: how many frames/clips would self-collide
  play    interactive viewer replaying a motion (macOS: run with `mjpython`)

usage:
  python view_collisions.py pose   --mjcf model/a2_serial_selfcol/a2_serial.xml --out renders/
  python view_collisions.py motion --mjcf ... --motion clip.npz [--mp4 out.mp4] [--fps 50]
  python view_collisions.py audit  --mjcf ... --dataset motions/bonesseed_a2 [--n 2000] [--seed 0]
  mjpython view_collisions.py play --mjcf ... --motion clip.npz
Motion files: any4hdmi npz with qpos [T, 7 + 26] (root xyz, quat wxyz, joints in MJCF order).
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
from pathlib import Path
import random
import time

import mujoco
import numpy as np

A2_PY = Path(__file__).resolve().parents[1] / "overlay/mimic_lite/assets/a2.py"
SELF_BIT = 2
HIT_RGBA = np.array([1.0, 0.05, 0.05, 0.85])


def init_qpos(m: mujoco.MjModel) -> np.ndarray:
    """INIT_POS / INIT_JOINT_POS literals from a2.py (parsed, a2.py needs the training env)."""
    consts = {}
    for node in ast.parse(A2_PY.read_text()).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in ("INIT_POS", "INIT_JOINT_POS"):
            consts[node.targets[0].id] = ast.literal_eval(node.value)
    q = m.qpos0.copy()
    q[:3] = consts["INIT_POS"]
    for name, val in consts["INIT_JOINT_POS"].items():
        q[m.jnt_qposadr[m.joint(name).id]] = val
    return q


def self_geoms(m: mujoco.MjModel) -> np.ndarray:
    """Return the ids of the self-collision geoms (contype bit SELF_BIT).

    Raises:
        SystemExit: The model has no self-collision geoms.
    """
    ids = np.flatnonzero((m.geom_contype & SELF_BIT) != 0)
    if len(ids) == 0:
        raise SystemExit("no self-collision geoms (contype bit 2) - build with --self-collision")
    return ids


def hits(m: mujoco.MjModel, d: mujoco.MjData, ids: set) -> list[tuple[str, str, float]]:
    """Penetrating contacts between self-collision geoms after mj_forward: (geom, geom, depth m)."""
    out = []
    for c in d.contact[: d.ncon]:
        if c.geom1 in ids and c.geom2 in ids and c.dist < 0:
            a, b = sorted((m.geom(c.geom1).name, m.geom(c.geom2).name))
            out.append((a, b, -float(c.dist)))
    return out


def load_motion(path: Path, m: mujoco.MjModel) -> np.ndarray:
    """Load the qpos array of an any4hdmi npz motion as float64.

    Raises:
        SystemExit: The qpos width does not match the model's nq.
    """
    qpos = np.load(path)["qpos"].astype(np.float64)
    if qpos.shape[1] != m.nq:
        raise SystemExit(f"{path}: qpos width {qpos.shape[1]} != model nq {m.nq}")
    return qpos


def color_hits(m: mujoco.MjModel, base_rgba: np.ndarray, frame_hits) -> None:
    """Restore base_rgba on all geoms, then paint both geoms of every hit in HIT_RGBA."""
    m.geom_rgba[:] = base_rgba
    for a, b, _ in frame_hits:
        m.geom_rgba[m.geom(a).id] = HIT_RGBA
        m.geom_rgba[m.geom(b).id] = HIT_RGBA


def make_renderer(m: mujoco.MjModel, size=(720, 960)):
    """Create an offscreen renderer with collision group 3 visible and a camera tracking the torso.

    Enlarges the model's offscreen buffer if needed.

    Args:
        m: Model to render.
        size: (height, width) in pixels.

    Returns:
        (renderer, scene option, camera).
    """
    m.vis.global_.offheight = max(m.vis.global_.offheight, size[0])  # default buffer is 640x480
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, size[1])
    renderer = mujoco.Renderer(m, *size)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1  # collision shapes live in group 3
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("torso").id
    cam.distance, cam.elevation = 2.6, -10.0
    return renderer, opt, cam


def cmd_pose(args, m, d) -> None:
    """Report self-penetrations at the INIT pose and render it from 4 sides to args.out."""
    ids = set(self_geoms(m))
    d.qpos[:] = init_qpos(m)
    mujoco.mj_forward(m, d)
    h = hits(m, d, ids)
    print(f"INIT pose: {len(ids)} self-collision shapes, penetrating pairs: {h or 'none'}")
    base = m.geom_rgba.copy()
    color_hits(m, base, h)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio  # noqa: PLC0415 - render-only optional dep

    renderer, opt, cam = make_renderer(m)
    for az in (180, 90, 0, 270):  # front, left side, back, right side
        cam.azimuth = az
        renderer.update_scene(d, cam, opt)
        iio.imwrite(out / f"init_az{az:03d}.png", renderer.render())
    print(f"wrote {out}/init_az*.png")


def scan(m, d, qpos, ids):
    """Run mj_forward on every frame and collect self-collision hits.

    Returns:
        (per-frame hit lists, Counter of frames per geom pair, max depth per geom pair).
    """
    per_frame, pairs, depth = [], collections.Counter(), collections.defaultdict(float)
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        h = hits(m, d, ids)
        per_frame.append(h)
        for a, b, dep in h:
            pairs[(a, b)] += 1
            depth[(a, b)] = max(depth[(a, b)], dep)
    return per_frame, pairs, depth


def cmd_motion(args, m, d) -> None:
    """Print per-motion hit stats and the worst pairs; optionally write an mp4 with hits in red."""
    ids = set(self_geoms(m))
    qpos = load_motion(Path(args.motion), m)
    per_frame, pairs, depth = scan(m, d, qpos, ids)
    n_hit = sum(1 for h in per_frame if h)
    print(f"{args.motion}: {len(qpos)} frames, {n_hit} with self-penetration ({100 * n_hit / len(qpos):.1f}%)")
    for (a, b), n in pairs.most_common(10):
        print(f"  {a:28s} {b:28s} frames={n:5d} max_depth={1000 * depth[(a, b)]:.0f} mm")
    if args.mp4:
        import imageio.v3 as iio  # noqa: PLC0415 - only needed with --mp4

        base = m.geom_rgba.copy()
        renderer, opt, cam = make_renderer(m, (540, 720))
        cam.azimuth = args.azimuth
        frames = []
        for q, h in zip(qpos, per_frame, strict=True):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            color_hits(m, base, h)
            renderer.update_scene(d, cam, opt)
            frames.append(renderer.render())
        iio.imwrite(args.mp4, np.stack(frames), fps=args.fps)
        print(f"wrote {args.mp4}")


def cmd_audit(args, m, d) -> None:
    """Scan a random sample of dataset motions (every --stride frame) and print a JSON hit report (also to --report)."""
    ids = set(self_geoms(m))
    files = sorted(Path(args.dataset).glob("motions/**/*.npz"))
    if not files:
        raise SystemExit(f"no motions under {args.dataset}/motions")
    random.Random(args.seed).shuffle(files)
    files = files[: args.n]
    frames = hit_frames = clips_hit = 0
    pairs, depth, worst = collections.Counter(), collections.defaultdict(float), []
    t0 = time.time()
    for i, f in enumerate(files):
        qpos = load_motion(f, m)
        per_frame, p, dep = scan(m, d, qpos[:: args.stride], ids)
        k = sum(1 for h in per_frame if h)
        frames += len(per_frame)
        hit_frames += k
        clips_hit += k > 0
        pairs.update(p)
        for key, v in dep.items():
            depth[key] = max(depth[key], v)
        worst.append((k / len(per_frame), str(f)))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)} clips, {time.time() - t0:.0f}s", flush=True)
    worst.sort(reverse=True)
    report = {
        "clips": len(files),
        "frames": frames,
        "stride": args.stride,
        "clips_with_hit_pct": round(100 * clips_hit / len(files), 2),
        "frames_with_hit_pct": round(100 * hit_frames / max(frames, 1), 2),
        "pairs": [
            {"a": a, "b": b, "frames": n, "max_depth_mm": round(1000 * depth[(a, b)], 1)}
            for (a, b), n in pairs.most_common(20)
        ],
        "worst_clips": [{"hit_frame_pct": round(100 * r, 1), "file": f} for r, f in worst[:20]],
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


def cmd_play(args, m, d) -> None:
    """Replay a motion in the interactive passive viewer, looping until the window closes."""
    import mujoco.viewer  # noqa: PLC0415 - viewer-only dep (needs a display / mjpython)

    ids = set(self_geoms(m))
    qpos = load_motion(Path(args.motion), m)
    base = m.geom_rgba.copy()
    with mujoco.viewer.launch_passive(m, d) as v:
        v.opt.geomgroup[3] = 1
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        while v.is_running():
            for q in qpos:
                if not v.is_running():
                    break
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                color_hits(m, base, hits(m, d, ids))
                v.sync()
                time.sleep(1.0 / args.fps)


def main() -> None:
    """Parse the subcommand and options, load the MJCF and dispatch to cmd_<subcommand>."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("pose", "motion", "audit", "play"))
    p.add_argument("--mjcf", required=True)
    p.add_argument("--out", default="renders")
    p.add_argument("--motion")
    p.add_argument("--mp4")
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--azimuth", type=float, default=150.0)
    p.add_argument("--dataset")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--stride", type=int, default=5, help="audit: check every k-th frame")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report")
    args = p.parse_args()
    m = mujoco.MjModel.from_xml_path(args.mjcf)
    d = mujoco.MjData(m)
    {"pose": cmd_pose, "motion": cmd_motion, "audit": cmd_audit, "play": cmd_play}[args.cmd](args, m, d)


if __name__ == "__main__":
    main()
