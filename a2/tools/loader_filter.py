"""Post-filter a converted A2 dataset with MimicLite's own loader and validator limits.

The converter checks the validator quantities itself, but the loader derives body velocities its
own way (edge frames, fast hand swings can differ). This pass loads the dataset exactly as training
does, finds frames that break mimic_lite/tasks/motion.py:_validate_motion_dataset, splits the
affected motion files there (dropping +-PAD frames and pieces shorter than --min-seconds), rewrites
the manifest and repeats until the dataset validates. Run inside the container.

usage: python loader_filter.py /data/motions/bonesseed_a2 [--min-seconds 1.0] [--rounds 3]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/opt/aa/projects/mimic-lite")
from mimic_lite.tasks.motion import _motion_dataset_field_chunk, create_dataset_from_path  # noqa: E402

PAD = 2
CHUNK = 131072


def bad_frames(ds) -> torch.Tensor:
    """Boolean per global frame, same limits as _validate_motion_dataset."""
    n = int(ds.ends.max())
    out = torch.zeros(n, dtype=torch.bool)
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        f = lambda name: _motion_dataset_field_chunk(ds, field_name=name, start=s, end=e)  # noqa: E731
        jp, jv = f("joint_pos").abs(), f("joint_vel").abs()
        z = f("body_pos_w")[..., 2]
        lv = torch.linalg.vector_norm(f("body_lin_vel_w"), dim=-1)
        av = torch.linalg.vector_norm(f("body_ang_vel_w"), dim=-1)
        qe = (torch.linalg.vector_norm(f("body_quat_w"), dim=-1) - 1.0).abs()
        bad = ((~torch.isfinite(jp)) | (jp >= 3.0)).any(1) | ((~torch.isfinite(jv)) | (jv >= 20.0)).any(1)
        bad |= ((~torch.isfinite(z)) | (z >= 2.5)).any(1) | ((~torch.isfinite(lv)) | (lv >= 10.0)).any(1)
        bad |= ((~torch.isfinite(av)) | (av >= 30.0)).any(1) | ((~torch.isfinite(qe)) | (qe > 1e-3)).any(1)
        out[s:e] = bad
    return out


def rewrite_manifest(root: Path) -> None:
    man = json.loads((root / "manifest.json").read_text())
    files = sorted((root / "motions").rglob("*.npz"))
    frames = 0
    for p in files:
        with np.load(p) as z:
            frames += z["qpos"].shape[0]
    man["num_motions"] = len(files)
    man["total_hours"] = frames * man["timestep"] / 3600.0
    (root / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    timestep = json.loads((root / "manifest.json").read_text())["timestep"]
    min_len = int(round(args.min_seconds / timestep))
    log = (root / "loader_filter_report.jsonl").open("a")

    # Round 0 loads the full dataset (one full FK-cache build). Later rounds only re-check the
    # pieces written in the previous round: a motion's validity does not depend on the others, and
    # every full reload would rebuild the whole cache (~80 min for BonesSeedA2).
    cache_root = Path(os.environ.get("ANY4HDMI_QPOS_CACHE_ROOT", "/cache/any4hdmi"))
    caches_before = set(cache_root.glob("*")) if cache_root.is_dir() else set()
    targets: str | list[str] = str(root)
    changed = False
    for rnd in range(args.rounds):
        ds = create_dataset_from_path(targets)
        bad = bad_frames(ds)
        starts, ends = ds.starts.cpu().tolist(), ds.ends.cpu().tolist()
        paths = [Path(p) for p in ds.motion_paths]
        n_bad = int(bad.sum())
        print(f"round {rnd}: {len(paths)} motions, {int(ds.ends.max())} frames, {n_bad} invalid frames", flush=True)
        if n_bad == 0:
            break
        changed = True
        new_files: list[str] = []
        touched = 0
        for k, (s, e, path) in enumerate(zip(starts, ends, paths)):
            local = torch.nonzero(bad[s:e]).flatten().tolist()
            if not local:
                continue
            qpos = np.load(path)["qpos"]
            if len(qpos) != e - s:
                raise RuntimeError(f"{path}: file has {len(qpos)} frames, loader {e - s} (fps mismatch?)")
            keep = np.ones(len(qpos), dtype=bool)
            for f in local:
                keep[max(0, f - PAD): f + PAD + 1] = False
            pieces, cur = [], []
            for i, ok in enumerate(keep):
                if ok:
                    cur.append(i)
                elif cur:
                    pieces.append(cur); cur = []
            if cur:
                pieces.append(cur)
            pieces = [p for p in pieces if len(p) >= min_len]
            stem = path.with_suffix("")
            for j, p in enumerate(pieces):
                dst = Path(f"{stem}_f{rnd}{j}.npz")
                tmp = Path(str(dst) + ".tmp.npz")
                np.savez_compressed(tmp, qpos=qpos[p])
                os.replace(tmp, dst)
                new_files.append(str(dst))
            path.unlink()
            touched += 1
            log.write(json.dumps({"round": rnd, "motion": str(path.relative_to(root)), "bad_frames": local,
                                  "pieces_kept": [len(p) for p in pieces], "frames_before": len(qpos)}) + "\n")
        log.flush()
        rewrite_manifest(root)
        print(f"round {rnd}: split/dropped {touched} motions into {len(new_files)} pieces", flush=True)
        if not new_files:
            break
        targets = new_files
    else:
        raise SystemExit("pieces still invalid after all rounds")
    if changed:
        # The full-dataset cache from round 0 is keyed by the old file set: stale now. Subset
        # caches from the re-check rounds are stale too. Training rebuilds the final one once.
        for p in (set(cache_root.glob("*")) - caches_before):
            shutil.rmtree(p, ignore_errors=True)
            print(f"removed stale cache {p}", flush=True)
    print(f"loader filter done (dataset {'changed' if changed else 'unchanged'})", flush=True)


if __name__ == "__main__":
    main()
