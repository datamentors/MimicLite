"""Keep only the latest and the best checkpoint of a MimicLite run (user rule: no duplicates).

Best = lowest mean of reward.tracking_metrics/body_pos (local body-position error, m) over the
SCORE_WINDOW iterations up to the checkpoint's iteration, read from the W&B run history (online)
or <run>/metrics.jsonl (W&B disabled). Without metrics nothing is deleted.
checkpoint_temp.pt and symlinks are never touched; files younger than MIN_AGE_S are left alone so
W&B finishes uploading them. <run>/checkpoint_best.pt is a symlink to the best file (no copy).

usage: python prune_checkpoints.py <run_dir> [--watch SECONDS] [--stop-file PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

METRIC = "reward.tracking_metrics/body_pos"
SCORE_WINDOW = 50
MIN_AGE_S = 600
CKPT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")


def checkpoints(run: Path) -> dict[int, Path]:
    out = {}
    for p in run.glob("wandb/*/files/checkpoint_*.pt"):
        m = CKPT_RE.match(p.name)
        if m and not p.is_symlink() and "latest-run" not in p.parts:
            out[int(m.group(1))] = p
    return out


def metric_rows(run: Path) -> list[tuple[int, float]]:
    local = run / "metrics.jsonl"
    if local.is_file():
        rows = []
        for line in local.read_text().splitlines():
            rec = json.loads(line)
            if METRIC in rec and rec[METRIC] is not None:
                rows.append((int(rec["iteration"]), float(rec[METRIC])))
        return rows
    runs = sorted(run.glob("wandb/run-*"))
    if not runs or not os.environ.get("WANDB_API_KEY"):
        return []
    run_id = runs[-1].name.rsplit("-", 1)[-1]
    project = os.environ.get("WANDB_PROJECT")
    try:
        import wandb

        api = wandb.Api(timeout=60)
        hist = api.run(f"{api.default_entity}/{project}/{run_id}").scan_history(keys=["_step", METRIC])
        return [(int(r["_step"]), float(r[METRIC])) for r in hist if r.get(METRIC) is not None]
    except Exception as exc:  # network / API hiccup: skip this round, never delete blind
        print(f"[prune] metrics unavailable ({type(exc).__name__}: {exc})", flush=True)
        return []


def prune_once(run: Path) -> None:
    ckpts = checkpoints(run)
    if len(ckpts) <= 2:
        return
    rows = metric_rows(run)
    if not rows:
        print(f"[prune] {len(ckpts)} checkpoints, no {METRIC} yet - keeping all", flush=True)
        return
    def score(i: int) -> float | None:
        vals = [v for s, v in rows if i - SCORE_WINDOW < s <= i]
        return sum(vals) / len(vals) if vals else None
    latest = max(ckpts)
    scored = {i: score(i) for i in ckpts if i > 0}
    scored = {i: s for i, s in scored.items() if s is not None}
    if not scored:
        return
    best = min(scored, key=scored.get)
    keep = {latest, best}
    now = time.time()
    for i, p in sorted(ckpts.items()):
        if i in keep or now - p.stat().st_mtime < MIN_AGE_S:
            continue
        print(f"[prune] delete {p.name} (score {scored.get(i)}), keep latest {latest} best {best} "
              f"({scored[best]:.4f})", flush=True)
        p.unlink()
    link = run / "checkpoint_best.pt"
    tmp = run / ".checkpoint_best.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(ckpts[best])
    os.replace(tmp, link)
    (run / "checkpoint_best.json").write_text(json.dumps(
        {"iteration": best, "metric": METRIC, "score": scored[best], "window": SCORE_WINDOW,
         "file": str(ckpts[best])}, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--watch", type=float, default=0, help="poll interval s (0 = run once)")
    ap.add_argument("--stop-file", help="exit after the round in which this file exists")
    args = ap.parse_args()
    run = Path(args.run)
    while True:
        try:
            prune_once(run)
        except Exception as exc:
            print(f"[prune] error {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if not args.watch or (args.stop_file and Path(args.stop_file).exists()):
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
