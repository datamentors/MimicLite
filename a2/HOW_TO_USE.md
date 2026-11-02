# A2 MimicLite — how to use

MimicLite motion tracker for the AgiBot A2 (Ultra), trained with mjlab inside Docker.
Lives in `a2/` of the datamentors/MimicLite fork; copy/sync it to the training host (e.g. `~/a2-mimiclite/`).
Private values (AWS profile, S3 source, W&B key) go in `docker/.env` - copy `docker/.env.example`, never commit it.

## Layout (host)
| Dir | Content |
|---|---|
| `docker/` | `Dockerfile`, `compose.yaml`, env `pyproject.toml` (+ `uv.lock` once exported) |
| `model/` | `build_a2_serial.py`, `stand_test.py`, built `a2_serial/` MJCF + meshes |
| `overlay/` | A2 files copied into mimic-lite: `assets/a2.py`, `cfg/task/robot/a2.yaml`, `cfg/task/motion/a2/default.yaml`, `cfg/stages/smoke.yaml` |
| `tools/convert_bonesseed_a2.py` | S3 CSV -> MimicLite motions |
| `scripts/` | `convert.sh`, `smoke.sh`, `train.sh` (run inside the container) |
| `motions/` | converted datasets (only thing kept from S3) |
| `runs/` | training runs: `stages/01-train`, `02-adapt`, `03-finetune` |
| `cache/` | any4hdmi FK cache, warp / torch compile caches (safe to delete) |
| `logs/` | conversion / smoke / build logs, `task-logs.md` |

## Commands (on the training host, from the a2/ copy)
```bash
# build image (pinned active-adaptation b6a6b3f, mimic-lite 3253810)
docker build -f docker/Dockerfile -t a2-mimiclite:latest .

# 1. convert motions (needs AWS SSO on the host: aws sso login --profile $AWS_PROFILE --use-device-code)
docker compose -f docker/compose.yaml run --rm convert
#    test on a few clips:  -e DATES=210531 -e LIMIT=20 -e OUT=/data/motions/test_a2

# 2. smoke test (1 GPU, ~a few minutes)
GPUS=0 docker compose -f docker/compose.yaml run --rm smoke

# 3. full training (4 GPUs, Train 4000 / Adapt 1000 / Finetune 2000)
docker compose -f docker/compose.yaml run -d --name a2_train train
docker logs -f a2_train
#    options: -e MODULE=huge -e NUM_ENVS=8192 -e RUN_NAME=a2_v1 -e WANDB_API_KEY=... -e WANDB_MODE=online

# shell inside the image
docker compose -f docker/compose.yaml run --rm shell
```

## Export / play (inside the shell)
```bash
cd /opt/aa
python projects/mimic-lite/scripts/play.py \
  --config-path /runs/<run>/stages/03-finetune/.hydra --config-name config \
  checkpoint_path=/runs/<run>/stages/03-finetune/checkpoint_latest.pt \
  backend=mjlab task.num_envs=16 task.termination.root_pos_error.enabled=false \
  export_policy=true ++export_only=true
```

## What the data pipeline does
`BonesSeedA2/soma_uniform` CSVs (120 fps, 142,214 clips incl. `_M` mirrors, read-only) ->
1. root cm/deg -> m/quat (extrinsic xyz), head dropped (welded)
2. height fix: stance ankle at the 0.059 m sole height (retarget leaves feet 1-3 cm in the floor)
3. leg joints re-solved to the retargeted foot poses (the retarget pins the knee at -5 deg and flips branch)
4. cut at whole-frame source glitches (> 40 rad/s), segments < 1 s dropped
5. resample to 50 fps, 18 rad/s joint-speed clamp, MimicLite validator limits checked per segment
Report per clip: `motions/bonesseed_a2/conversion_report.jsonl` (non-ok clips only).

## Known estimates (no AgiBot data)
- Armature and joint friction: estimates in `a2.py`, randomized 0.5-1.5x / 0.05-0.25 in `a2.yaml`.
- Arm torque limits (60/40/20/10 Nm) are estimates; legs from the Ultra URDF (120/140/50 Nm).
- Ankle roll reference is always 0 in the data: not rewarded, not fed to the policy.
- Collision: AgiBot model has only foot points, torso and hands -> no limb self-collision signal
  in `a2_serial`. Use the self-collision variant below.

## Self-collision variant (`a2_serial_selfcol`, image `a2-mimiclite:selfcol`)
`build_a2_serial.py --self-collision` adds 20 shapes (torso = 3 stacked boxes; head, thigh, shank,
foot, shoulder, upper arm, elbow, forearm, hand capsules) fitted to the meshes, sitting ~1 cm inside
the surface so an arm resting against the torso is not a hit. They collide only with each other
(contype = conaffinity = 2; the floor is 1), so floor contact and foot friction are the same as
`a2_serial`. Body pairs <= 3 links apart are excluded (they always touch at the joint).
Dataset audit (2000 random clips, every 5th frame, `logs/selfcol_audit_v3.json` on AMAX): 28.7% of
reference frames self-penetrate - mostly real retarget overlap (hands into thighs, hand into head,
forearm into torso; up to 5-10 cm), so the policy is penalized for copying those poses exactly.
In training, MimicLite's `self_collision` sensor + `self_collisions` penalty (-1 per step with a
self-contact > 1 N) now fire.
```bash
# rebuild the model (Mac, from ~/Datamentors/a2-mimiclite)
uv run --python 3.12 --with mujoco==3.11.0 --with numpy python model/build_a2_serial.py \
  --src ../A2Docs/rl_deploy/mujoco_sim/resource/model/a2_t2d0_ultra/mujoco/model.xml \
  --out model/a2_serial_selfcol --self-collision
# image (AMAX, from the source dir); :latest stays on a2_serial
docker build -f docker/Dockerfile --build-arg A2_MODEL=a2_serial_selfcol -t a2-mimiclite:selfcol .
```
Train with it: same commands as above with `image: a2-mimiclite:selfcol` (or `docker run ... a2-mimiclite:selfcol`).

### Look at the collision shapes (Mac)
```bash
V() { uv run --python 3.12 --with mujoco==3.11.0 --with numpy --with imageio --with imageio-ffmpeg "$@"; }
M=model/a2_serial_selfcol/a2_serial.xml
# interactive viewer: shapes are geom group 3 -> left panel "Group enable" -> Geom 3 on;
# Rendering -> "Contact Point" to see contacts. Drag joints with Ctrl+right-click.
V python -m mujoco.viewer --mjcf=$M
# stills of the standing pose from 4 sides (self-hits drawn red)
V python tools/view_collisions.py pose --mjcf $M --out renders/
# one motion: per-frame self-penetration stats, optional video with hits in red
V python tools/view_collisions.py motion --mjcf $M --motion clip.npz --mp4 clip.mp4
# replay a motion in the interactive viewer (macOS needs mjpython)
V mjpython tools/view_collisions.py play --mjcf $M --motion clip.npz
# dataset audit (AMAX, inside the selfcol container): share of clips/frames that self-penetrate
python /opt/a2/tools/view_collisions.py audit --mjcf $A2_MJCF --dataset /data/motions/bonesseed_a2 --n 2000
```
