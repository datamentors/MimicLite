# A2 VR teleop in sim (PICO -> GMR -> A2 MimicLite policy)

MimicLite's G1 teleop path, made to drive the AgiBot A2 instead:

```
PICO 4 + leg trackers --XRoboToolkit PC Service--> pico_retarget_pub.py (GMR: xrobot -> agibot_a2)
   --ZMQ :28701 motion stream / :5592 buttons--> tracking.py (A2 MimicLite policy) --> base_sim.py (MuJoCo A2)
```

The `sim2real` submodule is **not modified**. `a2_sim2real.py` registers A2 at runtime and then runs the
unchanged sim2real entry point:
- RobotCfg `a2`: joints, limits and actuator values from `a2/overlay/mimic_lite/assets/a2.py`
- GMR target `agibot_a2` with the operator's T-pose calibration `<a2>/../calib/xrobot_to_a2_tpose.json`
  (pico_retarget_pub hardcodes `unitree_g1`; with `--robot a2` the GMR class is swapped)
- sim model `a2/model/a2_serial/a2_serial_sim.xml` (generated, git-ignored): the training MJCF with a
  named free joint and without its own floor assets (sim2real adds its scene); the injected floor is made
  collidable (A2's main default class zeroes contype/conaffinity)

## Self-collision policy
Policies trained with image `a2-mimiclite:selfcol` (model `a2_serial_selfcol`: same bodies/joints + 20
self-collision shapes) run the same way with the matching sim model. Export with that image (`docker run ...
a2-mimiclite:selfcol`, same play.py command), then set in **every** terminal (sim, retarget, policy):
```bash
export A2_MODEL=a2_serial_selfcol A2_POLICY_DIR=<checkpoints>/a2-huge-selfcol-13000
```
Checked 2026-10-28 (huge selfcol ckpt 13000): wave_a2 4.4 cm, retargeted LAFAN dance1 3.8 cm, no falls.

## Calibration (per operator, required)
The T-pose calibration is the only GMR config: there is no generic fallback. Calibrate once per operator
(again if the trackers are worn differently); `retarget`, `fake-pico` and `check_retarget.py` all use it.
`teleop.sh calibrate <height>` records a T-pose (`tools/record_pico.py`) and runs
`tools/make_xrobot_to_a2.py --tpose`: it replays the recording through pico_retarget_pub's own preprocessing
(Unity -> z-up, pelvis heading removed, floor alignment), straightens it to an ideal T-pose and sets every
rotation **and position** offset so that T-pose lands exactly on A2's zero pose (also a T-pose).
Check: the straightened T-pose retargets to A2 joints within 0.001 rad, torso 0.987 m, tilt 0.
Output `<a2>/../calib/xrobot_to_a2_tpose_<ts>.json` (+ `xrobot_to_a2_tpose.json` link to the latest), used by
`teleop.sh retarget` automatically; `A2_GMR_CONFIG=<json>` picks another (e.g. an earlier operator's
`xrobot_to_a2_tpose_<ts>.json`), `A2_CALIB_DIR` moves the directory. Pass `retarget` the same height as
`calibrate` (stored as `_calibration.height_m`). IK weights are copied from GMR's `xrobot_to_g1.json`. Note: in GMR's input frame the operator faces +y (the publisher
removes the PICO pelvis frame's yaw, not the facing direction); the calibration measures and keeps that.

## Files
| File | What |
|---|---|
| `teleop.sh` | launcher: `service`, `sim`, `retarget`, `policy`, `fake-pico` |
| `a2_sim2real.py` | runtime A2 registration + `python a2_sim2real.py <sim2real script.py> --robot a2 ...` |
| `tools/make_xrobot_to_a2.py` | T-pose recording -> calibrated GMR config (run by `teleop.sh calibrate`; derivation in its docstring) |
| `tools/check_retarget.py` | offline check without a headset: G1 clip -> synthetic PICO frames -> A2, metrics + A2 dataset clip |
| `tools/record_pico.py` | records raw PICO body tracking (works next to a running publisher) |
| `tools/fake_pico.py` | runs the real `pico_retarget_pub.py` with a fake PICO (G1 clip as body data, scripted buttons) |

## Setup (once)
Needs a MimicLite checkout whose sim2real venvs exist (`sim2real/.venv` and `sim2real/venv/teleop/.venv`,
built by MimicLite's own setup) - used read-only. Default `MIMICLITE_ROOT=~/MimicLite`.

Policy: export an A2 checkpoint to `policy.onnx` + `policy.yaml` (from the a2/ Docker image, see
`a2/HOW_TO_USE.md`; run the container as root because play.py writes under /opt/aa):
```bash
cd <a2 training copy>    # has docker/compose.yaml, runs/<ckpt>.pt, motions/wave_a2
A2_UID=$(id -u) A2_GID=$(id -g) GPUS=0 docker compose -f docker/compose.yaml run --rm -u 0 \
  -e A2_MOTION_DATASET=/data/motions/wave_a2 shell bash -lc "cd /opt/aa && \
  python projects/mimic-lite/scripts/play.py task=tracking-base task/robot=a2 backend=mjlab +exp=ppo/train \
    algo/ppo/module=huge task.num_envs=1 checkpoint_path=/runs/<ckpt>.pt headless=true \
    export_policy=true ++export_only=true hydra.run.dir=/runs/play/export hydra.job.chdir=true && \
  mkdir -p /runs/exports && cp projects/mimic-lite/scripts/exports/*/policy-* /runs/exports/ && chown -R $(id -u):$(id -g) /runs"
# then: <checkpoints>/<name>/policy.onnx + policy.yaml  (A2_POLICY_DIR, default <a2>/../checkpoints/a2-huge-12600)
```
GMR config (after changing the generator, re-run on an existing recording instead of re-recording):
```bash
cd $MIMICLITE_ROOT && sim2real/venv/teleop/.venv/bin/python <a2>/teleop/tools/make_xrobot_to_a2.py \
  --tpose <a2>/../calib/tpose_<ts>.npz --height 1.75 --out <a2>/../calib/xrobot_to_a2_tpose_<ts>.json
```

## Run (one terminal each, in order)
```bash
a2/teleop/teleop.sh service          # XRoboToolkit PC Service; PICO app -> connect to this machine
a2/teleop/teleop.sh sim              # MuJoCo A2, viewer URL printed (http://<host>:808x)
a2/teleop/teleop.sh calibrate 1.75   # once per operator: 5 s countdown, hold a T-pose for 10 s
a2/teleop/teleop.sh retarget 1.75    # operator height [m]; PICO whole-body tracking with leg trackers
a2/teleop/teleop.sh policy
```
PICO buttons: **A** stand pose -> **A+B** policy on -> **X** start/pause following you -> **B** hold.
In the sim viewer, untick the elastic band once the policy follows. For A2 the band is scaled by the
A2/G1 mass ratio (x2.09, stiffness 418 / damping 209) so A2 hangs like G1 does (torso ~1.34 m, feet ~0.3 m
up); override with `A2_BAND_SCALE=<factor> teleop.sh sim`.

No headset: use `a2/teleop/teleop.sh fake-pico [lafan clip]` instead of `retarget` - it presses A, A+B, X
by itself (t = 3, 6, 9 s) and streams a G1 LAFAN clip as PICO body data through the real publisher.

Offline retarget check (no sim):
```bash
cd $MIMICLITE_ROOT && sim2real/venv/teleop/.venv/bin/python <a2>/teleop/tools/check_retarget.py \
  --motion any4hdmi/output/g1/lafan/motions/dance1_subject1.npz --manifest any4hdmi/output/g1/lafan/manifest.json \
  --out <a2>/teleop/out/lafan_a2
# policy on that clip, one process, headless:
cd $MIMICLITE_ROOT/sim2real && .venv/bin/python <a2>/teleop/a2_sim2real.py sim2real/sim_env/integrated_sim2sim.py \
  --robot a2 --policy_config <A2_POLICY_DIR>/policy.yaml --motion_path <a2>/teleop/out/lafan_a2/motions/dance1_subject1_a2.npz \
  --headless --run_once --trajectory_output /tmp/traj.npz
```

## Status / tuning (2026-10-28, policy huge ckpt 12600)
- Offline retarget (G1 LAFAN -> synthetic PICO -> A2): no joint-limit violations, knee/hip-pitch corr
  0.94-0.997 vs G1, feet on the floor (lowest sole -6..+13 mm).
- Policy on A2 training clips (integrated sim): body error 4.5-7.6 cm. On retargeted LAFAN: dance1 4.0 cm;
  walk1 falls at ~18 s.
- 2026-10-29, T-pose-only (calib `tpose_20261028_175751`), the offline numbers above were the removed
  G1-derived config: walk1 600 frames arm direction cos 0.91-0.996 (was 0.74-0.89), knee/hip-pitch corr
  0.93-0.985 but right knee mean|diff| 0.38 rad, lowest sole -36 mm. The synthetic G1 human is not the
  calibrated operator, so `check_retarget` / `fake-pico` are approximate checks now.
- A2 shoulder adducts only ~5 deg past vertical: arms hanging close to the body are not reachable.
- Not tested with a real headset yet.
- `npz_pub.py` cannot stream A2 (hardcodes root body "pelvis"); use `fake-pico`.
