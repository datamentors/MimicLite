# MimicLite G1 — How to Run

Everything uses MimicLite's own `sim2real` stack (no SONIC, no Ardia Pulse).
All files live in `~/MimicLite` on each machine. Remove a setup with `rm -rf ~/MimicLite`.

| Machine | Address | Login | Used for |
| --- | --- | --- | --- |
| Workstation | <WORKSTATION_LAN_IP> | `ssh datamentors@<WORKSTATION_LAN_IP>` | Simulation (MuJoCo) |
| G1 Restaurant | <HOST_G1_ROBOT_RESTAURANT_IP> | `ssh -t unitree@<HOST_G1_ROBOT_RESTAURANT_IP>` | Real robot |

Policies (`sim2real/checkpoints/mimiclite-g1/`):

| Policy | Clip playback | Live Pico teleop |
| --- | --- | --- |
| `v1_1` | yes (best tracking) | **no** — needs palm bodies the live stream doesn't have |
| `roa` | yes | **yes** (default for teleop) |
| `ppo` | yes (workstation only) | yes (workstation only) |

Viewers are web pages: open the URL a script prints (`http://<machine-ip>:808x`) from any browser on the network.

---

## 1. Simulation (Workstation)

```bash
ssh datamentors@<WORKSTATION_LAN_IP>
cd ~/MimicLite
```

### 1a. Play a clip (the wave)

One command runs policy + MuJoCo + motion together:

```bash
./run_sim.sh v1_1 ~/MimicLite/any4hdmi/output/g1/kimodo/motions/wave_s42_d3.npz --initial_pause_s 1.0
```

- Open `http://<WORKSTATION_LAN_IP>:8080` (next free port if busy).
- Robot starts at the clip's first pose, waits 1 s, plays the 3 s wave, holds the last pose.
- **Restart motion** button in the viewer replays it.
- Swap `v1_1` for `roa` / `ppo` to compare.
- Add `--headless --run_once --trajectory_output /abs/path/out.npz` for a run without viewer + metrics file.

### 1b. Pico teleop in simulation

Four terminals, in order (each: `cd ~/MimicLite`):

| # | Command | What to do |
| --- | --- | --- |
| 1 | `./teleop_1_service.sh` | Pico XRoboToolkit app → connect to **<WORKSTATION_LAN_IP>**, whole-body tracking ON (leg trackers calibrated) |
| 2 | `./teleop_2_retarget.sh 1.75` | Your height in metres. `http://<WORKSTATION_LAN_IP>:8080` = kinematic preview (no physics) — must copy your movements |
| 3 | `./teleop_3_sim.sh` | MuJoCo G1 (physics) at `http://<WORKSTATION_LAN_IP>:8081`, hanging on a virtual gantry |
| 4 | `./teleop_4_policy.sh` | ROA policy (or `./teleop_4_policy.sh ppo`) |

Pico buttons (step 4):

| Button | Action |
| --- | --- |
| **A** | Go to standing pose (10 s ramp) |
| **A + B** | Policy ON |
| **X** | Start / pause following you |
| **B** | Hold current pose |

Then switch off the elastic band (virtual gantry) in the :8081 viewer.

Stop: Ctrl+C terminals 4 → 3 → 2 → 1.

---

## 2. Real robot (G1 at <HOST_G1_ROBOT_RESTAURANT_IP>)

### Before every run

- [ ] Robot on the gantry, Unitree e-stop in hand, area clear.
- [ ] SONIC / `sonic_wbc` not running (scripts refuse to start if it is).
- [ ] Workstation teleop stopped (otherwise the Pico may connect to it instead).
- [ ] Only one controller at a time: never run the robot's MuJoCo script (`teleop_3_sim_robot.sh`) together with `real_1_bridge.sh` (scripts check).

```bash
ssh -t unitree@<HOST_G1_ROBOT_RESTAURANT_IP>
cd ~/MimicLite
```

`real_1_bridge.sh` releases Unitree's built-in controller (same MotionSwitcher release SONIC does at startup) and bridges the policy to the motors on `enP8p1s0`. Both real scripts ask you to type `MOVE`.

> **The motors go stiff (hold current pose) the moment the policy script starts — before any key.**
> "Hold" is stiff, not limp. To go limp use the Unitree e-stop.

### 2a. Play the wave (v1_1)

| Terminal | Command |
| --- | --- |
| 1 | `./real_1_bridge.sh` → `MOVE` → wait until the mode check prints an empty name (Unitree released) |
| 2 | `./real_2_policy_wave.sh` → `MOVE` → robot stiffens |

Keys in terminal 2:

| Key | Action |
| --- | --- |
| `i` | Go to standing pose (10 s ramp) — wait for it |
| `]` | Policy ON, holding the wave's first pose (does **not** play yet) |
| `space` | Play the wave (3 s), then holds the last pose |
| `]` then `space` | Replay |
| `o` | Hold current pose |

### 2b. Pico teleop (roa)

Four terminals, in order:

| # | Command | What to do |
| --- | --- | --- |
| 1 | `./teleop_1_service.sh` | Pico app → connect to **<HOST_G1_ROBOT_RESTAURANT_IP>**, whole-body tracking ON |
| 2 | `./teleop_2_retarget.sh 1.75` | Check `http://<HOST_G1_ROBOT_RESTAURANT_IP>:8080` copies you **before** going on |
| 3 | `./real_1_bridge.sh` → `MOVE` | Unitree controller released; robot still limp |
| 4 | `./teleop_4_policy.sh` → `MOVE` | ROA policy; robot stiffens immediately |

Pico buttons: **A** stand (wait 10 s) → **A + B** policy ON → **X** follow you / pause → **B** hold.

### Stopping (both 2a and 2b)

1. Hold (`o` key or Pico **B**).
2. Take the weight on the gantry.
3. Ctrl+C the policy terminal, then the bridge, then (teleop) retargeter and service.
4. Unitree's normal controller comes back after a reboot.

---

## Reference

**Rates**
- Pico pose → retargeter → policy: 30 Hz (interpolated to 50 Hz inside the policy).
- Policy → motors: 50 Hz joint targets + kp/kd; each motor's driver runs the PD loop between commands.
- Robot state → policy: 100 Hz.

**Teleop delay**: roa / ppo buffer the live stream ~0.12 s.

**Troubleshooting**

| Symptom | Cause / fix |
| --- | --- |
| Viewer moves but "no physics" | You're on the retargeter preview (:8080). Physics G1 is the sim viewer (:8081) or the real robot |
| Robot/sim frozen, policy prints `tuple.index(x): x not in tuple` | `v1_1` on live teleop — use `roa` / `ppo` |
| `:8080` preview doesn't copy you | Pico not connected or no body data: check app IP, same network, leg trackers + calibration, body streaming on |
| Shaky / laggy on fast moves | Wi-Fi jitter (Pico → robot); try closer to the access point |
| Script says ABORT | Read the message: SONIC running, wrong order, or sim + bridge both up |

**Logs**: `~/MimicLite/logs/` (bridge, service, task log). Setup notes: `~/MimicLite/README_TEST.md` (G1), `~/MimicLite/logs/HOW_TO_USE.md` (workstation).
