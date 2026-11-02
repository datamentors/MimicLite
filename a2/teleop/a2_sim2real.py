"""AgiBot A2 support for MimicLite sim2real, injected at runtime (the sim2real submodule stays upstream).

Adds:
  - RobotCfg "a2" to sim2real's robot registry (sim, policy, teleop publisher all call get_robot_cfg)
  - GMR target "agibot_a2" (robot MJCF + xrobot IK config) to GMR's module-level registries
  - pico_retarget_pub hardcodes tgt_robot="unitree_g1": with --robot a2 the GMR class is swapped for
    one that retargets to agibot_a2 instead
  - PICO drift lock (--robot a2): GMR's XRobotStreamer is swapped for one that removes slow PICO world-frame
    drift while both feet stand still (see DriftLock)

Launcher (run any sim2real entry point with A2 registered):
  python a2/teleop/a2_sim2real.py <sim2real script.py> --robot a2 [script args...]

Model: a2/model/<A2_MODEL>/a2_serial.xml, the MJCF the policy was trained on: A2_MODEL=a2_serial_selfcol (default;
same bodies/joints + self-collision shapes, for policies trained with image a2-mimiclite:selfcol) or a2_serial
(no self-collision, for policies trained with image a2-mimiclite:latest). sim2real needs a named root joint and adds its own floor scene, so a copy with the
free joint named "floating_base_joint" and without A2's skybox/groundplane assets is generated next to it
(a2_serial_sim.xml, git-ignored). Actuator values below mirror a2/overlay/mimic_lite/assets/a2.py MOTORS.
"""

from __future__ import annotations

import os
import re
import runpy
import sys
import time
from pathlib import Path

import numpy as np

A2_DIR = Path(__file__).resolve().parents[1]
A2_MODEL = os.environ.get("A2_MODEL", "a2_serial_selfcol")
A2_SRC_MJCF = A2_DIR / "model" / A2_MODEL / "a2_serial.xml"
if not A2_SRC_MJCF.exists():
    raise FileNotFoundError(f"A2_MODEL={A2_MODEL}: {A2_SRC_MJCF} not found (a2_serial | a2_serial_selfcol)")
A2_SIM_MJCF = A2_SRC_MJCF.with_name("a2_serial_sim.xml")
CALIB_DIR = Path(os.environ.get("A2_CALIB_DIR") or A2_DIR.parent / "calib").expanduser()  # same as teleop.sh
# GMR config = the per-operator T-pose calibration (teleop.sh calibrate, latest linked as xrobot_to_a2_tpose.json);
# A2_GMR_CONFIG picks another calibration
GMR_CONFIG = Path(os.environ.get("A2_GMR_CONFIG") or CALIB_DIR / "xrobot_to_a2_tpose.json").expanduser()
GMR_ROBOT = "agibot_a2"
A2_MASS_KG = 70.51  # a2_serial.xml total mass (sim model)
G1_MASS_KG = 33.74  # sim2real G1 (g1-mode_13_15.xml), what base_sim's elastic band is tuned for

JOINT_NAMES = (
    "idx01_left_hip_roll", "idx02_left_hip_yaw", "idx03_left_hip_pitch",
    "idx04_left_tarsus", "idx05_left_toe_pitch", "idx06_left_toe_roll",
    "idx07_right_hip_roll", "idx08_right_hip_yaw", "idx09_right_hip_pitch",
    "idx10_right_tarsus", "idx11_right_toe_pitch", "idx12_right_toe_roll",
    "idx13_left_arm_joint1", "idx14_left_arm_joint2", "idx15_left_arm_joint3", "idx16_left_arm_joint4",
    "idx17_left_arm_joint5", "idx18_left_arm_joint6", "idx19_left_arm_joint7",
    "idx20_right_arm_joint1", "idx21_right_arm_joint2", "idx22_right_arm_joint3", "idx23_right_arm_joint4",
    "idx24_right_arm_joint5", "idx25_right_arm_joint6", "idx26_right_arm_joint7",
)

# Serial chain (no linkage rods / hands / welded head). Includes every body the exported policy tracks
# (motion.body_names: torso, *_hip_yaw, *_tarsus, *_toe_pitch, (left|right)-arm-link{3,4,7}); all exist in
# both a2_serial.xml and the datasets' a2_kin.xml (npz_pub runs FK on the dataset MJCF, which has no paws).
BODY_NAMES = (
    "torso",
    "left_hip_roll", "left_hip_yaw", "left_hip_pitch", "left_tarsus", "left_toe_pitch", "left_toe_roll",
    "right_hip_roll", "right_hip_yaw", "right_hip_pitch", "right_tarsus", "right_toe_pitch", "right_toe_roll",
    "left-arm-link1", "left-arm-link2", "left-arm-link3", "left-arm-link4",
    "left-arm-link5", "left-arm-link6", "left-arm-link7",
    "right-arm-link1", "right-arm-link2", "right-arm-link3", "right-arm-link4",
    "right-arm-link5", "right-arm-link6", "right-arm-link7",
)

# a2.py MOTORS: (joint regex, effort Nm, velocity rad/s, armature kg m^2, friction Nm)
MOTORS = (
    (r".*_hip_roll", 120.0, 12.0, 0.03, 0.05),
    (r".*_hip_yaw", 120.0, 12.0, 0.03, 0.05),
    (r".*_hip_pitch", 140.0, 12.0, 0.06, 0.05),
    (r".*_tarsus", 140.0, 12.0, 0.06, 0.05),
    (r".*_toe_(pitch|roll)", 50.0, 12.0, 0.02, 0.05),
    (r".*_arm_joint[12]", 60.0, 6.0, 0.02, 0.05),
    (r".*_arm_joint[34]", 40.0, 6.0, 0.015, 0.05),
    (r".*_arm_joint5", 20.0, 6.0, 0.01, 0.05),
    (r".*_arm_joint[67]", 10.0, 6.0, 0.005, 0.05),
)

# a2_serial.xml joint ranges [rad]
JOINT_RANGE = {
    "hip_roll": (-0.698, 0.698), "hip_yaw": (-1.5708, 1.5708), "hip_pitch": (-1.9199, 0.7854),
    "tarsus": (-0.0873, 2.4435), "toe_pitch": (-1.0472, 0.5236), "toe_roll": (-0.4887, 0.4887),
    "left_arm_joint1": (-2.967, 2.967), "left_arm_joint2": (-0.5236, 1.6581),
    "left_arm_joint3": (-2.967, 2.967), "left_arm_joint4": (-2.0944, 0.0),
    "right_arm_joint1": (-2.967, 2.967), "right_arm_joint2": (-1.6581, 0.5236),
    "right_arm_joint3": (-2.967, 2.967), "right_arm_joint4": (0.0, 2.0944),
    "arm_joint5": (-2.8798, 2.8798), "arm_joint6": (-0.7854, 0.7854), "arm_joint7": (-0.5236, 0.5236),
}

# a2.py INIT_POS / INIT_JOINT_POS (= exported policy default_joint_pos)
INIT_ROOT_POS = (0.0, 0.0, 0.97)
INIT_JOINT_POS = (
    0.0, 0.0, -0.25, 0.5, -0.25, 0.0,
    0.0, 0.0, -0.25, 0.5, -0.25, 0.0,
    0.0, 1.05, 0.0, -0.5, -0.05, 0.0, 0.0,
    0.0, -1.05, 0.0, 0.5, -1.51, 0.0, 0.0,
)


def _motor(joint: str) -> tuple[float, float, float, float]:
    for pattern, effort, velocity, armature, friction in MOTORS:
        if re.fullmatch(pattern, joint):
            return effort, velocity, armature, friction
    raise KeyError(f"no A2 motor entry for {joint}")


def _joint_range(joint: str) -> tuple[float, float]:
    name = joint.split("_", 1)[1]  # drop idxNN_
    for key in (name, name.split("_", 1)[1]):  # side-specific first, then generic
        if key in JOINT_RANGE:
            return JOINT_RANGE[key]
    raise KeyError(f"no A2 joint range for {joint}")


def ensure_sim_mjcf() -> Path:
    """Write a2_serial_sim.xml (named free joint) next to a2_serial.xml when missing or stale."""
    newest_input = max(A2_SRC_MJCF.stat().st_mtime, Path(__file__).stat().st_mtime)
    if A2_SIM_MJCF.exists() and A2_SIM_MJCF.stat().st_mtime >= newest_input:
        return A2_SIM_MJCF
    text = A2_SRC_MJCF.read_text(encoding="utf-8")
    if text.count('<joint type="free"/>') != 1:
        raise RuntimeError(f'expected exactly one <joint type="free"/> in {A2_SRC_MJCF}')
    text = text.replace('<joint type="free"/>', '<joint name="floating_base_joint" type="free"/>')
    # sim2real's load_sim_model injects its own skybox + groundplane texture/material and floor:
    # drop A2's copies (same names) so the scene is defined once
    scene_lines = re.compile(r'^\s*<(texture|material) [^>]*name="(skybox|groundplane)"[^>]*/>\s*\n', re.M)
    text, n_removed = scene_lines.subn("", text)
    if n_removed != 3:
        raise RuntimeError(f"expected 3 skybox/groundplane lines in {A2_SRC_MJCF}, found {n_removed}")
    tmp = A2_SIM_MJCF.with_suffix(".xml.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(A2_SIM_MJCF)
    return A2_SIM_MJCF


def make_robot_cfg():
    from sim2real.config.robots.base import RobotCfg

    motors = {j: _motor(j) for j in JOINT_NAMES}
    ranges = {j: _joint_range(j) for j in JOINT_NAMES}
    return RobotCfg(
        name="a2",
        joint_names=JOINT_NAMES,
        body_names=BODY_NAMES,
        joint_pos_lower_limit={j: r[0] for j, r in ranges.items()},
        joint_pos_upper_limit={j: r[1] for j, r in ranges.items()},
        joint_velocity_limit={j: m[1] for j, m in motors.items()},
        joint_effort_limit={j: m[0] for j, m in motors.items()},
        joint_armature={j: m[2] for j, m in motors.items()},
        joint_frictionloss={j: m[3] for j, m in motors.items()},
        mjcf_path=str(ensure_sim_mjcf()),
        default_qpos=INIT_ROOT_POS + (1.0, 0.0, 0.0, 0.0) + INIT_JOINT_POS,
        publish_hz=50.0,
        viewer_track_body_names=("torso",),
        elastic_band_attach_body_names=("torso",),
    )


def register() -> None:
    """Register A2 in sim2real (always) and GMR (when importable, i.e. in the teleop venv)."""
    import sim2real.config.robots as robots

    robots._ROBOT_CFGS["a2"] = make_robot_cfg()
    _patch_sim_floor()
    _patch_elastic_band()

    try:
        import general_motion_retargeting as gmr
        from general_motion_retargeting import params
    except ImportError:
        return
    if not GMR_CONFIG.exists():
        raise FileNotFoundError(f"{GMR_CONFIG} missing: calibrate first ('a2/teleop/teleop.sh calibrate <height>')")
    params.ROBOT_XML_DICT[GMR_ROBOT] = ensure_sim_mjcf()
    params.IK_CONFIG_DICT.setdefault("xrobot", {})[GMR_ROBOT] = GMR_CONFIG
    params.ROBOT_BASE_DICT[GMR_ROBOT] = "torso"
    params.VIEWER_CAM_DISTANCE_DICT[GMR_ROBOT] = 3.0
    _patch_xrobot_drift_lock(gmr)

    base_cls = gmr.GeneralMotionRetargeting
    if getattr(base_cls, "_a2_patched", False):
        return

    class A2GeneralMotionRetargeting(base_cls):
        """pico_retarget_pub passes tgt_robot="unitree_g1" regardless of --robot; redirect it for A2."""

        _a2_patched = True

        def __init__(self, src_human, tgt_robot, *args, **kwargs):
            if _active_robot() == "a2" and tgt_robot == "unitree_g1":
                tgt_robot = GMR_ROBOT
            super().__init__(src_human, tgt_robot, *args, **kwargs)

    gmr.GeneralMotionRetargeting = A2GeneralMotionRetargeting


def _patch_sim_floor() -> None:
    """sim2real injects <geom name="floor"> without contype/conaffinity, so it inherits A2's main default
    (contype=conaffinity=0, which keeps A2's visual meshes out of collisions) and nothing touches it.
    load_sim_model hands its MjSpec to ensure_joint_motor_actuators (module-global lookup): wrap that to
    make the floor collide with A2's foot spheres (contype 1)."""
    try:
        from sim2real.sim_env.utils import mjcf
    except ImportError:  # teleop venv without the sim package
        return
    base_fn = mjcf.ensure_joint_motor_actuators
    if getattr(base_fn, "_a2_patched", False):
        return

    def ensure_joint_motor_actuators(spec, robot_cfg):
        added = base_fn(spec, robot_cfg)
        if robot_cfg.name == "a2":
            floor = spec.geom("floor")
            if floor is None:
                raise RuntimeError("sim2real scene has no 'floor' geom to enable collisions on")
            floor.contype = 1
            floor.conaffinity = 1
        return added

    ensure_joint_motor_actuators._a2_patched = True
    mjcf.ensure_joint_motor_actuators = ensure_joint_motor_actuators


def _patch_elastic_band() -> None:
    """base_sim's elastic band (virtual gantry) is sized for G1 (33.7 kg): stiffness 200 / damping 100 toward
    a point 3 m up. A2 (70.5 kg) sags through it. Scale both by the mass ratio so A2 hangs like G1 does
    (override with env A2_BAND_SCALE). Applies only with --robot a2."""
    if _active_robot() != "a2":
        return
    try:
        from sim2real.sim_env.utils import elastic_band
    except ImportError:
        return
    base_cls = elastic_band.ElasticBand
    if getattr(base_cls, "_a2_patched", False):
        return
    scale = float(os.environ.get("A2_BAND_SCALE", A2_MASS_KG / G1_MASS_KG))

    class A2ElasticBand(base_cls):
        _a2_patched = True

        def __init__(self):
            super().__init__()
            self.stiffness *= scale
            self.damping *= scale

    elastic_band.ElasticBand = A2ElasticBand


class DriftLock:
    """Removes slow drift of PICO's world frame from the body stream (z-up, after the Unity transform).

    pico_retarget_pub passes PICO's pelvis/feet world XY straight to GMR (anchored once, at X / reconnect), and
    the A2 policy reads the reference root motion as a walk command: a slowly sliding PICO world frame makes
    the robot walk although the operator stands still. While both foot trackers are still (smoothed horizontal
    speed below STILL_SPEED for STILL_HOLD_S), every horizontal shift of the feet is folded into an offset
    applied to the whole body, so the corrected feet stay where they were; the pelvis still moves relative to
    the feet (weight shifts, leaning). When a foot moves the offset is frozen, so real steps pass through
    unchanged. Drift while walking is not removed. Heading (yaw) drift is not handled."""

    FEET = ("Left_Foot", "Right_Foot")
    STILL_SPEED = 0.08  # m/s; standing-still tracker noise (T-pose recordings) peaks at 0.03 m/s smoothed
    STILL_HOLD_S = 0.2
    SPEED_TAU_S = 0.1  # speed smoothing time constant
    GAP_S = 0.5  # longer pause between frames (reconnect) restarts the stillness detection

    def __init__(self):
        self.offset = np.zeros(2)
        self.locked = False
        self._lock_xy = np.zeros(2)  # corrected mean foot position held while locked
        self._prev_t: float | None = None
        self._prev_feet = np.zeros((2, 2))
        self._speed = np.zeros(2)
        self._still_s = 0.0

    def apply(self, body_pose_dict: dict, t_s: float) -> dict:
        feet = np.array([body_pose_dict[name][0][:2] for name in self.FEET], dtype=np.float64)
        dt = None if self._prev_t is None else t_s - self._prev_t
        if dt is None or dt > self.GAP_S:
            self._speed[:] = 0.0
            self._still_s = 0.0
        elif dt > 0.0:
            speed = np.linalg.norm(feet - self._prev_feet, axis=1) / dt
            self._speed += min(dt / self.SPEED_TAU_S, 1.0) * (speed - self._speed)
            both_still = bool((self._speed < self.STILL_SPEED).all())
            self._still_s = self._still_s + dt if both_still else 0.0
        self._prev_t, self._prev_feet = t_s, feet

        mean_feet = feet.mean(axis=0)
        lock = self._still_s >= self.STILL_HOLD_S
        if lock and not self.locked:
            self._lock_xy = mean_feet + self.offset  # offset stays continuous at lock start
        if lock:
            self.offset = self._lock_xy - mean_feet
        if lock != self.locked:
            print(f"[A2 drift lock] {'feet still: locked' if lock else 'feet moving: released'}, "
                  f"offset ({self.offset[0]:+.3f}, {self.offset[1]:+.3f}) m", flush=True)
            self.locked = lock

        ox, oy = float(self.offset[0]), float(self.offset[1])
        return {name: [[float(pos[0]) + ox, float(pos[1]) + oy, float(pos[2])], quat]
                for name, (pos, quat) in body_pose_dict.items()}


def _patch_xrobot_drift_lock(gmr) -> None:
    """pico_retarget_pub reads PICO body data through GMR's XRobotStreamer (get_raw_body_data, then
    coordinate_transform_unity_data): swap in a subclass whose transform runs DriftLock. Applies only with
    --robot a2. tools/fake_pico.py subclasses the swapped class, so the fake PICO goes through it too."""
    if _active_robot() != "a2":
        return
    base_cls = gmr.XRobotStreamer
    if getattr(base_cls, "_a2_drift_patched", False):
        return

    class A2DriftLockXRobotStreamer(base_cls):
        _a2_drift_patched = True

        def coordinate_transform_unity_data(self, body_pose_dict):
            body_pose_dict = super().coordinate_transform_unity_data(body_pose_dict)
            lock = self.__dict__.get("_a2_drift_lock")  # FakeXRobotStreamer skips __init__: create lazily
            if lock is None:
                lock = self._a2_drift_lock = DriftLock()
            return lock.apply(body_pose_dict, time.monotonic())

    gmr.XRobotStreamer = A2DriftLockXRobotStreamer


def _active_robot() -> str | None:
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg.startswith("--robot="):
            return arg.split("=", 1)[1].strip().lower()
        if arg == "--robot" and i + 1 < len(argv):
            return argv[i + 1].strip().lower()
    return None


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        sys.exit(f"usage: python {Path(__file__).name} <sim2real script.py> --robot a2 [args...]")
    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        sys.exit(f"script not found: {script}")
    sys.argv = [str(script)] + sys.argv[2:]
    sys.path.insert(0, str(script.parent))
    register()
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
