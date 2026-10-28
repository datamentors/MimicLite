"""AgiBot A2 (Ultra) serial-equivalent model for MimicLite.

Model: a2_serial.xml built by a2-mimiclite/model/build_a2_serial.py from the official AgiBot
rl_deploy MJCF (a2_t2d0_ultra). Parallel ankle / wrist linkages are replaced by the serial
toe_pitch/toe_roll and arm_joint6/7 joints the robot SDK commands; head and fingers are welded.

Actuator values (no AgiBot armature/friction data available - estimates, randomized wide):
- effort: legs from the Ultra URDF (hip roll/yaw 120, hip pitch/knee 140, ankle 50 Nm);
  arms are not specified by AgiBot (position servos in their MJCF) -> estimates
- kp/kd: legs from AgiBot's own RL config (rl_training a2_dh_stand), arms A3-like
"""

import os

from active_adaptation.registry import Registry

from .humanoid import Motor, make_biped

A2_MJCF = os.environ.get("A2_MJCF", "/opt/a2/a2_serial/a2_serial.xml")

INIT_POS = (0.0, 0.0, 0.97)
INIT_JOINT_POS = {
    "idx01_left_hip_roll": 0.0,
    "idx02_left_hip_yaw": 0.0,
    "idx03_left_hip_pitch": -0.25,
    "idx04_left_tarsus": 0.5,
    "idx05_left_toe_pitch": -0.25,
    "idx06_left_toe_roll": 0.0,
    "idx07_right_hip_roll": 0.0,
    "idx08_right_hip_yaw": 0.0,
    "idx09_right_hip_pitch": -0.25,
    "idx10_right_tarsus": 0.5,
    "idx11_right_toe_pitch": -0.25,
    "idx12_right_toe_roll": 0.0,
    "idx13_left_arm_joint1": 0.0,
    "idx14_left_arm_joint2": 1.05,
    "idx15_left_arm_joint3": 0.0,
    "idx16_left_arm_joint4": -0.5,
    "idx17_left_arm_joint5": -0.05,
    "idx18_left_arm_joint6": 0.0,
    "idx19_left_arm_joint7": 0.0,
    "idx20_right_arm_joint1": 0.0,
    "idx21_right_arm_joint2": -1.05,
    "idx22_right_arm_joint3": 0.0,
    "idx23_right_arm_joint4": 0.5,
    "idx24_right_arm_joint5": -1.51,
    "idx25_right_arm_joint6": 0.0,
    "idx26_right_arm_joint7": 0.0,
}

# (joint patterns, effort Nm, kp, kd, armature kg m^2)
MOTORS = [
    Motor((r".*_hip_roll",), 120.0, 12.0, 130.0, 5.0, 0.03, 0.05),
    Motor((r".*_hip_yaw",), 120.0, 12.0, 130.0, 3.0, 0.03, 0.05),
    Motor((r".*_hip_pitch",), 140.0, 12.0, 200.0, 6.0, 0.06, 0.05),
    Motor((r".*_tarsus",), 140.0, 12.0, 220.0, 7.0, 0.06, 0.05),
    Motor((r".*_toe_pitch", r".*_toe_roll"), 50.0, 12.0, 50.0, 1.5, 0.02, 0.05),
    Motor((r".*_arm_joint[12]",), 60.0, 6.0, 60.0, 3.0, 0.02, 0.05),
    Motor((r".*_arm_joint[34]",), 40.0, 6.0, 40.0, 2.0, 0.015, 0.05),
    Motor((r".*_arm_joint5",), 20.0, 6.0, 20.0, 1.0, 0.01, 0.05),
    Motor((r".*_arm_joint[67]",), 10.0, 6.0, 15.0, 1.0, 0.005, 0.05),
]


def make_cfg(backend):
    return make_biped(
        backend,
        mjcf=A2_MJCF,
        urdf=None,
        pos=INIT_POS,
        joint_pos=INIT_JOINT_POS,
        motors=MOTORS,
        root="torso",
        feet=r"^(left_toe_roll|right_toe_roll)$",
    )


Registry.instance().register("asset", "mlite-a2", make_cfg)
