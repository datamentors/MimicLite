"""PD stand test for the serial A2 model with the MimicLite actuator parameters from a2.py.

usage: python stand_test.py <a2_serial.xml> [seconds]
Adds a floor and position-like PD (tau = kp (q* - q) - kd qd, clipped to effort) with armature
and friction from a2.py, holds INIT_JOINT_POS and reports torso height / tilt / foot flatness.
"""
import importlib.util
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

spec_ = importlib.util.spec_from_file_location("a2cfg", Path(__file__).parents[1] / "overlay/mimic_lite/assets/a2.py")
a2 = importlib.util.module_from_spec(spec_)
sys.modules["active_adaptation.registry"] = type(sys)("stub")
sys.modules["active_adaptation.registry"].Registry = type("R", (), {"instance": staticmethod(lambda: type("I", (), {"register": lambda *a: None})())})
sys.modules["mimic_lite"] = type(sys)("stub"); sys.modules["mimic_lite.assets"] = type(sys)("stub")
hum = type(sys)("hum")
from dataclasses import dataclass


@dataclass(frozen=True)
class Motor:
    joints: tuple
    effort: float
    velocity: float
    stiffness: float
    damping: float
    armature: float = 0.0
    friction: float = 0.01


hum.Motor = Motor; hum.load_model = None; hum.make_biped = None
sys.modules["mimic_lite.assets.humanoid"] = hum
a2.__package__ = "mimic_lite.assets"
spec_.loader.exec_module(a2)

xml = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
# Passive PD cannot balance 70 kg on 50 Nm/rad ankles (the policy balances), so the test uses
# kp, kd x GAIN_SCALE with torques still clipped to the real effort limits: it checks contacts,
# mass and that holding the stance fits the torque budget.
GAIN_SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
spec = mujoco.MjSpec.from_file(xml)
floor = spec.worldbody.add_geom()
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [0, 0, 0.05]
floor.contype = 1  # the model's <default> zeroes contype/conaffinity
floor.conaffinity = 1
spec.option.timestep = 0.005  # mjlab tracking-base sim step
spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
names = [j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
kp = np.zeros(len(names)); kd = np.zeros(len(names)); eff = np.zeros(len(names))
for i, n in enumerate(names):
    for mot in a2.MOTORS:
        if any(re.fullmatch(p, n) for p in mot.joints):
            kp[i], kd[i], eff[i] = mot.stiffness * GAIN_SCALE, mot.damping * GAIN_SCALE, mot.effort
            joint = spec.joint(n)
            joint.armature = mot.armature
            joint.frictionloss = mot.friction
            act = spec.add_actuator()  # same form as mjlab BuiltinPositionActuator
            act.name, act.target = n, n
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = kp[i]
            act.biasprm[1], act.biasprm[2] = -kp[i], -kd[i]
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-eff[i], eff[i]]
assert (kp > 0).all(), [n for n, k in zip(names, kp) if k == 0]
m = spec.compile()
d = mujoco.MjData(m)
target = np.array([a2.INIT_JOINT_POS[n] for n in names])
d.qpos[:3] = a2.INIT_POS
d.qpos[3] = 1.0
d.qpos[7:] = target
mujoco.mj_forward(m, d)
feet = [m.body("left_toe_roll").id, m.body("right_toe_roll").id]
soles = np.array([d.geom_xpos[g, 2] - m.geom_size[g, 0] for g in range(m.ngeom)
                  if m.geom_bodyid[g] in feet and m.geom_contype[g]])
print(f"init: foot sphere bottoms z min={soles.min():+.4f} max={soles.max():+.4f} m (spread = foot not flat)")
hist = []
d.ctrl[:] = target
for step in range(int(secs / m.opt.timestep)):
    mujoco.mj_step(m, d)
    tau = d.actuator_force
    if step % 200 == 0:
        tilt = np.degrees(np.arccos(np.clip(d.xmat[1].reshape(3, 3)[2, 2], -1, 1)))
        hist.append((step * m.opt.timestep, d.qpos[2], tilt, np.abs(tau).max(), np.abs(target - d.qpos[7:]).max()))
for t, z, tilt, tmax, emax in hist:
    print(f"t={t:5.1f}s torso z={z:.3f} tilt={tilt:5.1f}deg max|tau|={tmax:6.1f} max joint err={emax:.3f} rad")
ok = hist[-1][1] > 0.8 * a2.INIT_POS[2] and hist[-1][2] < 15
print("STAND", "PASS" if ok else "FAIL")
tau_ss = np.abs(d.actuator_force)
print("steady |tau| / effort (top 6):", sorted(((round(float(t / e), 2), n) for t, e, n in zip(tau_ss, eff, names)), reverse=True)[:6])
