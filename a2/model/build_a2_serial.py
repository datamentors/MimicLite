"""Build a serial-equivalent AgiBot A2 (Ultra) MJCF for MimicLite training.

Source: A2Docs/rl_deploy/mujoco_sim/resource/model/a2_t2d0_ultra/mujoco/model.xml (official,
closed-loop ankles/wrists via equality constraints). The real robot is commanded in serial
joint space (aimrl_sdk: 12 leg joints incl. toe_pitch/toe_roll + 14 arm joints), so training
uses a tree model where those joints are driven directly:

- keep hinge joints: hip roll/yaw/pitch, tarsus (knee), toe_pitch, toe_roll, arm joint1-7 (x2)
- delete every other joint (ankle motor A/B, ball joints, wrist rod slides, head, fingers):
  their bodies become welded to the parent, so mass/inertia/geometry are preserved
- delete actuators, equalities, sensors, terrain floor and lights (mjlab adds its own)
- disable collision on the ankle/wrist linkage parts (welded rods would collide unphysically)
- --self-collision: add limb/hand/torso collision shapes that touch only each other (see
  add_self_collision); the AgiBot model has no limb collision geometry at all

usage: python build_a2_serial.py --src <official model.xml> --out <dir> [--self-collision]
Writes <dir>/a2_serial.xml and copies the referenced meshes to <dir>/meshes/.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np

KEEP_JOINTS = [
    "idx01_left_hip_roll", "idx02_left_hip_yaw", "idx03_left_hip_pitch",
    "idx04_left_tarsus", "idx05_left_toe_pitch", "idx06_left_toe_roll",
    "idx07_right_hip_roll", "idx08_right_hip_yaw", "idx09_right_hip_pitch",
    "idx10_right_tarsus", "idx11_right_toe_pitch", "idx12_right_toe_roll",
    "idx13_left_arm_joint1", "idx14_left_arm_joint2", "idx15_left_arm_joint3",
    "idx16_left_arm_joint4", "idx17_left_arm_joint5", "idx18_left_arm_joint6",
    "idx19_left_arm_joint7",
    "idx20_right_arm_joint1", "idx21_right_arm_joint2", "idx22_right_arm_joint3",
    "idx23_right_arm_joint4", "idx24_right_arm_joint5", "idx25_right_arm_joint6",
    "idx26_right_arm_joint7",
]
# Linkage parts that only exist to close the parallel ankle / wrist loops.
LINKAGE_KEYWORDS = ("motor", "rod", "_ball")
# Welded subtrees removed from the kinematics-only dataset model (a2_kin.xml).
KIN_DROP_SUBTREES = [
    "head_yaw",
    "left_toe_A", "left_toe_B", "right_toe_A", "right_toe_B",
    "left_wrist_motor_A", "left_wrist_motor_B", "right_wrist_motor_A", "right_wrist_motor_B",
    "L_paw", "R_paw",
]
# Self-collision shapes: (geom name, shape, body it is attached to, bodies whose meshes it covers).
# A "+" suffix means the body and its whole subtree (hand = paw + fingers, head = yaw + pitch).
SELF_COLLISION_SHAPES = [("torso_collision", "box", "torso", ["torso"]),
                         ("head_collision", "capsule", "head_yaw", ["head_yaw+"])]
for _s, _h in (("left", "L"), ("right", "R")):
    SELF_COLLISION_SHAPES += [
        (f"{_s}_thigh_collision", "capsule", f"{_s}_hip_pitch", [f"{_s}_hip_pitch"]),
        (f"{_s}_shank_collision", "capsule", f"{_s}_tarsus", [f"{_s}_tarsus"]),
        (f"{_s}_foot_collision", "capsule", f"{_s}_toe_roll", [f"{_s}_toe_roll"]),
        (f"{_s}_shoulder_collision", "capsule", f"{_s}-arm-link2", [f"{_s}-arm-link2"]),
        (f"{_s}_upper_arm_collision", "capsule", f"{_s}-arm-link3", [f"{_s}-arm-link3"]),
        (f"{_s}_elbow_collision", "capsule", f"{_s}-arm-link4", [f"{_s}-arm-link4"]),
        (f"{_s}_forearm_collision", "capsule", f"{_s}-arm-link5",
         [f"{_s}-arm-link5", f"{_s}_wrist_motor_A", f"{_s}_wrist_motor_B"]),
        (f"{_s}_hand_collision", "capsule", f"{_h}_paw", [f"{_h}_paw+"]),
    ]
SELF_COLLISION_BIT = 2  # floor/terrain use bit 1 -> these shapes never touch the floor
SELF_COLLISION_EXCLUDE_HOPS = 3  # bodies this close in the tree always overlap at the joint
# Shapes sit ~1 cm inside the surface: a 90th-percentile radius / one torso box stuck out past
# motor bulges and the narrow waist (61% of dataset frames flagged); at 75 / 5-95 an elbow resting
# against the torso side still registered (4 mm). Only real interpenetration should be penalized.
SELF_COLLISION_RADIUS_PCT = 60
SELF_COLLISION_BOX_SLICES = 3  # box shapes (torso) = this many boxes stacked along the long axis
SELF_COLLISION_BOX_WIDTH_PCT = (10, 90)


def _mesh_points(m: mujoco.MjModel, d: mujoco.MjData, bodies: list[str]) -> np.ndarray:
    """World-frame mesh vertices of the given bodies ("name+" = body and its subtree)."""
    ids = set()
    for name in bodies:
        root = m.body(name.rstrip("+")).id
        ids |= {b for b in range(m.nbody) if b == root or (name.endswith("+") and _is_ancestor(m, root, b))}
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in ids and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
            pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.concatenate(pts)


def _is_ancestor(m: mujoco.MjModel, a: int, b: int) -> bool:
    while b > 0:
        b = m.body_parentid[b]
        if b == a:
            return True
    return False


def _tree_hops(m: mujoco.MjModel, a: int, b: int) -> int:
    chain_a = [a]
    while chain_a[-1] > 0:
        chain_a.append(m.body_parentid[chain_a[-1]])
    hops_b = 0
    while b not in chain_a:
        b = m.body_parentid[b]
        hops_b += 1
    return chain_a.index(b) + hops_b


def add_self_collision(spec: mujoco.MjSpec) -> None:
    """Fit the SELF_COLLISION_SHAPES entries to the mesh vertices at qpos0.

    Principal axes of the vertex cloud. Capsule: length = 2nd..98th percentile along the long axis,
    radius = SELF_COLLISION_RADIUS_PCT percentile distance from it. Box: SELF_COLLISION_BOX_SLICES
    boxes along the long axis, SELF_COLLISION_BOX_WIDTH_PCT percentiles wide (follows the waist taper; geoms
    <name>_<i>_collision). contype = conaffinity = 2, so the shapes
    collide only with each other: floor contact (foot spheres, torso and hand meshes) and foot
    friction are unchanged. condim 1 (frictionless) as in mjlab's G1 self-collision setup; group 3
    (hidden by default in viewers). Body pairs within SELF_COLLISION_EXCLUDE_HOPS are excluded.
    """
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    for name, shape, body, cover in SELF_COLLISION_SHAPES:
        b = m.body(body).id
        R, p0 = d.xmat[b].reshape(3, 3), d.xpos[b]
        local = (_mesh_points(m, d, cover) - p0) @ R  # body frame
        c = local.mean(0)
        axes = np.linalg.svd(local - c, full_matrices=False)[2]  # rows = principal axes
        if np.linalg.det(axes) < 0:
            axes[2] *= -1
        q = (local - c) @ axes.T
        lo, hi = np.percentile(q, 2, axis=0), np.percentile(q, 98, axis=0)
        geoms = []
        if shape == "box":
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, axes.T.flatten())
            edges = np.linspace(lo[0], hi[0], SELF_COLLISION_BOX_SLICES + 1)
            for i in range(SELF_COLLISION_BOX_SLICES):
                band = q[(q[:, 0] >= edges[i]) & (q[:, 0] <= edges[i + 1])]
                blo = np.array([edges[i], *np.percentile(band[:, 1:], SELF_COLLISION_BOX_WIDTH_PCT[0], axis=0)])
                bhi = np.array([edges[i + 1], *np.percentile(band[:, 1:], SELF_COLLISION_BOX_WIDTH_PCT[1], axis=0)])
                geom = spec.body(body).add_geom(name=name.replace("_collision", f"_{i}_collision"))
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.size = (bhi - blo) / 2
                geom.pos = c + ((blo + bhi) / 2) @ axes
                geom.quat = quat
                geoms.append(geom)
        else:
            mid = c + ((lo + hi) / 2) @ axes
            radius = float(np.percentile(np.linalg.norm(q[:, 1:], axis=1), SELF_COLLISION_RADIUS_PCT))
            half = max((hi[0] - lo[0]) / 2 - radius, 0.005)
            geom = spec.body(body).add_geom(name=name)
            geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            geom.fromto = np.concatenate([mid - half * axes[0], mid + half * axes[0]])
            geom.size = [radius, 0, 0]
            geoms.append(geom)
        for geom in geoms:
            geom.contype = geom.conaffinity = SELF_COLLISION_BIT
            geom.condim = 1
            geom.group = 3
            geom.density = 0.0  # bodies keep their explicit inertial; mass must not change
            geom.rgba = [0.9, 0.35, 0.2, 0.45]
    bodies = [m.body(body).id for _, _, body, _ in SELF_COLLISION_SHAPES]
    excluded = []
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            if _tree_hops(m, a, b) <= SELF_COLLISION_EXCLUDE_HOPS:
                spec.add_exclude(bodyname1=m.body(a).name, bodyname2=m.body(b).name)
                excluded.append(f"{m.body(a).name}/{m.body(b).name}")
    print(f"self-collision: {len(SELF_COLLISION_SHAPES)} shapes, excluded pairs: {excluded}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--self-collision", action="store_true",
                        help="add limb/hand/torso self-collision shapes (add_self_collision)")
    args = parser.parse_args()
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    spec = mujoco.MjSpec.from_file(str(src))
    spec.modelname = "a2_serial"

    for act in list(spec.actuators):
        spec.delete(act)
    for eq in list(spec.equalities):
        spec.delete(eq)
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for exclude in list(spec.excludes):
        spec.delete(exclude)
    for key in list(spec.keys):  # keyframes are sized for the 99-dof source model
        spec.delete(key)
    for light in list(spec.lights):
        spec.delete(light)
    for geom in list(spec.worldbody.geoms):  # terrain floor from flat.xml
        spec.delete(geom)

    kept = []
    for joint in list(spec.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if joint.name in KEEP_JOINTS:
            kept.append(joint.name)
        else:
            spec.delete(joint)
    missing = sorted(set(KEEP_JOINTS) - set(kept))
    if missing:
        raise RuntimeError(f"joints not found in source model: {missing}")

    for body in spec.bodies:
        if any(k in body.name.lower() for k in LINKAGE_KEYWORDS):
            for geom in body.geoms:
                geom.contype = 0
                geom.conaffinity = 0

    if args.self_collision:
        add_self_collision(spec)

    # Copy meshes next to the output. to_xml() compiles against the source paths, so the
    # mesh file attributes are rewritten in the serialized text afterwards.
    mesh_out = out / "meshes"
    mesh_out.mkdir(parents=True, exist_ok=True)
    renames = {}
    for mesh in spec.meshes:
        if not mesh.file:
            continue
        mesh_src = (src.parent / spec.meshdir / mesh.file).resolve()
        if not mesh_src.is_file():
            raise FileNotFoundError(f"mesh {mesh.name}: {mesh_src} not found")
        shutil.copy2(mesh_src, mesh_out / mesh_src.name)
        renames[mesh.file] = f"meshes/{mesh_src.name}"
    xml = spec.to_xml()
    for old, new in renames.items():
        xml = xml.replace(f'file="{old}"', f'file="{new}"')
    xml = xml.replace(f'meshdir="{spec.meshdir}"', "") if spec.meshdir else xml

    xml_path = out / "a2_serial.xml"
    xml_path.write_text(xml, encoding="utf-8")

    reloaded = mujoco.MjModel.from_xml_path(str(xml_path))
    hinge = [reloaded.joint(i).name for i in range(reloaded.njnt)
             if reloaded.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    print(f"wrote {xml_path}")
    print(f"nbody={reloaded.nbody} njnt={reloaded.njnt} hinge={len(hinge)} nu={reloaded.nu} "
          f"neq={reloaded.neq} mass={reloaded.body_mass.sum():.2f} (src mass "
          f"{mujoco.MjModel.from_xml_path(str(src)).body_mass.sum():.2f})")
    print("joint order:", hinge)
    assert reloaded.nq == 7 + len(KEEP_JOINTS), reloaded.nq

    # Kinematics-only twin for motion datasets: any4hdmi caches FK for every body of the dataset
    # MJCF (float32), so drop the welded leaf bodies (head, ankle/wrist linkage, hands+fingers).
    # Same joints/qpos as a2_serial.xml; every body MimicLite tracks is kept.
    kin = mujoco.MjSpec.from_string(xml_path.read_text(encoding="utf-8"))
    for exclude in list(kin.excludes):  # self-collision excludes may name dropped bodies
        kin.delete(exclude)
    for name in KIN_DROP_SUBTREES:  # each is a subtree root; children go with it
        kin.delete(kin.body(name))
    kin.modelname = "a2_kin"
    kin.modelfiledir = str(out)  # to_xml() compiles; resolve meshes/ against the output dir
    kin_path = out / "a2_kin.xml"
    kin_path.write_text(kin.to_xml(), encoding="utf-8")
    km = mujoco.MjModel.from_xml_path(str(kin_path))
    print(f"wrote {kin_path}: nbody={km.nbody} nq={km.nq} bodies={[km.body(i).name for i in range(1, km.nbody)]}")
    assert km.nq == reloaded.nq


if __name__ == "__main__":
    main()
