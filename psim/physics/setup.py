"""Which objects the rigid-body solver handles, and how (built with the execution plan).

Ownership rules (how the native solver and processes share objects):

* A solver body is *dynamic* when its translation is integrated, i.e. only contribution
  processes (forces, torques, accelerations) act on it. The solver adds contact and joint
  impulses to its velocity and corrects its position. Processes never write it directly.
* A body moved by prescribed or update processes is *kinematic*: it pushes other bodies but
  nothing pushes it (infinite mass for the solver), so the processes stay the only writers.
* Objects that are not simulated are *static* (fixed colliders / joint anchors).
* Colliders and joints on objects inside a simulated assembly belong to that assembly, which
  then behaves as one compound rigid body.

A collider is only created from the explicit ``physics.collider`` setting (rigid-body types
default to "auto"); mass alone never makes an object collide.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import numerics
from ..scene.analysis import body_mass_properties
from .colliders import ColliderShape, collider_for
from .joints import Joint


@dataclass
class ColliderEntry:
    body: str                     # solver body key (object id of the body that carries it)
    source: str                   # object that owns the collider
    shape: ColliderShape
    rel_pos: np.ndarray           # collider object's origin in the body's frame
    rel_rot: np.ndarray           # (3, 3) collider object's orientation in the body's frame
    restitution: float
    friction: float


@dataclass
class SolverBody:
    key: str
    role: str                     # dynamic | kinematic | static
    engine_index: int | None      # row in the engine's BodySet (None for static bodies)
    pose: tuple[np.ndarray, np.ndarray] | None = None   # static bodies: world origin and quaternion
    com_local: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class JointEntry:
    joint: Joint
    body_a: str
    body_b: str | None            # None = the fixed world
    anchor_a: np.ndarray          # in body A's frame (origin frame)
    anchor_b: np.ndarray          # in body B's frame, or world
    axis_a: np.ndarray
    axis_b: np.ndarray
    ref_rot: np.ndarray           # (3, 3): R_A⁻¹·R_B of the two body frames at creation


@dataclass
class PhysicsSetup:
    bodies: dict[str, SolverBody]
    colliders: list[ColliderEntry]
    joints: list[JointEntry]

    @property
    def has_work(self) -> bool:
        return bool(self.joints) or len(self.colliders) >= 2


def _rot(q) -> np.ndarray:
    return np.asarray(numerics.quat_to_matrix(np.asarray(q, float)), float)


def build_physics_setup(scene, body_ids: list[str], trans: np.ndarray, rot: np.ndarray, diag) -> PhysicsSetup | None:
    """`diag(level, message)` records diagnostics. Returns None when nothing uses the solver."""
    index = {oid: i for i, oid in enumerate(body_ids)}

    def driver(oid: str) -> str | None:
        for cand in [oid, *scene.ancestors(oid)]:
            if cand in index:
                return cand
        return None

    bodies: dict[str, SolverBody] = {}

    def solver_body(oid: str) -> str:
        d = driver(oid)
        key = d or oid
        if key not in bodies:
            if d is None:
                p, q = scene.world_pose(oid)
                bodies[key] = SolverBody(key, "static", None, (np.asarray(p, float), np.asarray(q, float)))
            else:
                k = index[d]
                mp = body_mass_properties(scene, d)
                com, mass = np.asarray(mp["com"], float), mp["mass"]
                role = "dynamic" if trans[k] and mass > 0 else "kinematic"
                bodies[key] = SolverBody(key, role, k, None, com)
        return key

    def relative(oid: str, key: str) -> tuple[np.ndarray, np.ndarray]:
        if oid == key:
            return np.zeros(3), np.eye(3)
        kp, kq = scene.world_pose(key)
        op, oq = scene.world_pose(oid)
        r = _rot(kq)
        return r.T @ (np.asarray(op) - np.asarray(kp)), r.T @ _rot(oq)

    colliders = []
    for obj in scene.objects.values():
        shape = collider_for(obj)
        if shape is None:
            continue
        key = solver_body(obj.id)
        rel_pos, rel_rot = relative(obj.id, key)
        colliders.append(ColliderEntry(key, obj.id, shape, rel_pos, rel_rot, float(obj.physics_setting("restitution")),
                                       float(obj.physics_setting("friction"))))
        if shape.kind == "plane" and bodies[key].role != "static":
            diag("error", f"{obj.id} ({obj.name}): plane colliders are infinite and must be static; use a box "
                          "collider for moving objects")
        if shape.approximate:
            diag("info", f"{obj.id} ({obj.name}): collides as an approximate {shape.describe()}")

    joints = []
    for j in scene.joints:
        if not j.enabled:
            continue
        missing = [b for b in (j.body_a, j.body_b) if b and b not in scene.objects]
        if missing:
            diag("error", f"{j.id} ({j.name}): connects the missing object '{missing[0]}'")
            continue
        ka = solver_body(j.body_a)
        kb = solver_body(j.body_b) if j.body_b else None
        if kb == ka:
            diag("warning", f"{j.id} ({j.name}): both ends belong to the same rigid body {ka}; the joint has no effect")
            continue
        ra_pos, ra_rot = relative(j.body_a, ka)
        anchor_a = ra_pos + ra_rot @ np.asarray(j.anchor_a, float)
        axis_a = ra_rot @ np.asarray(j.axis_a, float)
        if kb:
            rb_pos, rb_rot = relative(j.body_b, kb)
            anchor_b = rb_pos + rb_rot @ np.asarray(j.anchor_b, float)
            axis_b = rb_rot @ np.asarray(j.axis_b, float)
        else:
            anchor_b, axis_b, rb_rot = np.asarray(j.anchor_b, float), np.asarray(j.axis_b, float), np.eye(3)
        # reference relative orientation of the two *solver body* frames
        ref = ra_rot @ _rot(j.ref_orientation) @ rb_rot.T
        joints.append(JointEntry(j, ka, kb, anchor_a, anchor_b, axis_a, axis_b, ref))
        if j.type in ("ball", "revolute", "fixed", "prismatic"):
            for key, anchor in ((ka, anchor_a), (kb, anchor_b if kb else None)):
                if key is None or bodies[key].role != "dynamic":
                    continue
                inertia = body_mass_properties(scene, key)["inertia"]
                if not np.any(inertia) and np.linalg.norm(anchor - bodies[key].com_local) > 1e-9:
                    diag("warning", f"{j.id} ({j.name}): {key} has no rotational inertia (point mass), so it cannot "
                                    "turn about an anchor away from its centre and will not move; use a distance "
                                    "joint for a point mass on a rod")
        roles = {bodies[ka].role, bodies[kb].role if kb else "static"}
        if "dynamic" not in roles:
            diag("warning", f"{j.id} ({j.name}): neither end is a dynamic body (moved only by force processes), "
                            "so the joint cannot move anything")

    if not colliders and not joints:
        return None
    for key, b in bodies.items():
        if b.role == "kinematic":
            obj = scene.get(key)
            diag("info", f"{key} ({obj.name}) is moved by prescribed or update processes (or has no mass): the "
                         "rigid-body solver treats it as kinematic — it pushes other bodies but is not pushed")
        elif b.role == "dynamic" and not rot[b.engine_index]:
            diag("info", f"{key}: its orientation is prescribed, so contacts and joints change only its translation")
    return PhysicsSetup(bodies, colliders, joints)
