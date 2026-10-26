"""Joints: constraints between two bodies, or between a body and the fixed world.

Anchors and axes are stored in the local frames of the connected objects (their
origins), so a joint keeps its meaning when the objects move before a run. Body B
may be None (the world): its anchor and axis are then global.

Types (constrained degrees of freedom):

* ``fixed``     — all six: the two bodies move as one
* ``ball``      — the anchors coincide (3 rotational DOF free)
* ``revolute``  — ball + the hinge axes stay aligned (1 rotational DOF); optional angle limits and motor
* ``prismatic`` — no relative rotation + the anchor moves only along the axis (1 translational DOF);
                  optional travel limits and motor

Joint coordinates, limits and motor speeds describe the motion of body A relative to body B (or the
world): the rotation of A about the axis (right-hand rule), or the displacement of A's anchor along it.
* ``distance``  — the anchor distance equals ``length`` (rod), or stays ≤ ``length`` (rope)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .. import numerics

JOINT_ID = re.compile(r"^JNT-(\d{4,})$")
JOINT_TYPES = ("fixed", "ball", "revolute", "prismatic", "distance")
JOINT_KEYS = {"id", "name", "type", "body_a", "body_b", "anchor_a", "anchor_b", "axis_a", "axis_b",
              "ref_orientation", "limit_enabled", "lower", "upper", "motor_enabled", "motor_speed", "motor_max",
              "length", "rope", "collide_connected", "enabled", "description"}


class JointError(ValueError):
    pass


def _vec(value, n: int, what: str) -> list[float]:
    try:
        out = [float(x) for x in value]
    except (TypeError, ValueError):
        raise JointError(f"{what} must be a list of {n} numbers") from None
    if len(out) != n or not all(math.isfinite(x) for x in out):
        raise JointError(f"{what} must be a list of {n} finite numbers")
    return out


def _unit(value, what: str) -> list[float]:
    v = np.asarray(_vec(value, 3, what))
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise JointError(f"{what} must be a non-zero vector")
    return [float(x) for x in v / n]


@dataclass
class Joint:
    id: str
    name: str
    type: str
    body_a: str
    body_b: str | None = None
    anchor_a: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    anchor_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis_a: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    axis_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    ref_orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])   # q_a⁻¹·q_b at creation
    limit_enabled: bool = False
    lower: float = 0.0            # rad (revolute) or m (prismatic), A relative to B
    upper: float = 0.0
    motor_enabled: bool = False
    motor_speed: float = 0.0      # rad/s or m/s (A relative to B about/along the axis)
    motor_max: float = 0.0        # maximum torque (N·m) or force (N)
    length: float = 1.0           # distance joints
    rope: bool = False            # distance joints: only the maximum length is enforced
    collide_connected: bool = False
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict) -> "Joint":
        if not isinstance(d, dict):
            raise JointError("Joint entry must be an object")
        unknown = set(d) - JOINT_KEYS
        if unknown:
            raise JointError(f"Joint has unknown field(s) {', '.join(sorted(unknown))}")
        jid = str(d.get("id", ""))
        if not JOINT_ID.match(jid):
            raise JointError(f"Invalid joint id '{jid}'")
        name = str(d.get("name", "")).strip()
        if not name:
            raise JointError(f"{jid}: a joint needs a name")
        jtype = str(d.get("type", ""))
        if jtype not in JOINT_TYPES:
            raise JointError(f"{jid}: unknown joint type '{jtype}'")
        a, b = d.get("body_a"), d.get("body_b")
        if not isinstance(a, str) or not a:
            raise JointError(f"{jid}: body A is required")
        if b is not None and not isinstance(b, str):
            raise JointError(f"{jid}: body B must be an object id or null (the world)")
        if b == a:
            raise JointError(f"{jid}: a joint must connect two different bodies")
        q = np.asarray(_vec(d.get("ref_orientation", [1, 0, 0, 0]), 4, f"{jid}: reference orientation"))
        if np.linalg.norm(q) < 1e-9:
            raise JointError(f"{jid}: reference orientation must be non-zero")
        try:
            lower, upper = float(d.get("lower", 0.0)), float(d.get("upper", 0.0))
            speed, fmax = float(d.get("motor_speed", 0.0)), float(d.get("motor_max", 0.0))
            length = float(d.get("length", 1.0))
        except (TypeError, ValueError):
            raise JointError(f"{jid}: limits, motor and length must be numbers") from None
        if not all(math.isfinite(x) for x in (lower, upper, speed, fmax, length)):
            raise JointError(f"{jid}: limits, motor and length must be finite")
        if bool(d.get("limit_enabled", False)) and lower > upper:
            raise JointError(f"{jid}: the lower limit must not exceed the upper limit")
        if fmax < 0 or length < 0:
            raise JointError(f"{jid}: motor force and length must not be negative")
        return Joint(jid, name, jtype, a, b or None, _vec(d.get("anchor_a", [0, 0, 0]), 3, f"{jid}: anchor A"),
                     _vec(d.get("anchor_b", [0, 0, 0]), 3, f"{jid}: anchor B"),
                     _unit(d.get("axis_a", [0, 0, 1]), f"{jid}: axis A"), _unit(d.get("axis_b", [0, 0, 1]),
                                                                                 f"{jid}: axis B"),
                     [float(x) for x in q / np.linalg.norm(q)], bool(d.get("limit_enabled", False)), lower, upper,
                     bool(d.get("motor_enabled", False)), speed, fmax, length, bool(d.get("rope", False)),
                     bool(d.get("collide_connected", False)), bool(d.get("enabled", True)),
                     str(d.get("description", "")))


def validate_joints(data: list) -> list[Joint]:
    if not isinstance(data, list):
        raise JointError("Joints must be a list")
    joints = [Joint.from_dict(d) for d in data]
    ids, names = set(), set()
    for j in joints:
        if j.id in ids:
            raise JointError(f"Duplicate joint id '{j.id}'")
        if j.name.lower() in names:
            raise JointError(f"Duplicate joint name '{j.name}'")
        ids.add(j.id)
        names.add(j.name.lower())
    return joints


def make_joint(scene, jid: str, name: str, jtype: str, body_a: str, body_b: str | None, anchor_world,
               axis_world=(0.0, 0.0, 1.0), **options) -> dict:
    """Joint data from a world-space anchor and axis, using the bodies' current poses."""
    anchor = np.asarray(anchor_world, float)
    axis = np.asarray(axis_world, float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    pa, qa = scene.world_pose(body_a)
    ra = np.asarray(numerics.quat_to_matrix(qa), float)
    if body_b:
        pb, qb = scene.world_pose(body_b)
    else:
        pb, qb = np.zeros(3), np.array([1.0, 0, 0, 0])
    rb = np.asarray(numerics.quat_to_matrix(qb), float)
    data = {"id": jid, "name": name, "type": jtype, "body_a": body_a, "body_b": body_b,
            "anchor_a": list(ra.T @ (anchor - pa)), "anchor_b": list(rb.T @ (anchor - pb)),
            "axis_a": list(ra.T @ axis), "axis_b": list(rb.T @ axis),
            "ref_orientation": list(numerics.quat_multiply(numerics.quat_conjugate(qa), qb))}
    other = options.pop("anchor_world_b", None)
    if other is not None:   # a second anchor point (distance joints): the length defaults to the current distance
        data["anchor_b"] = list(rb.T @ (np.asarray(other, float) - pb))
        data["length"] = float(np.linalg.norm(np.asarray(other, float) - anchor))
    data.update(options)
    return Joint.from_dict(data).to_dict()


def is_joint_id(ref: str | None) -> bool:
    return bool(ref) and bool(JOINT_ID.match(ref))
