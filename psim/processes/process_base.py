"""Process definitions and assignments.

A *process* is an approved mathematical model evaluated during every
simulation step. Two kinds exist:

* ``contribution`` processes add forces, torques or accelerations to the
  bodies they target. Contributions to the same quantity are combined by
  summation, which is order independent, so several contributors may target
  the same object.
* ``prescribed`` processes set a state quantity (position or orientation, with
  the matching velocity) as an analytic function of time. Their writes are
  exclusive: two writers of the same quantity on the same object are a conflict.

Every process evaluates against the *previous* (shared) state of a step or of
an integrator stage, never against values other processes wrote earlier in the
same evaluation, so results never depend on accidental execution order.

A ``ProcessAssignment`` binds a process definition to target objects (by ID,
type or tag) with parameter values. Assignments are user data stored in the
project; process *definitions* are application code (built-in or, from
Phase 4, developer-authored formula processes).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..properties.property_system import PropertyDef, PropertyError

if TYPE_CHECKING:  # pragma: no cover
    from ..scene.object_base import SceneObject

# Quantities a process may write. Contributions are summed; the others are exclusive.
W_FORCE = "force"                    # translational acceleration contribution
W_TORQUE = "torque"                  # angular acceleration contribution
W_POSITION = "position"              # prescribed position (+ linear velocity)
W_ORIENTATION = "orientation"        # prescribed orientation (+ angular velocity)
SUMMED = {W_FORCE, W_TORQUE}
EXCLUSIVE = {W_POSITION, W_ORIENTATION}


class ProcessError(Exception):
    pass


@dataclass
class ProcessDef:
    process_id: str
    name: str
    version: str
    category: str
    description: str
    kind: str                                  # "contribution" | "prescribed"
    mode: str                                  # "per_object" | "global"
    parameters: list[PropertyDef]
    reads: tuple[str, ...]                     # object properties / state read
    writes: tuple[str, ...]                    # subset of W_* constants
    formula: str                               # human-readable mathematical definition
    evaluate: Callable[..., Any]
    requires_mass: bool = False
    min_targets: int = 1
    max_targets: int | None = None
    priority: int = 0                          # documentation / display ordering only (sums commute)
    eligible: Callable[["SceneObject"], bool] | None = None
    preconditions: str = ""
    error_handling: str = "The simulation stops and reports the error if a result is not finite."

    def is_eligible(self, obj: "SceneObject") -> tuple[bool, str]:
        if self.requires_mass and not obj.typedef.physical:
            return False, f"{obj.typedef.name} has no mass"
        if self.requires_mass and not obj.mass():
            return False, "mass must be positive"
        if self.eligible is not None and not self.eligible(obj):
            return False, "not an eligible object type"
        return True, ""

    def default_params(self) -> dict:
        return {p.key: p.default_value() for p in self.parameters}

    def coerce_params(self, params: dict) -> dict:
        out = self.default_params()
        known = {p.key: p for p in self.parameters}
        for k, v in params.items():
            if k not in known:
                raise PropertyError(f"{self.name}: unknown parameter '{k}'")
            out[k] = known[k].coerce(v)
        return out


@dataclass
class TargetFilter:
    """Which objects an assignment applies to (union of the listed criteria)."""

    ids: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def matches(self, obj: "SceneObject") -> bool:
        return obj.id in self.ids or obj.type_id in self.types or any(t in obj.tags for t in self.tags)

    def describe(self) -> str:
        parts = []
        if self.ids:
            parts.append(", ".join(self.ids[:3]) + ("…" if len(self.ids) > 3 else ""))
        if self.types:
            parts.append("type: " + ", ".join(self.types))
        if self.tags:
            parts.append("tag: " + ", ".join(self.tags))
        return "; ".join(parts) or "no targets"

    def to_dict(self) -> dict:
        return {"ids": list(self.ids), "types": list(self.types), "tags": list(self.tags)}

    @staticmethod
    def from_dict(d: dict) -> "TargetFilter":
        def strs(key):
            v = d.get(key, [])
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ProcessError(f"targets.{key} must be a list of strings")
            return list(v)
        return TargetFilter(strs("ids"), strs("types"), strs("tags"))


ASSIGNMENT_ID = re.compile(r"^PRC-(\d{4,})$")


@dataclass
class ProcessAssignment:
    uid: str
    process_id: str
    targets: TargetFilter
    params: dict = field(default_factory=dict)
    enabled: bool = True
    label: str = ""

    def to_dict(self) -> dict:
        return {"uid": self.uid, "process": self.process_id, "targets": self.targets.to_dict(),
                "params": copy.deepcopy(self.params), "enabled": self.enabled, "label": self.label}

    @staticmethod
    def from_dict(d: dict) -> "ProcessAssignment":
        try:
            uid = str(d["uid"])
            if not ASSIGNMENT_ID.match(uid):
                raise ProcessError(f"invalid process assignment id '{uid}'")
            params = d.get("params", {})
            if not isinstance(params, dict):
                raise ProcessError("params must be an object")
            return ProcessAssignment(uid, str(d["process"]), TargetFilter.from_dict(d.get("targets", {})),
                                     copy.deepcopy(params), bool(d.get("enabled", True)), str(d.get("label", "")))
        except KeyError as exc:
            raise ProcessError(f"process assignment is missing {exc}") from None


class StepContext:
    """What a process sees while it is evaluated for one integrator stage.

    All arrays are indexed by body index. ``acc`` and ``alpha`` are the
    accumulators that contribution processes add to; they are never read
    by processes, which keeps evaluation order irrelevant.
    """

    def __init__(self, t: float, dt: float, bodies, x, q, v, w, acc, alpha) -> None:
        self.t, self.dt = t, dt
        self.bodies = bodies          # BodySet (masses, inertia, ids, initial values)
        self.x, self.q, self.v, self.w = x, q, v, w
        self.acc, self.alpha = acc, alpha

    def add_force(self, idx: np.ndarray, force: np.ndarray) -> None:
        self.acc[idx] += np.asarray(force, float) * self.bodies.inv_mass[idx, None]

    def add_acceleration(self, idx: np.ndarray, a: np.ndarray) -> None:
        self.acc[idx] += a

    def add_torque(self, idx: np.ndarray, torque: np.ndarray) -> None:
        tau = np.broadcast_to(np.asarray(torque, float), (len(idx), 3))
        inv_world = self.bodies.inv_inertia_world(idx, self.q)
        self.alpha[idx] += np.einsum("nij,nj->ni", inv_world, tau)

    def add_angular_acceleration(self, idx: np.ndarray, alpha: np.ndarray) -> None:
        self.alpha[idx] += alpha
