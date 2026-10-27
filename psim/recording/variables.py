"""Plottable variables computed from a recording.

A variable names an object, a quantity, a component and (for kinematic
quantities) a reference frame, e.g. "velocity of OBJ-000002, magnitude,
relative to FRM-0001". Values are computed from the recorded samples only (a
plot never uses display-frame values) and are SI. Frame-relative values are the
motion measured by an observer moving with the frame (see
psim.coordinates.kinematics); all computations are vectorised over samples.

Quantities with an object:

    position, com, velocity, acceleration          (frame-capable; velocity/acceleration are those of the COM)
    angular_velocity, angular_acceleration, euler  (frame-capable)
    speed, displacement, distance, momentum, angular_momentum, kinetic_energy,
    net_force (m·a), net_torque (I·α, as integrated by the solver), mass
    contact_force, contact_torque, joint_force, joint_torque   (constraint loads on a simulated body: world axes,
                                                    torques about its COM, mean over each step; runs with
                                                    contacts or joints only)
    custom.<key>                                   (custom properties written by update processes)
    proc:<uid>:acceleration | force | angular_acceleration | torque   (contribution of one assignment)

Quantities without an object: time, step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .recording import Recording

VECTOR = ("x", "y", "z", "mag")


@dataclass(frozen=True)
class QuantityInfo:
    label: str
    unit_quantity: str | None     # UnitRegistry quantity key for display units
    vector: bool
    frame: bool = False           # can be measured relative to a frame


QUANTITIES: dict[str, QuantityInfo] = {
    "time": QuantityInfo("Time", "time", False),
    "step": QuantityInfo("Step", None, False),
    "position": QuantityInfo("Position", "length", True, True),
    "com": QuantityInfo("Centre of mass", "length", True, True),
    "velocity": QuantityInfo("Velocity", "velocity", True, True),
    "acceleration": QuantityInfo("Acceleration", "acceleration", True, True),
    "angular_velocity": QuantityInfo("Angular velocity", "angular_velocity", True, True),
    "angular_acceleration": QuantityInfo("Angular acceleration", "angular_acceleration", True, True),
    "euler": QuantityInfo("Orientation (Euler XYZ)", "angle", True, True),
    "speed": QuantityInfo("Speed", "velocity", False),
    "displacement": QuantityInfo("Displacement", "length", True),
    "distance": QuantityInfo("Distance travelled", "length", False),
    "momentum": QuantityInfo("Linear momentum", "momentum", True),
    "angular_momentum": QuantityInfo("Angular momentum", "angular_momentum", True),
    "kinetic_energy": QuantityInfo("Kinetic energy", "energy", False),
    "net_force": QuantityInfo("Net force", "force", True),
    "net_torque": QuantityInfo("Net torque", "torque", True),
    "mass": QuantityInfo("Mass", "mass", False),
    "contact_force": QuantityInfo("Contact force", "force", True),
    "contact_torque": QuantityInfo("Contact torque", "torque", True),
    "joint_force": QuantityInfo("Joint force", "force", True),
    "joint_torque": QuantityInfo("Joint torque", "torque", True),
}
LOAD_CHANNELS = {"contact_force": "f_contact", "contact_torque": "t_contact", "joint_force": "f_joint",
                 "joint_torque": "t_joint"}
PROCESS_OUTPUTS = {"acceleration": ("acceleration", "acc"), "force": ("force", "acc"),
                   "angular_acceleration": ("angular_acceleration", "alpha"), "torque": ("torque", "alpha")}


class VariableError(ValueError):
    pass


@dataclass(frozen=True)
class Variable:
    object_id: str | None
    quantity: str
    component: str = ""          # x | y | z | mag for vectors, "" for scalars
    frame: str | None = None     # None = global; an object id or a named frame id (relative motion)

    def to_dict(self) -> dict:
        return {"object": self.object_id, "quantity": self.quantity, "component": self.component,
                "frame": self.frame}

    @staticmethod
    def from_dict(d: dict) -> "Variable":
        return Variable(d.get("object"), str(d["quantity"]), str(d.get("component") or ""), d.get("frame"))

    @staticmethod
    def time() -> "Variable":
        return Variable(None, "time")


# ---- metadata -----------------------------------------------------------------------------------------------------

def info(var: Variable, custom_quantities: dict | None = None) -> QuantityInfo:
    q = var.quantity
    if q in QUANTITIES:
        return QUANTITIES[q]
    if q.startswith("custom."):
        key = q[7:]
        qty, is_vec = (custom_quantities or {}).get((var.object_id, key), (None, False))
        return QuantityInfo(key, qty, is_vec)
    if q.startswith("proc:"):
        _, uid, what = q.split(":", 2)
        label, _col = PROCESS_OUTPUTS[what]
        return QuantityInfo(f"{uid} {what.replace('_', ' ')}", label, True)
    raise VariableError(f"Unknown quantity '{q}'")


def label(var: Variable, names: dict | None = None, frame_names: dict | None = None,
          custom_quantities: dict | None = None) -> str:
    qi = info(var, custom_quantities)
    text = qi.label
    if var.component and qi.vector:
        text += " |·|" if var.component == "mag" else f" {var.component}"
    if var.frame:
        text += f" rel. {(frame_names or {}).get(var.frame, var.frame)}"
    if var.object_id:
        text = f"{(names or {}).get(var.object_id, var.object_id)}: {text}"
    return text


# ---- batched rotation helpers ---------------------------------------------------------------------------------------

def quat_matrices(q: np.ndarray) -> np.ndarray:
    """(n, 4) quaternions (w, x, y, z) → (n, 3, 3) rotation matrices."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def euler_from_matrices(m: np.ndarray) -> np.ndarray:
    """Extrinsic XYZ Euler angles (same convention and gimbal handling as numerics.quat_to_euler)."""
    sy = -m[:, 2, 0]
    regular = np.abs(sy) < 0.9999999
    rx = np.where(regular, np.arctan2(m[:, 2, 1], m[:, 2, 2]), 0.0)
    ry = np.where(regular, np.arcsin(np.clip(sy, -1, 1)), np.where(sy > 0, np.pi / 2, -np.pi / 2))
    rz = np.where(regular, np.arctan2(m[:, 1, 0], m[:, 0, 0]), np.arctan2(-m[:, 0, 1], m[:, 1, 1]))
    return np.stack([rx, ry, rz], 1)


def _apply_t(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rᵀ·v per sample."""
    return np.einsum("nji,nj->ni", r, v)


def _apply(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", r, v)


# ---- evaluation -------------------------------------------------------------------------------------------------------

@dataclass
class FrameSeries:
    origin: np.ndarray       # (n, 3)
    rot: np.ndarray          # (n, 3, 3)
    v: np.ndarray            # origin velocity
    w: np.ndarray
    a: np.ndarray
    alpha: np.ndarray


class Evaluator:
    """Computes variables from one recording (with a cache of per-object arrays).

    With `start` > 0 only samples from that index on are used (incremental updates); quantities that depend on
    earlier samples (displacement, distance) are then relative to the first included sample.
    """

    def __init__(self, rec: Recording, scene=None, start: int = 0) -> None:
        # one consistent snapshot: samples [start, stop) as they were when the evaluator was created, even
        # while the simulation keeps appending
        self.rec, self.scene, self.start = rec, scene, int(start)
        self.stop = rec.n
        self.generation = rec.generation
        self._cache: dict = {}

    def _col(self, key) -> np.ndarray:
        if key not in self._cache:
            try:
                col = self.rec.column(key, start=self.start, stop=self.stop)
            except KeyError:
                raise VariableError(f"'{key[1] if len(key) > 1 else key}' of {key[0]} was not recorded") from None
            if self.rec.generation != self.generation:
                raise VariableError("The recording was thinned while it was being read; please retry")
            self._cache[key] = col
        return self._cache[key]

    @property
    def n(self) -> int:
        return len(self._col("t"))

    def _mass(self, oid: str) -> float:
        return float(self.rec.meta.get("objects", {}).get(oid, {}).get("mass", 0.0))

    def _inertia_local(self, oid: str) -> np.ndarray:
        meta = self.rec.meta.get("objects", {}).get(oid, {})
        if "inertia_local" in meta:
            return np.asarray(meta["inertia_local"], float)
        return np.linalg.pinv(np.asarray(meta.get("inv_inertia_local", np.zeros((3, 3))), float))

    def rotations(self, oid: str) -> np.ndarray:
        key = ("_rot", oid)
        if key not in self._cache:
            self._cache[key] = quat_matrices(self._col((oid, "q")))
        return self._cache[key]

    # -- frames --
    def frame_series(self, ref: str | None) -> FrameSeries:
        n = self.n
        zeros = np.zeros((n, 3))
        if not ref:
            return FrameSeries(zeros, np.tile(np.eye(3), (n, 1, 1)), zeros, zeros, zeros, zeros)
        if ref.startswith("FRM-"):
            f = self.scene.frame(ref) if self.scene is not None else None
            if f is None:
                raise VariableError(f"Missing reference frame '{ref}'")
            off = np.asarray(f.position, float)
            r_off = quat_matrices(np.asarray([f.orientation], float))[0]
            if f.parent is None:
                return FrameSeries(np.tile(off, (n, 1)), np.tile(r_off, (n, 1, 1)), zeros, zeros, zeros, zeros)
            parent = self._object_frame(f.parent, point_offset=off)
            rot = np.einsum("nij,jk->nik", parent.rot, r_off) if f.follow_rotation else np.tile(r_off, (n, 1, 1))
            w = parent.w if f.follow_rotation else zeros
            alpha = parent.alpha if f.follow_rotation else zeros
            return FrameSeries(parent.origin, rot, parent.v, w, parent.a, alpha)
        return self._object_frame(ref)

    def _object_frame(self, oid: str, point_offset=None) -> FrameSeries:
        """Frame of an object (at its origin, or at a body-fixed point offset in its axes)."""
        n = self.n
        if not self.rec.has((oid, "origin")):
            # not moving during the run: its current pose in the scene
            if self.scene is None or oid not in self.scene.objects:
                raise VariableError(f"Object {oid} is neither recorded nor in the scene")
            p, q = self.scene.world_pose(oid)
            rot = quat_matrices(np.asarray([q], float))[0]
            origin = np.asarray(p, float) + (rot @ point_offset if point_offset is not None else 0.0)
            zeros = np.zeros((n, 3))
            return FrameSeries(np.tile(origin, (n, 1)), np.tile(rot, (n, 1, 1)), zeros, zeros, zeros, zeros)
        rot = self.rotations(oid)
        origin = self._col((oid, "origin"))
        if point_offset is not None:
            origin = origin + _apply(rot, np.tile(point_offset, (n, 1)))
        com, v, w = self._col((oid, "com")), self._col((oid, "v")), self._col((oid, "w"))
        acc, alpha = self._col((oid, "acc")), self._col((oid, "alpha"))
        r = origin - com
        return FrameSeries(origin, rot, v + np.cross(w, r), w, acc + np.cross(alpha, r) + np.cross(w, np.cross(w, r)),
                           alpha)

    # -- quantities --
    def vector(self, var: Variable) -> np.ndarray:
        """(n, 3) for vector quantities, (n,) for scalars, SI."""
        q, oid = var.quantity, var.object_id
        if q == "time":
            return self._col("t")[:, 0]
        if q == "step":
            return self._col("step")[:, 0]
        if oid is None:
            raise VariableError(f"'{q}' needs an object")
        if q.startswith("custom."):
            col = self._col((oid, q))
            return col[:, 0] if col.shape[1] == 1 else col
        if q.startswith("proc:"):
            _, uid, what = q.split(":", 2)
            col = self._col(("proc", uid, oid, PROCESS_OUTPUTS[what][1]))
            if what == "force":
                m = self._mass(oid)
                return col * m if m > 0 else np.full_like(col, np.nan)
            if what == "torque":
                return self._torque(oid, col)
            return col
        if q == "mass":
            return np.full(self.n, self._mass(oid))
        if var.frame and QUANTITIES.get(q, QuantityInfo("", None, False)).frame:
            return self._relative(var)
        origin = self._col((oid, "origin"))
        if q == "position":
            return origin
        if q == "com":
            return self._col((oid, "com"))
        if q == "velocity":
            return self._col((oid, "v"))
        if q == "acceleration":
            return self._col((oid, "acc"))
        if q == "angular_velocity":
            return self._col((oid, "w"))
        if q == "angular_acceleration":
            return self._col((oid, "alpha"))
        if q == "euler":
            return euler_from_matrices(self.rotations(oid))
        if q == "speed":
            return np.linalg.norm(self._col((oid, "v")), axis=1)
        if q == "displacement":
            return origin - origin[:1]
        if q == "distance":
            steps = np.linalg.norm(np.diff(origin, axis=0), axis=1)
            return np.concatenate([[0.0], np.cumsum(np.nan_to_num(steps))])
        if q == "momentum":
            return self._mass(oid) * self._col((oid, "v"))
        if q == "angular_momentum":
            return self._inertia_world_times(oid, self._col((oid, "w")))
        if q == "kinetic_energy":
            v, w = self._col((oid, "v")), self._col((oid, "w"))
            lw = self._inertia_world_times(oid, w)
            return 0.5 * self._mass(oid) * np.einsum("ni,ni->n", v, v) + 0.5 * np.einsum("ni,ni->n", w, lw)
        if q == "net_force":
            m = self._mass(oid)
            acc = self._col((oid, "acc"))
            return acc * m if m > 0 else np.full_like(acc, np.nan)
        if q == "net_torque":
            return self._torque(oid, self._col((oid, "alpha")))
        if q in LOAD_CHANNELS:
            key = (oid, LOAD_CHANNELS[q])
            if not self.rec.has(key):
                raise VariableError(f"{QUANTITIES[q].label} of {oid} was not recorded (only simulated bodies in "
                                    "runs with contacts or joints have it)")
            return self._col(key)
        raise VariableError(f"Unknown quantity '{q}'")

    def _inertia_world_times(self, oid: str, vec: np.ndarray) -> np.ndarray:
        rot = self.rotations(oid)
        local = _apply_t(rot, vec)
        return _apply(rot, local @ self._inertia_local(oid).T)

    def _torque(self, oid: str, alpha: np.ndarray) -> np.ndarray:
        i_local = self._inertia_local(oid)
        if not np.any(i_local):
            return np.full_like(alpha, np.nan)
        return self._inertia_world_times(oid, alpha)

    def _relative(self, var: Variable) -> np.ndarray:
        oid, q = var.object_id, var.quantity
        f = self.frame_series(var.frame)
        if q in ("angular_velocity", "angular_acceleration", "euler"):
            w = self._col((oid, "w"))
            if q == "angular_velocity":
                return _apply_t(f.rot, w - f.w)
            if q == "angular_acceleration":
                return _apply_t(f.rot, self._col((oid, "alpha")) - f.alpha - np.cross(f.w, w))
            return euler_from_matrices(np.einsum("nji,njk->nik", f.rot, self.rotations(oid)))
        point = self._col((oid, "origin")) if q == "position" else self._col((oid, "com"))
        r = point - f.origin
        if q in ("position", "com"):
            return _apply_t(f.rot, r)
        v = self._col((oid, "v"))
        v_rel = v - f.v - np.cross(f.w, r)
        if q == "velocity":
            return _apply_t(f.rot, v_rel)
        a = self._col((oid, "acc"))
        return _apply_t(f.rot, a - f.a - np.cross(f.alpha, r) - np.cross(f.w, np.cross(f.w, r))
                        - 2 * np.cross(f.w, v_rel))

    def values(self, var: Variable) -> np.ndarray:
        """(n,) values of a variable (component selected)."""
        v = self.vector(var)
        if v.ndim == 1:
            return v
        comp = var.component or "mag"
        if comp == "mag":
            return np.linalg.norm(v, axis=1)
        return v[:, "xyz".index(comp)]


# ---- catalogue ----------------------------------------------------------------------------------------------------------

def recorded_objects(rec: Recording) -> list[str]:
    return [k[0] for k in rec.keys() if len(k) == 2 and k[1] == "origin"]


def object_quantities(rec: Recording, oid: str) -> list[str]:
    """Quantities available for an object in this recording."""
    out = [q for q in QUANTITIES if q not in ("time", "step")
           and (q not in LOAD_CHANNELS or rec.has((oid, LOAD_CHANNELS[q])))]
    for k in rec.keys():
        if len(k) == 2 and k[0] == oid and k[1].startswith("custom."):
            out.append(k[1])
        if len(k) == 4 and k[0] == "proc" and k[2] == oid and k[3] == "acc":
            out += [f"proc:{k[1]}:{w}" for w in PROCESS_OUTPUTS]
    return out
