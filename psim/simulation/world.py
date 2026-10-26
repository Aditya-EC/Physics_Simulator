"""The scene as seen by formula processes during a simulation run.

Objects are addressed by *refs* (indices into the list of all scene objects
captured at Start). Simulated bodies read their attributes from the engine's
current solver-stage state. A descendant of a simulated body moves rigidly
with it (its pose is composed with the body's, its velocity follows the body's
rigid velocity field); every other object is at rest during the run, so its
velocity and angular velocity are zero. Named reference frames are attached to
such objects and derive their motion the same way. Custom properties written
by update processes are kept here and published with each snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import numerics
from ..processes.formula_runtime import FormulaRuntimeError, QuatValue
from ..scene.scene_manager import Scene


@dataclass
class StaticObject:
    id: str
    type_id: str
    tags: tuple[str, ...]
    parent: str | None
    origin: np.ndarray
    q: np.ndarray
    v: np.ndarray
    w: np.ndarray
    mass: float
    density: float
    volume: float
    inertia_local: np.ndarray
    com_local: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    linear_acceleration: np.ndarray
    angular_acceleration: np.ndarray
    scale: np.ndarray
    custom: dict = field(default_factory=dict)


def _custom_value(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return np.asarray(v, float)
    return v


def capture_static(scene: Scene) -> list[StaticObject]:
    out = []
    for o in scene.objects.values():
        p, q = scene.world_pose(o.id)
        mass = density = volume = 0.0
        inertia = np.zeros((3, 3))
        com = np.zeros(3)
        if o.typedef.physical:
            mp = o.mass_properties()
            mass = mp["mass"] or 0.0
            density = mp["density"] or 0.0
            volume = mp["volume"] or 0.0
            inertia = np.asarray(mp["inertia"], float)
            com = np.asarray(mp["com"], float)
        out.append(StaticObject(
            o.id, o.type_id, tuple(o.tags), o.parent_id, np.asarray(p, float), np.asarray(q, float),
            np.asarray(o.linear_velocity, float), np.asarray(o.angular_velocity, float), mass, density, volume,
            inertia, com, np.asarray(o.force, float), np.asarray(o.torque, float),
            np.asarray(o.linear_acceleration, float), np.asarray(o.angular_acceleration, float),
            np.asarray(o.scale, float), {k: _custom_value(cp.value) for k, cp in o.custom.items()}))
    return out


@dataclass
class StaticFrame:
    id: str
    parent: int | None          # ref of the parent object, None = fixed in the global frame
    position: np.ndarray        # origin offset (parent axes) or global origin
    q: np.ndarray               # orientation offset (follow_rotation) or global orientation
    follow_rotation: bool


def capture_frames(scene: Scene, statics: list[StaticObject]) -> list[StaticFrame]:
    index = {s.id: i for i, s in enumerate(statics)}
    out = []
    for f in scene.frames:
        if f.parent is not None and f.parent not in index:
            continue  # detached frame: reported by the scheduler if a formula uses it
        out.append(StaticFrame(f.id, index.get(f.parent) if f.parent else None, np.asarray(f.position, float),
                               np.asarray(f.orientation, float), f.follow_rotation))
    return out


def global_values(scene: Scene) -> dict:
    return {g["name"]: (np.asarray(g["value"], float) if isinstance(g["value"], list) else g["value"])
            for g in scene.global_parameters}


class EngineWorld:
    def __init__(self, statics: list[StaticObject], body_ids: list[str], globals_: dict, dt: float,
                 frames: list[StaticFrame] | None = None) -> None:
        self.statics = statics
        self.ids = [s.id for s in statics]
        self.ref_index = {oid: i for i, oid in enumerate(self.ids)}
        self.frames = list(frames or [])
        self.frame_index = {f.id: i for i, f in enumerate(self.frames)}
        self.set_bodies(body_ids)
        self.globals = globals_
        self.dt = dt
        self.custom_state = {i: dict(s.custom) for i, s in enumerate(statics)}
        self.custom_dirty: set[tuple[int, str]] = set()   # written since the last snapshot
        self.written: set[tuple[int, str]] = set()        # ever written by update processes this run
        self.updated_values: dict[tuple[int, str], object] = {}
        self.skips: list[str] = []
        self.quiet = False      # True while the recorder re-evaluates processes (no duplicate skip reports)
        self.x = self.q = self.v = self.w = None

    def set_bodies(self, body_ids: list[str]) -> None:
        self.body_refs = [self.ref_index[b] for b in body_ids]
        self.ref_to_body = {r: i for i, r in enumerate(self.body_refs)}
        # each object's driver (itself or nearest simulated ancestor) and its pose in the driver's frame
        self.driver: dict[int, int] = {}
        self.rel_pose: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for ref, s in enumerate(self.statics):
            cur = s
            while cur is not None:
                d = self.ref_index.get(cur.id)
                if d in self.ref_to_body:
                    self.driver[ref] = d
                    if d != ref:
                        ds = self.statics[d]
                        ip, iq = numerics.pose_inverse(ds.origin, ds.q)
                        rp, rq = numerics.pose_compose(ip, iq, s.origin, s.q)
                        self.rel_pose[ref] = (np.asarray(rp, float), np.asarray(rq, float))
                    break
                cur = self.statics[self.ref_index[cur.parent]] if cur.parent in self.ref_index else None

    def set_stage(self, x, q, v, w) -> None:
        self.x, self.q, self.v, self.w = x, q, v, w

    # ---- ref <-> body index ------------------------------------------------------------------------
    def ref_of_body(self, i: int) -> int:
        return self.body_refs[int(i)]

    def body_of_ref(self, ref: int) -> int:
        try:
            return self.ref_to_body[ref]
        except KeyError:
            raise FormulaRuntimeError(f"{self.ids[ref]} is not simulated by this process") from None

    def report_skip(self, process_id: str, message: str) -> None:
        if not self.quiet and len(self.skips) < 100:
            self.skips.append(f"{process_id}: {message}")

    # ---- World protocol -------------------------------------------------------------------------------
    def _pose(self, ref: int):
        """(origin, q, COM velocity, angular velocity, COM) of any object at the current solver stage."""
        s = self.statics[ref]
        d = self.driver.get(ref)
        if d is None or self.x is None:
            com = s.origin + np.asarray(numerics.quat_rotate(s.q, s.com_local), float)
            return s.origin, s.q, np.zeros(3), np.zeros(3), com
        i = self.ref_to_body[d]
        q, com_b, v_b, w_b = self.q[i], self.x[i], self.v[i], self.w[i]
        if d == ref:
            origin = com_b - np.asarray(numerics.quat_rotate(q, s.com_local), float)
            return origin, q, v_b, w_b, com_b
        ds = self.statics[d]
        origin_b = com_b - np.asarray(numerics.quat_rotate(q, ds.com_local), float)
        rp, rq = self.rel_pose[ref]
        origin, q_ref = numerics.pose_compose(origin_b, q, rp, rq)
        origin, q_ref = np.asarray(origin, float), np.asarray(q_ref, float)
        com = origin + np.asarray(numerics.quat_rotate(q_ref, s.com_local), float)
        return origin, q_ref, v_b + np.cross(w_b, com - com_b), np.array(w_b, float), com

    # ---- named frames ---------------------------------------------------------------------------------
    def resolve_frame(self, frame_id: str) -> int:
        try:
            return self.frame_index[frame_id]
        except KeyError:
            raise FormulaRuntimeError(f"Missing reference frame '{frame_id}'") from None

    def frame_state(self, fref: int):
        """(origin, q, origin velocity, angular velocity) of a named frame, global."""
        f = self.frames[fref]
        if f.parent is None:
            return f.position.copy(), f.q.copy(), np.zeros(3), np.zeros(3)
        origin_p, q_p, v, w, com = self._pose(f.parent)
        origin = origin_p + np.asarray(numerics.quat_rotate(q_p, f.position), float)
        vel = v + np.cross(w, origin - com)
        if f.follow_rotation:
            return origin, np.asarray(numerics.quat_multiply(q_p, f.q), float), vel, np.array(w, float)
        return origin, f.q.copy(), vel, np.zeros(3)

    def attr(self, ref: int, name: str):
        s = self.statics[ref]
        if name in ("position", "orientation", "velocity", "linear_velocity", "angular_velocity", "com", "euler",
                    "inertia", "x_axis", "y_axis", "z_axis"):
            origin, q, v, w, com = self._pose(ref)
            if name == "position":
                return np.array(origin, float)
            if name == "orientation":
                return QuatValue(q)
            if name in ("velocity", "linear_velocity"):
                return np.array(v, float)
            if name == "angular_velocity":
                return np.array(w, float)
            if name == "com":
                return np.array(com, float)
            r = np.asarray(numerics.quat_to_matrix(q), float)
            if name == "euler":
                return np.asarray(numerics.quat_to_euler(q), float)
            if name == "inertia":
                return r @ s.inertia_local @ r.T
            return r[:, "xyz".index(name[0])].copy()
        if name == "position0":
            return s.origin.copy()
        if name == "orientation0":
            return QuatValue(s.q)
        if name == "parent":
            return self.ref_index.get(s.parent) if s.parent else None
        if name == "children":
            return self.children(ref)
        if name in ("mass", "density", "volume"):
            return float(getattr(s, name))
        mapping = {"inertia_local": s.inertia_local, "com_local": s.com_local, "force": s.force, "torque": s.torque,
                   "linear_acceleration": s.linear_acceleration, "angular_acceleration": s.angular_acceleration,
                   "scale": s.scale}
        return np.array(mapping[name], float)

    def custom(self, ref: int, key: str):
        try:
            return self.custom_state[ref][key]
        except KeyError:
            raise FormulaRuntimeError(f"{self.ids[ref]} has no custom property '{key}'") from None

    def updated(self, ref: int, quantity: str):
        return self.updated_values[(ref, quantity)]

    def resolve(self, object_id: str) -> int:
        try:
            return self.ref_index[object_id]
        except KeyError:
            raise FormulaRuntimeError(f"Missing object reference '{object_id}'") from None

    def all_refs(self) -> tuple[int, ...]:
        return tuple(range(len(self.statics)))

    def tagged(self, tag: str) -> tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.statics) if tag in s.tags)

    def of_type(self, type_id: str) -> tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.statics) if s.type_id == type_id)

    def children(self, ref: int) -> tuple[int, ...]:
        oid = self.ids[ref]
        return tuple(i for i, s in enumerate(self.statics) if s.parent == oid)
