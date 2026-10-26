"""The scene as seen by formula processes during a simulation run.

Objects are addressed by *refs* (indices into the list of all scene objects
captured at Start). Simulated bodies read their attributes from the engine's
current solver-stage state; other objects read the values captured at Start
(they do not move during the run). Custom properties written by update
processes are kept here and published with each snapshot.
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


def global_values(scene: Scene) -> dict:
    return {g["name"]: (np.asarray(g["value"], float) if isinstance(g["value"], list) else g["value"])
            for g in scene.global_parameters}


class EngineWorld:
    def __init__(self, statics: list[StaticObject], body_ids: list[str], globals_: dict, dt: float) -> None:
        self.statics = statics
        self.ids = [s.id for s in statics]
        self.ref_index = {oid: i for i, oid in enumerate(self.ids)}
        self.set_bodies(body_ids)
        self.globals = globals_
        self.dt = dt
        self.custom_state = {i: dict(s.custom) for i, s in enumerate(statics)}
        self.custom_dirty: set[tuple[int, str]] = set()   # written since the last snapshot
        self.written: set[tuple[int, str]] = set()        # ever written by update processes this run
        self.updated_values: dict[tuple[int, str], object] = {}
        self.skips: list[str] = []
        self.x = self.q = self.v = self.w = None

    def set_bodies(self, body_ids: list[str]) -> None:
        self.body_refs = [self.ref_index[b] for b in body_ids]
        self.ref_to_body = {r: i for i, r in enumerate(self.body_refs)}

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
        if len(self.skips) < 100:
            self.skips.append(f"{process_id}: {message}")

    # ---- World protocol -------------------------------------------------------------------------------
    def _pose(self, ref: int):
        s = self.statics[ref]
        i = self.ref_to_body.get(ref)
        if i is None or self.x is None:
            return s.origin, s.q, s.v, s.w, s.origin + numerics.quat_rotate(s.q, s.com_local)
        q = self.q[i]
        com = self.x[i]
        origin = com - np.asarray(numerics.quat_rotate(q, s.com_local), float)
        return origin, q, self.v[i], self.w[i], com

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
