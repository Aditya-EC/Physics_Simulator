"""Simulation bodies and state arrays.

Translational state is the global centre-of-mass position and velocity
(COM, ``x`` / ``v``); rotational state is the global orientation quaternion
``q`` and world-frame angular velocity ``w``. Object origins are recovered as
``origin = x - R c`` with ``c`` the COM in the object's local frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import numerics
from ..scene.scene_manager import Scene


@dataclass
class BodySet:
    ids: list[str]
    mass: np.ndarray                  # (N,)  0 for massless objects
    inv_mass: np.ndarray              # (N,)
    com_local: np.ndarray             # (N, 3)
    inv_inertia_local: np.ndarray     # (N, 3, 3) pseudo-inverse about the COM (zeros for point masses)
    linear_acceleration: np.ndarray   # (N, 3) user input properties captured at build time
    angular_acceleration: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    origin0: np.ndarray               # (N, 3) object origin at the start of the run
    q0: np.ndarray                    # (N, 4)
    world: object = None              # EngineWorld for formula processes (set by the engine)

    @property
    def index(self) -> dict[str, int]:
        return {oid: i for i, oid in enumerate(self.ids)}

    def inv_inertia_world(self, idx: np.ndarray, q: np.ndarray) -> np.ndarray:
        return numerics.rotate_tensors(q[idx], self.inv_inertia_local[idx])  # R I⁻¹ Rᵀ (native)

    def com_offset_world(self, q: np.ndarray) -> np.ndarray:
        if not self.has_com_offset:
            return np.zeros((len(q), 3))
        return numerics.quat_rotate_batch(q, self.com_local)

    @property
    def has_com_offset(self) -> bool:
        return bool(np.any(self.com_local))


@dataclass
class SimState:
    x: np.ndarray   # (N, 3) COM position
    q: np.ndarray   # (N, 4) orientation
    v: np.ndarray   # (N, 3) COM velocity
    w: np.ndarray   # (N, 3) angular velocity (world)

    def copy(self) -> "SimState":
        return SimState(self.x.copy(), self.q.copy(), self.v.copy(), self.w.copy())


def capture_bodies(scene: Scene, ids: list[str]) -> tuple[BodySet, SimState]:
    """Read the simulation inputs of `ids` from the scene (GUI thread)."""
    n = len(ids)
    mass = np.zeros(n)
    com = np.zeros((n, 3))
    inv_i = np.zeros((n, 3, 3))
    la, aa, fo, to = (np.zeros((n, 3)) for _ in range(4))
    origin = np.zeros((n, 3))
    q = np.zeros((n, 4))
    v = np.zeros((n, 3))
    w = np.zeros((n, 3))
    for k, oid in enumerate(ids):
        o = scene.get(oid)
        p, qq = scene.world_pose(oid)
        origin[k], q[k] = p, qq
        if o.typedef.physical:
            mp = o.mass_properties()
            mass[k] = mp["mass"] or 0.0
            com[k] = mp["com"]
            inertia = np.asarray(mp["inertia"], float)
            if np.any(inertia):
                inv_i[k] = np.linalg.pinv(inertia)
        la[k], aa[k] = o.linear_acceleration, o.angular_acceleration
        fo[k], to[k] = o.force, o.torque
        v[k], w[k] = o.linear_velocity, o.angular_velocity
    inv_m = np.divide(1.0, mass, out=np.zeros(n), where=mass > 0)
    bodies = BodySet(list(ids), mass, inv_m, com, inv_i, la, aa, fo, to, origin.copy(), q.copy())
    x = origin + bodies.com_offset_world(q)
    return bodies, SimState(np.ascontiguousarray(x), np.ascontiguousarray(q), np.ascontiguousarray(v),
                            np.ascontiguousarray(w))


def origins(bodies: BodySet, state: SimState) -> np.ndarray:
    return state.x - bodies.com_offset_world(state.q)

