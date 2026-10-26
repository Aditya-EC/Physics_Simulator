"""Kinematic state of scene objects and reference frames, as the GUI sees it.

* During a simulation session an object moves only if it, or an ancestor, is a
  simulated body (``scene.runtime`` holds its accelerations). A descendant of a
  simulated body moves with that body's rigid velocity field; everything else is
  at rest.
* Outside a session the velocity properties are *initial conditions*: they are
  reported as the object's velocity, and accelerations are unknown (they are
  computed by the processes once the simulation runs).
"""
from __future__ import annotations

import numpy as np

from ..scene.scene_manager import Scene
from .kinematics import FrameState, RigidState


def _com(scene: Scene, oid: str, origin: np.ndarray) -> np.ndarray:
    obj = scene.get(oid)
    if not obj.typedef.physical:
        return origin.copy()
    m = scene.world_matrix(oid)
    return (m @ np.append(np.asarray(obj.mass_properties()["com"], float), 1.0))[:3]


def driver_of(scene: Scene, oid: str) -> str | None:
    """The simulated body that moves `oid` (itself or its nearest simulated ancestor), if any."""
    for cand in [oid, *scene.ancestors(oid)]:
        if cand in scene.runtime:
            return cand
    return None


def rigid_state(scene: Scene, oid: str) -> RigidState:
    obj = scene.get(oid)
    p, q = scene.world_pose(oid)
    p, q = np.asarray(p, float), np.asarray(q, float)
    com = _com(scene, oid, p)
    driver = driver_of(scene, oid)
    if driver is not None:
        d = scene.get(driver)
        dp, dq = scene.world_pose(driver)
        rt = scene.runtime[driver]
        a, alpha = rt.get("acceleration"), rt.get("angular_acceleration")
        body = RigidState(np.asarray(dp, float), np.asarray(dq, float), _com(scene, driver, np.asarray(dp, float)),
                          np.asarray(d.linear_velocity, float), np.asarray(d.angular_velocity, float),
                          None if a is None else np.asarray(a, float), None if alpha is None else np.asarray(alpha, float))
        if driver == oid:
            return body
        return RigidState(p, q, com, body.point_velocity(com), body.angular_velocity.copy(),
                          body.point_acceleration(com),
                          None if body.angular_acceleration is None else body.angular_acceleration.copy())
    if scene.simulating:
        return RigidState.at_rest(p, q, com)
    return RigidState(p, q, com, np.asarray(obj.linear_velocity, float), np.asarray(obj.angular_velocity, float))


def reference_state(scene: Scene, ref: str | None) -> FrameState:
    """Kinematic state of a reference: None (global), an object's local frame, or a named frame."""
    if not ref:
        return FrameState.fixed()
    if ref.startswith("FRM-"):
        f = scene.frame(ref)
        if f is None:
            scene.frame_world_pose(ref)  # raises the standard "missing frame" error
        if f.parent is None:
            return FrameState.fixed(f.position, f.orientation)
        if f.parent not in scene.objects:
            scene.frame_world_pose(ref)  # raises "attached to the missing object"
        return rigid_state(scene, f.parent).attached(f.position, f.orientation, f.follow_rotation)
    return rigid_state(scene, ref).frame()


def reference_choices(scene: Scene, exclude: str | None = None, include_parent_of: str | None = None
                      ) -> list[tuple[str, str | None]]:
    """(label, ref) pairs for frame selectors: Global, parent assembly, named frames, other objects."""
    out: list[tuple[str, str | None]] = [("Global", None)]
    if include_parent_of:
        o = scene.find(include_parent_of)
        if o is not None and o.parent_id in scene.objects:
            out.append((f"Parent ({scene.objects[o.parent_id].name})", "@parent"))
    for f in scene.frames:
        out.append((f"⌖ {f.name} ({f.id})", f.id))
    for o in scene.objects.values():
        if o.id != exclude:
            out.append((f"{o.name} ({o.id})", o.id))
    return out
