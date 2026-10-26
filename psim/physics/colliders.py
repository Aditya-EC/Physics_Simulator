"""Collision geometry, kept separate from the visual mesh.

An object takes part in collisions only if it has a collider (``physics.collider``
≠ "none"). Types flagged as rigid bodies default to an automatic collider; every
other object needs an explicit choice, so a mass property alone never turns an
object into a colliding rigid body.

Shapes (in the object's local frame, scale applied):

* ``sphere``  — centre + radius
* ``box``     — centre + half extents along the local axes (oriented box)
* ``capsule`` — centre + radius + half length of the core segment along local Z
* ``plane``   — the object's local XY plane, solid below (an infinite half-space; static only)
* ``convex``  — the convex hull of the (scaled) mesh vertices, centred on its bounding box

"auto" picks the exact shape where the type has one (sphere, cube/cuboid, capsule, plane) and
otherwise the convex hull of the mesh. The hull is exact for convex polyhedral meshes (cylinders,
cones, prisms, pyramids, wedges, …); it is marked approximate when the mesh is not convex or had to
be reduced to support points (curved meshes with many vertices).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COLLIDER_KINDS = ("none", "auto", "sphere", "box", "capsule", "plane", "convex")
_EXACT = {"basic.sphere": "sphere", "basic.cube": "box", "basic.cuboid": "box", "body.rigid_box": "box",
          "basic.capsule": "capsule", "basic.plane": "plane", "basic.disk": "plane", "body.point_mass": "sphere"}


@dataclass
class ColliderShape:
    kind: str
    center: np.ndarray            # local frame (scaled)
    radius: float = 0.0
    half_extents: np.ndarray | None = None
    half_length: float = 0.0      # capsule core half length (local Z)
    approximate: bool = False
    note: str = ""
    hull: object = None           # psim.physics.hull.Hull for "convex" (vertices relative to `center`)

    def bounding_radius(self) -> float:
        if self.kind == "sphere":
            return self.radius
        if self.kind in ("box", "convex"):
            return float(np.linalg.norm(self.half_extents))
        if self.kind == "capsule":
            return self.radius + self.half_length
        return float("inf")

    def describe(self) -> str:
        if self.kind == "sphere":
            text = f"sphere r = {self.radius:.4g} m"
        elif self.kind == "box":
            text = "box " + " × ".join(f"{2 * h:.4g}" for h in self.half_extents) + " m"
        elif self.kind == "capsule":
            text = f"capsule r = {self.radius:.4g} m, length {2 * self.half_length:.4g} m"
        elif self.kind == "convex":
            text = f"convex hull ({len(self.hull.verts)} vertices, {len(self.hull.faces)} faces)"
        else:
            text = "infinite plane (local XY, solid below)"
        return text + (" — approximate" if self.approximate else "") + (f" ({self.note})" if self.note else "")


def collider_kind(obj) -> str:
    kind = obj.physics_setting("collider")
    if kind == "auto":
        return _EXACT.get(obj.type_id, "convex")
    return kind


def collider_for(obj) -> ColliderShape | None:
    """The collision shape of an object, or None if it does not collide."""
    if obj.physics_setting("collider") == "none":
        return None
    auto = obj.physics_setting("collider") == "auto"
    kind = collider_kind(obj)
    scale = np.abs(np.asarray(obj.scale, float))
    lo, hi = obj.mesh().bounds()
    lo, hi = lo * scale, hi * scale
    center = (lo + hi) / 2 + np.asarray(obj.physics_setting("offset"), float)
    half_auto = np.maximum((hi - lo) / 2, 1e-6)
    approx = auto and obj.type_id not in _EXACT
    note = ""
    if kind == "plane":
        return ColliderShape("plane", np.asarray(obj.physics_setting("offset"), float))
    if kind == "sphere":
        r = obj.physics_setting("radius")
        if r <= 0:
            r = float(np.max(half_auto))
            if not np.allclose(half_auto, half_auto[0], rtol=1e-6):
                approx, note = True, "fitted to the largest extent"
        return ColliderShape("sphere", center, radius=float(r), approximate=approx, note=note)
    if kind == "capsule":
        r = obj.physics_setting("radius")
        if r <= 0:
            r = float(max(half_auto[0], half_auto[1]))
        half_len = obj.physics_setting("length") / 2
        if half_len <= 0:
            half_len = max(float(half_auto[2]) - r, 0.0)
        return ColliderShape("capsule", center, radius=float(r), half_length=float(half_len), approximate=approx)
    if kind == "convex":
        return _convex(obj, scale, auto)
    over = np.asarray(obj.physics_setting("half_extents"), float)
    half = np.where(over > 0, over, half_auto)
    if obj.type_id in ("basic.plane", "basic.disk"):
        half = np.maximum(half, [1e-3, 1e-3, 1e-3])
    return ColliderShape("box", center, half_extents=half, approximate=approx,
                         note="bounding box of the mesh" if approx else "")


def _convex(obj, scale: np.ndarray, auto: bool) -> ColliderShape:
    from .hull import mesh_hull, translated
    mesh = obj.mesh()
    verts = np.asarray(mesh.vertices, float) * scale + np.asarray(obj.physics_setting("offset"), float)
    hull = mesh_hull(verts)
    lo, hi = hull.verts.min(axis=0), hull.verts.max(axis=0)
    center = (lo + hi) / 2
    approx, note = hull.reduced, "from support points of a detailed mesh" if hull.reduced else ""
    mp = obj.mass_properties() if obj.typedef.physical else {}
    vol = mp.get("volume") or 0.0
    if not hull.reduced and vol > 0 and abs(hull.volume - vol) > 0.01 * vol:
        approx, note = True, "the mesh is not convex"
    return ColliderShape("convex", center, half_extents=np.maximum((hi - lo) / 2, 1e-6),
                         approximate=approx, note=note, hull=translated(hull, -center))
