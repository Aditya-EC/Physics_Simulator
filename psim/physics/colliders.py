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
* ``convex_parts`` — a compound of convex hulls from an approximate convex decomposition of the mesh
  (psim/physics/decompose.py, native/src/decompose.cpp); each part collides as a ``convex`` shape

"auto" picks the exact shape where the type has one (sphere, cube/cuboid, capsule, plane), the convex
hull for convex meshes, and convex parts for concave ones (a torus, gear, I-beam, …). The hull is exact
for convex polyhedral meshes (cylinders, cones, prisms, pyramids, wedges, …); shapes are marked
approximate when curved meshes had to be reduced to support points, or when the parts' hulls still
exceed the mesh volume by more than 1 %.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COLLIDER_KINDS = ("none", "auto", "sphere", "box", "capsule", "plane", "convex", "convex_parts")
CONCAVE = 0.01           # a mesh whose hull exceeds its volume by more than this fraction is concave
DECOMPOSITION_HULL_POINTS = 128   # pieces are compared through the hulls of their support points (both backends)
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
    parts: list | None = None     # "convex_parts": the parts as "convex" shapes (in the same local frame)

    def bounding_radius(self) -> float:
        if self.kind == "sphere":
            return self.radius
        if self.kind in ("box", "convex"):
            return float(np.linalg.norm(self.half_extents))
        if self.kind == "convex_parts":
            return max(float(np.linalg.norm(p.center - self.center)) + p.bounding_radius() for p in self.parts)
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
        elif self.kind == "convex_parts":
            text = f"{len(self.parts)} convex parts"
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
    if kind in ("convex", "convex_parts"):
        return _convex(obj, scale, auto, kind == "convex_parts")
    over = np.asarray(obj.physics_setting("half_extents"), float)
    half = np.where(over > 0, over, half_auto)
    if obj.type_id in ("basic.plane", "basic.disk"):
        half = np.maximum(half, [1e-3, 1e-3, 1e-3])
    return ColliderShape("box", center, half_extents=half, approximate=approx,
                         note="bounding box of the mesh" if approx else "")


_PARTS_CACHE: dict[bytes, list] = {}


def _mesh_arrays(obj, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mesh = obj.mesh()
    verts = np.asarray(mesh.vertices, float) * scale + np.asarray(obj.physics_setting("offset"), float)
    if np.prod(np.sign(scale)) < 0:         # a mirroring scale flips the triangles' orientation
        tris = np.asarray(mesh.indices, np.int64)[:, ::-1]
    else:
        tris = np.asarray(mesh.indices, np.int64)
    return verts, np.ascontiguousarray(tris)


def _hull_shape(points, approx: bool, note: str):
    from .hull import convex_hull, translated
    hull = convex_hull(points)
    lo, hi = hull.verts.min(axis=0), hull.verts.max(axis=0)
    center = (lo + hi) / 2
    return ColliderShape("convex", center, half_extents=np.maximum((hi - lo) / 2, 1e-6), approximate=approx,
                         note=note, hull=translated(hull, -center))


def _decomposition(verts: np.ndarray, tris: np.ndarray) -> list:
    """Convex part point sets of a closed mesh, cached by its exact geometry."""
    from .. import numerics
    key = verts.tobytes() + tris.tobytes()
    parts = _PARTS_CACHE.get(key)
    if parts is None:
        if len(_PARTS_CACHE) > 64:
            _PARTS_CACHE.clear()
        parts = _PARTS_CACHE[key] = numerics.convex_decomposition(verts, tris,
                                                                    max_hull_points=DECOMPOSITION_HULL_POINTS)
    return parts


_SHAPE_CACHE: dict[bytes, ColliderShape] = {}


def _convex(obj, scale: np.ndarray, auto: bool, parts_requested: bool) -> ColliderShape:
    """Convex hull or convex parts of the scaled mesh, cached by its exact geometry and the mode (the overlay asks
    for the collider of every shown object at every frame)."""
    verts, tris = _mesh_arrays(obj, scale)
    key = verts.tobytes() + tris.tobytes() + bytes([auto, parts_requested])
    shape = _SHAPE_CACHE.get(key)
    if shape is None:
        if len(_SHAPE_CACHE) > 256:
            _SHAPE_CACHE.clear()
        shape = _SHAPE_CACHE[key] = _convex_shape(verts, tris, auto, parts_requested)
    return shape


def _convex_shape(verts: np.ndarray, tris: np.ndarray, auto: bool, parts_requested: bool) -> ColliderShape:
    from .. import numerics
    from .decompose import chain_volume
    from .hull import mesh_hull, translated
    hull = mesh_hull(verts)
    volume = chain_volume(verts[tris]) if len(tris) else 0.0
    exact_hull = float(numerics.hull_volume(verts))
    concave = volume > 0 and exact_hull > (1 + CONCAVE) * volume
    if parts_requested or (auto and concave):
        if volume <= 0:
            note = "the mesh is not a closed solid, so it cannot be split into convex parts"
        else:
            points = _decomposition(verts, tris)
            if len(points) > 1 or parts_requested:
                shapes = [_hull_shape(p, False, "") for p in points]
                total = sum(float(numerics.hull_volume(p)) for p in points)
                excess = total / volume - 1
                reduced = any(p.hull.reduced for p in shapes)
                notes = []
                if excess > CONCAVE:
                    notes.append(f"the parts exceed the mesh volume by {100 * excess:.0f} %")
                if reduced:
                    notes.append("curved parts reduced to support points")
                lo = np.min([p.center - p.half_extents for p in shapes], axis=0)
                hi = np.max([p.center + p.half_extents for p in shapes], axis=0)
                return ColliderShape("convex_parts", (lo + hi) / 2, half_extents=(hi - lo) / 2,
                                     approximate=bool(notes), note="; ".join(notes), parts=shapes)
            note = ""
    else:
        note = ""
    lo, hi = hull.verts.min(axis=0), hull.verts.max(axis=0)
    center = (lo + hi) / 2
    approx = hull.reduced or concave
    if concave:
        note = note or "the mesh is not convex"
    elif hull.reduced:
        note = "from support points of a detailed mesh"
    return ColliderShape("convex", center, half_extents=np.maximum((hi - lo) / 2, 1e-6),
                         approximate=approx, note=note, hull=translated(hull, -center))
