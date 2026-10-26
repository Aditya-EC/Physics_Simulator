"""Ray picking against scene objects and drag-and-drop placement computation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import numerics
from ..scene.scene_manager import Scene
from .grid import PLANE_AXES


@dataclass
class PickResult:
    object_id: str
    distance: float
    point: np.ndarray
    normal: np.ndarray


def pick(scene: Scene, origin, direction, exclude: set[str] | None = None) -> PickResult | None:
    """Closest visible object hit by the world-space ray (uses the native ray caster)."""
    origin = np.asarray(origin, float)
    direction = np.asarray(direction, float)
    best: PickResult | None = None
    for obj in scene.objects.values():
        if exclude and obj.id in exclude:
            continue
        if not scene.is_effectively_visible(obj.id) or not (obj.show_surface or obj.wireframe):
            continue
        model = scene.world_matrix(obj.id)
        # bounding-sphere rejection
        center = model[:3, 3]
        r = obj.bounding_radius() * 1.01
        oc = center - origin
        tca = float(oc @ direction)
        if float(oc @ oc) - tca * tca > r * r and float(oc @ oc) > r * r:
            continue
        inv = np.linalg.inv(model)
        lo = inv[:3, :3] @ origin + inv[:3, 3]
        ld = inv[:3, :3] @ direction
        mesh = obj.mesh()
        hit, t, tri = numerics.ray_mesh_intersect(lo, ld, mesh.vertices, mesh.indices)
        if not hit:
            continue
        local_point = lo + t * ld
        world_point = model[:3, :3] @ local_point + model[:3, 3]
        dist = float(np.linalg.norm(world_point - origin))
        if best is None or dist < best.distance:
            a, b, c = (mesh.vertices[i].astype(float) for i in mesh.indices[tri])
            n_local = np.cross(b - a, c - a)
            n = np.linalg.inv(model[:3, :3]).T @ n_local
            n /= max(np.linalg.norm(n), 1e-12)
            if n @ direction > 0:
                n = -n
            best = PickResult(obj.id, dist, world_point, n)
    return best


def support_distance(lo, hi, direction) -> float:
    """Distance from the origin to the supporting plane of box [lo, hi] in `direction` (unit)."""
    d = np.asarray(direction, float)
    return float(np.sum(np.maximum(d * np.asarray(lo, float), d * np.asarray(hi, float))))


def placement_position(scene: Scene, origin, direction, mesh_lo, mesh_hi, work_plane: str, plane_offset: float,
                       grid_snap: bool, increment: float, surface_snap: bool,
                       fallback=None) -> tuple[np.ndarray, str | None]:
    """Where to place a new object so that it rests on a surface or on the work plane.

    Returns (world position of the object's local origin, id of the surface object or None).
    """
    origin = np.asarray(origin, float)
    direction = np.asarray(direction, float)
    hit = pick(scene, origin, direction) if surface_snap else None
    if hit is not None:
        n = hit.normal
        base = hit.point
        surface_id = hit.object_id
    else:
        _, _, ni = PLANE_AXES[work_plane]
        n = np.zeros(3)
        n[ni] = 1.0
        den = float(direction @ n)
        if abs(den) < 1e-9:
            base = np.asarray(fallback if fallback is not None else np.zeros(3), float)
        else:
            t = (plane_offset - float(origin @ n)) / den
            base = origin + t * direction if t > 0 else np.asarray(fallback if fallback is not None else np.zeros(3), float)
        if float(direction @ n) > 0:  # looking at the plane from below: rest on the underside
            n = -n
        surface_id = None
    pos = base + n * support_distance(mesh_lo, mesh_hi, -n)
    if grid_snap and increment > 0:
        for k in range(3):
            if abs(n[k]) < 0.1:  # only snap tangential coordinates
                pos[k] = round(pos[k] / increment) * increment
    return pos, surface_id


def objects_in_rect(scene: Scene, camera, width: float, height: float, x0: float, y0: float, x1: float,
                    y1: float) -> list[str]:
    """Visible objects whose world bounding-box centre projects inside the screen rectangle (any corner order).

    Used by box selection; object order follows the scene (deterministic)."""
    lo_x, hi_x = sorted((x0, x1))
    lo_y, hi_y = sorted((y0, y1))
    ids, centres = [], []
    for oid in scene.objects:
        if not scene.is_effectively_visible(oid):
            continue
        b = scene.world_bounds([oid])
        if b is None:
            continue
        ids.append(oid)
        centres.append((np.asarray(b[0], float) + np.asarray(b[1], float)) / 2)
    if not ids:
        return []
    xy, ok = camera.project(np.array(centres), width, height)
    return [oid for oid, (x, y), vis in zip(ids, xy, ok)
            if vis and lo_x <= x <= hi_x and lo_y <= y <= hi_y]
