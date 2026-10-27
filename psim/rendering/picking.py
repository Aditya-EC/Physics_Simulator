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
    triangle: np.ndarray | None = None     # (3, 3) world corners of the triangle that was hit
    source: tuple | None = None            # (mesh vertices, triangles, index of the hit triangle)


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
            corners = np.array([a, b, c]) @ model[:3, :3].T + model[:3, 3]
            best = PickResult(obj.id, dist, world_point, n, corners, (mesh.vertices, mesh.indices, int(tri)))
    return best


def _feature_edges(verts: np.ndarray, tris: np.ndarray, k: int, cos_tol: float = 0.9998) -> tuple:
    """For the edges of triangle k: True if the neighbouring triangle (found by vertex position, since meshes split
    vertices for flat shading) bends away by more than ~1°, or there is none."""
    t = verts[tris]                                         # (T, 3, 3)
    normals = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-300)
    tol = 1e-9 * max(float(np.max(np.ptp(verts, axis=0))), 1e-12)
    out = []
    for i in range(3):
        a, b = t[k, i], t[k, (i + 1) % 3]
        has_a = np.any(np.all(np.abs(t - a) <= tol, axis=2), axis=1)
        has_b = np.any(np.all(np.abs(t - b) <= tol, axis=2), axis=1)
        nb = np.flatnonzero(has_a & has_b)
        nb = nb[nb != k]
        out.append(not len(nb) or bool(np.min(normals[nb] @ normals[k]) < cos_tol))
    return tuple(out)


def snap_point(hit: PickResult, camera, width: float, height: float, x: float, y: float,
               radius: float = 12.0) -> tuple[np.ndarray, str]:
    """Snap a picked surface point to a corner of the triangle that was hit, else to one of its real (feature)
    edges, when the cursor (x, y) is within `radius` pixels of it. Returns (world point, "vertex" | "edge" |
    "surface")."""
    if hit.triangle is None:
        return np.asarray(hit.point, float), "surface"
    tri = np.asarray(hit.triangle, float)
    feature = (_feature_edges(np.asarray(hit.source[0], float), np.asarray(hit.source[1]), hit.source[2])
               if hit.source is not None else (True, True, True))
    xy, ok = camera.project(tri, width, height)
    cursor = np.array([x, y], float)
    d = np.where(ok, np.linalg.norm(xy - cursor, axis=1), np.inf)
    k = int(np.argmin(d))
    if d[k] <= radius:
        return tri[k].copy(), "vertex"
    o, ray = camera.screen_ray(x, y, width, height)
    best = None
    for i in range(3):
        if not feature[i]:
            continue                                         # a seam inside a flat face is no edge
        a, b = tri[i], tri[(i + 1) % 3]
        e = b - a
        # closest point of the edge to the cursor ray
        w0 = a - o
        aa, bb, cc, dd, ee = e @ e, e @ ray, ray @ ray, e @ w0, ray @ w0
        den = aa * cc - bb * bb
        t = 0.0 if den <= 1e-18 else float(np.clip((bb * ee - cc * dd) / den, 0.0, 1.0))
        q = a + t * e
        qxy, qok = camera.project(q[None, :], width, height)
        dist = float(np.linalg.norm(qxy[0] - cursor)) if qok[0] else np.inf
        if dist <= radius and (best is None or dist < best[0]):
            best = (dist, q)
    if best is not None:
        return best[1], "edge"
    return np.asarray(hit.point, float), "surface"


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
