"""Line geometry for physics overlays: collision shapes, joints and contact points (pure NumPy, no GL)."""
from __future__ import annotations

import numpy as np

from .. import numerics
from ..physics.colliders import ColliderShape, collider_for

COLLIDER_COLOR = (0.30, 0.92, 0.55)
APPROX_COLOR = (0.95, 0.80, 0.30)
JOINT_COLOR = (0.95, 0.45, 0.95)
CONTACT_COLOR = (1.0, 0.35, 0.25)
_SEGMENTS = 32


def _circle(center, e1, e2, r: float, n: int = _SEGMENTS) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi, n + 1)
    ring = center + r * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
    return np.repeat(ring, 2, axis=0)[1:-1]          # consecutive pairs = segments


def _arc(center, e1, e2, r: float, a0: float, a1: float, n: int = _SEGMENTS // 2) -> np.ndarray:
    t = np.linspace(a0, a1, n + 1)
    ring = center + r * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
    return np.repeat(ring, 2, axis=0)[1:-1]


def shape_lines(shape: ColliderShape, position, rotation: np.ndarray, plane_size=2.0) -> np.ndarray:
    """Segment endpoints (2k, 3) of a collider placed at an object's world pose. `plane_size` is the half size
    (scalar or (x, y)) of the patch drawn for an infinite plane."""
    p = np.asarray(position, float)
    r = np.asarray(rotation, float)
    c = p + r @ shape.center
    ex, ey, ez = r[:, 0], r[:, 1], r[:, 2]
    if shape.kind == "sphere":
        return np.vstack([_circle(c, ex, ey, shape.radius), _circle(c, ey, ez, shape.radius),
                          _circle(c, ez, ex, shape.radius)])
    if shape.kind == "box":
        h = shape.half_extents
        corners = np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])
        w = c + corners @ r.T
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        return np.array([w[i] for e in edges for i in e])
    if shape.kind == "convex_parts":
        return np.vstack([shape_lines(part, position, rotation, plane_size) for part in shape.parts])
    if shape.kind == "convex":
        w = c + shape.hull.verts @ r.T
        return np.array([w[i] for e in shape.hull.edges for i in e])
    if shape.kind == "capsule":
        rad, hl = shape.radius, shape.half_length
        top, bottom = c + ez * hl, c - ez * hl
        parts = [_circle(top, ex, ey, rad), _circle(bottom, ex, ey, rad)]
        for e in (ex, ey):
            parts.append(np.array([top + e * rad, bottom + e * rad, top - e * rad, bottom - e * rad]))
            parts.append(_arc(top, e, ez, rad, 0.0, np.pi))
            parts.append(_arc(bottom, e, ez, rad, np.pi, 2 * np.pi))
        return np.vstack(parts)
    # plane: the object's own extent of the infinite plane, its centre and the outward normal
    hx, hy = (plane_size, plane_size) if np.isscalar(plane_size) else plane_size
    corners = [c + i * hx * ex + j * hy * ey for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    pts = []
    for k in range(4):
        pts += [corners[k], corners[(k + 1) % 4]]
    tick = 0.15 * min(hx, hy)
    pts += [c - tick * ex, c + tick * ex, c - tick * ey, c + tick * ey, c, c + ez * 2 * tick]
    return np.array(pts)


def collider_lines(scene, show_all: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pts, cols = [], []
    for o in scene.objects.values():
        if not o.visible or not (show_all or o.physics_setting("show_collider")):
            continue
        try:
            shape = collider_for(o)
            if shape is None:
                continue
            p, q = scene.world_pose(o.id)
        except Exception:   # an invalid collider is reported by validation, never by the renderer
            continue
        lo, hi = o.mesh().bounds()
        half = np.maximum((np.asarray(hi) - np.asarray(lo)) * np.abs(np.asarray(o.scale, float)) / 2, 0.5)
        seg = shape_lines(shape, p, numerics.quat_to_matrix(q), (float(half[0]), float(half[1])))
        pts.append(seg)
        cols.append(np.tile([*(APPROX_COLOR if shape.approximate else COLLIDER_COLOR), 1.0], (len(seg), 1)))
    return pts, cols


def joint_lines(scene, marker: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Anchor crosses, the A–B anchor link and the axis of each enabled joint at the current poses."""
    pts, cols = [], []
    for j in scene.joints:
        try:
            pa, qa = scene.world_pose(j.body_a)
            ra = np.asarray(numerics.quat_to_matrix(qa), float)
            if j.body_b:
                pb, qb = scene.world_pose(j.body_b)
                rb = np.asarray(numerics.quat_to_matrix(qb), float)
            else:
                pb, rb = np.zeros(3), np.eye(3)
        except Exception:   # dangling joints are reported by validation
            continue
        a = np.asarray(pa, float) + ra @ np.asarray(j.anchor_a, float)
        b = np.asarray(pb, float) + rb @ np.asarray(j.anchor_b, float)
        alpha = 1.0 if j.enabled else 0.35
        seg = []
        for c in (a, b):
            for i in range(3):
                e = np.zeros(3)
                e[i] = marker
                seg += [c - e, c + e]
        seg += [np.asarray(pa, float), a, a, b]
        if j.body_b:
            seg += [np.asarray(pb, float), b]
        if j.type in ("revolute", "prismatic"):
            axis = ra @ np.asarray(j.axis_a, float)
            seg += [a - axis * marker * 4, a + axis * marker * 4]
        seg = np.array(seg)
        pts.append(seg)
        cols.append(np.tile([*JOINT_COLOR, alpha], (len(seg), 1)))
    return pts, cols


def contact_lines(contacts, marker: float, force_scale: float = 0.02) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """A cross at each contact point and the normal, scaled by the contact force (at least the marker size)."""
    if not contacts:
        return [], []
    seg = []
    for point, normal, force in contacts:
        c = np.asarray(point, float)
        for i in range(3):
            e = np.zeros(3)
            e[i] = marker
            seg += [c - e, c + e]
        seg += [c, c + np.asarray(normal, float) * max(abs(force) * force_scale, marker * 3)]
    seg = np.array(seg)
    return [seg], [np.tile([*CONTACT_COLOR, 1.0], (len(seg), 1))]


MEASURE_COLOR = (0.98, 0.86, 0.35)


def measurement_geometry(scene) -> list[tuple]:
    """(measurement, value, segment endpoints (2k, 3), label anchor) of every visible, evaluable measurement."""
    from ..scene.measurements import evaluate
    out = []
    for m in scene.measurements:
        if not m.visible:
            continue
        try:
            val = evaluate(scene, m)
        except Exception:   # missing objects are shown in the Measurements panel, not drawn
            continue
        pts = val.points
        if m.kind == "distance":
            seg = np.array([pts[0], pts[1]])
            anchor = (pts[0] + pts[1]) / 2
        else:
            a, v, b = pts
            seg = [v, a, v, b]
            u, w = a - v, b - v
            nu, nw = np.linalg.norm(u), np.linalg.norm(w)
            anchor = v
            if nu > 1e-12 and nw > 1e-12 and np.isfinite(val.value):
                r = 0.3 * min(nu, nw)
                e1 = u / nu
                e2 = w - (w @ e1) * e1
                n2 = np.linalg.norm(e2)
                e2 = e2 / n2 if n2 > 1e-12 else np.zeros(3)
                seg += list(_arc(v, e1, e2, r, 0.0, val.value, 16))
                mid = val.value / 2
                anchor = v + 1.3 * r * (np.cos(mid) * e1 + np.sin(mid) * e2)
            seg = np.array(seg)
        out.append((m, val, seg, anchor))
    return out


def measurement_lines(scene) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pts, cols = [], []
    for _m, _v, seg, _a in measurement_geometry(scene):
        pts.append(seg)
        cols.append(np.tile([*MEASURE_COLOR, 1.0], (len(seg), 1)))
    return pts, cols


def measurement_labels(scene, units) -> list[tuple[np.ndarray, str]]:
    """(world anchor, text) of every visible measurement, in the project's display units."""
    lu, au = units.preferred_unit("length"), units.preferred_unit("angle")
    out = []
    for m, val, _seg, anchor in measurement_geometry(scene):
        if not np.isfinite(val.value):
            continue
        unit = lu if m.kind == "distance" else au
        out.append((anchor, f"{m.name}: {float(units.from_si(val.value, unit)):.4g} {unit}"))
    return out
