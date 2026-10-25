"""Transform gizmo: handle geometry, picking and drag mathematics.

The gizmo is drawn at the pivot (primary selected object) with a constant
on-screen size. Translation and rotation handles can be aligned with global
axes or with the primary object's local frame; scaling always acts along local
axes. All maths here is pure NumPy and independent of OpenGL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import numerics
from ..scene import geometry as g
from .view_settings import SnapSettings

AXIS_COLORS = [(0.93, 0.32, 0.30), (0.45, 0.82, 0.34), (0.33, 0.57, 0.98)]
CENTER_COLOR = (0.88, 0.88, 0.90)
HOVER_COLOR = (1.0, 0.85, 0.25)
GIZMO_PIXELS = 105.0

_ROT_Z_TO = {  # rotations taking +Z to each axis
    0: numerics.quat_from_axis_angle([0, 1, 0], math.pi / 2),
    1: numerics.quat_from_axis_angle([1, 0, 0], -math.pi / 2),
    2: np.array([1.0, 0.0, 0.0, 0.0]),
}


def _rot(q) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = numerics.quat_to_matrix(q)
    return m


@dataclass
class Handle:
    name: str           # e.g. "t_x", "t_xy", "t_free", "r_z", "s_y", "s_uniform"
    kind: str           # translate | rotate | scale
    axis: int | None    # axis index, or None
    plane: tuple[int, int] | None
    color: tuple[float, float, float]
    mesh: g.Mesh
    pick: g.Mesh


def _arrow_along(axis: int, length: float, r: float, head_r: float, head_len: float) -> g.Mesh:
    shaft = g.translated(g.cylinder(r, length - head_len, 12), [0, 0, (length - head_len) / 2])
    head = g.translated(g.cone(head_r, head_len, 20), [0, 0, length - head_len / 2])
    return g.merge([shaft, head]).transformed(_rot(_ROT_Z_TO[axis]))


def _scale_along(axis: int, length: float, r: float, cube: float) -> g.Mesh:
    shaft = g.translated(g.cylinder(r, length - cube, 12), [0, 0, (length - cube) / 2])
    tip = g.translated(g.box(cube, cube, cube), [0, 0, length - cube / 2])
    return g.merge([shaft, tip]).transformed(_rot(_ROT_Z_TO[axis]))


def _ring(axis: int, radius: float, tube: float) -> g.Mesh:
    return g.torus(radius, tube, 64, 8).transformed(_rot(_ROT_Z_TO[axis]))


def _plane_square(a: int, b: int, lo: float, hi: float) -> g.Mesh:
    c = np.zeros(3)
    c[a] = c[b] = (lo + hi) / 2
    size = [0.004] * 3
    size[a] = size[b] = hi - lo
    return g.translated(g.box(*size), c)


def build_handles() -> dict[str, list[Handle]]:
    t, r, s = [], [], []
    names = "xyz"
    for i in range(3):
        t.append(Handle(f"t_{names[i]}", "translate", i, None, AXIS_COLORS[i],
                        _arrow_along(i, 1.0, 0.012, 0.05, 0.2), _arrow_along(i, 1.0, 0.06, 0.09, 0.25)))
    for a, b in ((0, 1), (1, 2), (0, 2)):
        normal = 3 - a - b
        t.append(Handle(f"t_{names[a]}{names[b]}", "translate", None, (a, b), AXIS_COLORS[normal],
                        _plane_square(a, b, 0.22, 0.42), _plane_square(a, b, 0.2, 0.44)))
    t.append(Handle("t_free", "translate", None, None, CENTER_COLOR, g.sphere(0.06, 20, 10), g.sphere(0.1, 12, 6)))
    for i in range(3):
        r.append(Handle(f"r_{names[i]}", "rotate", i, None, AXIS_COLORS[i], _ring(i, 0.9, 0.012), _ring(i, 0.9, 0.07)))
    for i in range(3):
        s.append(Handle(f"s_{names[i]}", "scale", i, None, AXIS_COLORS[i],
                        _scale_along(i, 0.95, 0.012, 0.09), _scale_along(i, 0.95, 0.06, 0.14)))
    s.append(Handle("s_uniform", "scale", None, None, CENTER_COLOR, g.box(0.12, 0.12, 0.12), g.box(0.16, 0.16, 0.16)))
    return {"translate": t, "rotate": r, "scale": s}


def gizmo_matrix(center, axes: np.ndarray, size: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = axes * size
    m[:3, 3] = center
    return m


def pick_handle(handles: list[Handle], model: np.ndarray, origin, direction) -> str | None:
    inv = np.linalg.inv(model)
    o = inv[:3, :3] @ np.asarray(origin, float) + inv[:3, 3]
    d = inv[:3, :3] @ np.asarray(direction, float)
    best, best_t = None, math.inf
    for h in handles:
        hit, t, _ = numerics.ray_mesh_intersect(o, d, h.pick.vertices, h.pick.indices)
        if hit and t < best_t:
            best, best_t = h.name, t
    return best


# ---- drag mathematics -------------------------------------------------------------

def closest_param_on_axis(center, axis, origin, direction) -> float | None:
    """Parameter s of the point center + s*axis closest to the ray (None if nearly parallel)."""
    a = np.asarray(axis, float)
    d = np.asarray(direction, float)
    w = np.asarray(center, float) - np.asarray(origin, float)
    b = float(a @ d)
    den = 1.0 - b * b
    if den < 1e-6:
        return None
    return (b * float(d @ w) - float(a @ w)) / den


def ray_plane(origin, direction, point, normal) -> np.ndarray | None:
    n = np.asarray(normal, float)
    den = float(np.asarray(direction, float) @ n)
    if abs(den) < 1e-8:
        return None
    t = float((np.asarray(point, float) - np.asarray(origin, float)) @ n) / den
    if t < 0:
        return None
    return np.asarray(origin, float) + t * np.asarray(direction, float)


def _snap(value: float, inc: float) -> float:
    return round(value / inc) * inc if inc > 0 else value


@dataclass
class DragState:
    handle: Handle
    pivot: np.ndarray
    axes: np.ndarray                       # columns = gizmo axes in world space
    view_forward: np.ndarray
    start: dict[str, tuple]                # id -> (world pos, world quat, scale)
    primary: str
    global_frame: bool
    ref_param: float = 0.0
    ref_point: np.ndarray | None = None
    prev_vec: np.ndarray | None = None
    angle: float = 0.0
    extra: dict = field(default_factory=dict)


def begin_drag(handle: Handle, pivot, axes, view_forward, start, primary, global_frame, origin, direction) -> DragState | None:
    st = DragState(handle, np.asarray(pivot, float), np.asarray(axes, float), np.asarray(view_forward, float),
                   start, primary, global_frame)
    if handle.kind in ("translate", "scale") and handle.axis is not None:
        s = closest_param_on_axis(st.pivot, st.axes[:, handle.axis], origin, direction)
        if s is None:
            return None
        st.ref_param = s
    elif handle.kind == "rotate":
        n = st.axes[:, handle.axis]
        p = ray_plane(origin, direction, st.pivot, n) if abs(float(n @ direction)) > 0.05 else None
        if p is None:
            p = _view_plane_hit(st, origin, direction)
            if p is None:
                return None
        v = p - st.pivot
        v -= n * float(v @ n)
        if np.linalg.norm(v) < 1e-9:
            return None
        st.prev_vec = v
    else:
        normal = _plane_normal(st)
        p = ray_plane(origin, direction, st.pivot, normal)
        if p is None:
            return None
        st.ref_point = p
    return st


def _plane_normal(st: DragState) -> np.ndarray:
    h = st.handle
    if h.plane is not None:
        return st.axes[:, 3 - h.plane[0] - h.plane[1]]
    return st.view_forward


def _view_plane_hit(st: DragState, origin, direction):
    return ray_plane(origin, direction, st.pivot, st.view_forward) if abs(float(st.view_forward @ direction)) > 1e-6 \
        else None


def update_drag(st: DragState, origin, direction, snap: SnapSettings) -> dict[str, tuple] | None:
    """Return new (world pos, world quat, scale) for every dragged object, or None if no update."""
    h = st.handle
    if h.kind == "translate":
        if h.axis is not None:
            s = closest_param_on_axis(st.pivot, st.axes[:, h.axis], origin, direction)
            if s is None:
                return None
            delta = st.axes[:, h.axis] * (s - st.ref_param)
        else:
            p = ray_plane(origin, direction, st.pivot, _plane_normal(st))
            if p is None:
                return None
            delta = p - st.ref_point
        if snap.grid_snap and snap.translate_increment > 0:
            delta = _snap_translation(st, delta, snap.translate_increment)
        return {i: (p + delta, q, sc) for i, (p, q, sc) in st.start.items()}

    if h.kind == "rotate":
        n = st.axes[:, h.axis]
        p = ray_plane(origin, direction, st.pivot, n) if abs(float(n @ direction)) > 0.05 else None
        if p is None:
            p = _view_plane_hit(st, origin, direction)
            if p is None:
                return None
        v = p - st.pivot
        v -= n * float(v @ n)
        if np.linalg.norm(v) < 1e-9:
            return None
        st.angle += math.atan2(float(n @ np.cross(st.prev_vec, v)), float(st.prev_vec @ v))
        st.prev_vec = v
        angle = st.angle
        if snap.rotate_snap and snap.rotate_increment_deg > 0:
            angle = _snap(angle, math.radians(snap.rotate_increment_deg))
        qd = numerics.quat_from_axis_angle(n, angle)
        rm = np.asarray(numerics.quat_to_matrix(qd))
        out = {}
        for i, (pos, q, sc) in st.start.items():
            newp = st.pivot + rm @ (pos - st.pivot)
            out[i] = (newp, np.asarray(numerics.quat_multiply(qd, q)), sc)
        st.extra["angle"] = angle
        return out

    # scale
    if h.axis is not None:
        s = closest_param_on_axis(st.pivot, st.axes[:, h.axis], origin, direction)
        if s is None or abs(st.ref_param) < 1e-9:
            return None
        factor = s / st.ref_param
    else:
        p = ray_plane(origin, direction, st.pivot, st.view_forward)
        if p is None:
            return None
        r0 = float(np.linalg.norm(st.ref_point - st.pivot)) if st.ref_point is not None else 0.0
        if r0 < 1e-9:
            return None
        factor = float(np.linalg.norm(p - st.pivot)) / r0
    factor = max(factor, 1e-3)
    out = {}
    for i, (pos, q, sc) in st.start.items():
        sc = np.asarray(sc, float).copy()
        idx = [h.axis] if h.axis is not None else [0, 1, 2]
        for k in idx:
            val = sc[k] * factor
            if snap.scale_snap and snap.scale_increment > 0:
                val = max(_snap(val, snap.scale_increment), snap.scale_increment)
            sc[k] = float(np.clip(val, 1e-4, 1e4))
        out[i] = (pos, q, sc)
    st.extra["factor"] = factor
    return out


def _snap_translation(st: DragState, delta: np.ndarray, inc: float) -> np.ndarray:
    """Global frame: snap the primary's absolute coordinates; local frame: snap the displacement."""
    h = st.handle
    if h.axis is not None:
        dims = [h.axis]
    elif h.plane is not None:
        dims = list(h.plane)
    else:
        dims = [0, 1, 2]
    p0 = st.start[st.primary][0]
    if st.global_frame:
        target = p0 + delta
        out = delta.copy()
        for k in dims:
            out[k] = _snap(target[k], inc) - p0[k]
        return out
    comps = st.axes.T @ delta
    for k in dims:
        comps[k] = _snap(comps[k], inc)
    return st.axes @ comps
