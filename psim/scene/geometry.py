"""Triangle-mesh generators for the built-in object library.

All meshes are expressed in the object's local frame (Z up), centred on the
bounding-box centre unless noted, with outward-facing counter-clockwise
triangles and per-vertex normals. Closed meshes are flagged `solid=True` and
are valid input for volume / inertia integration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import numerics


@dataclass(eq=False)
class Mesh:
    vertices: np.ndarray            # (N, 3) float32
    normals: np.ndarray             # (N, 3) float32
    indices: np.ndarray             # (M, 3) uint32
    colors: np.ndarray | None = None  # (N, 3) float32 per-vertex colours (optional)
    solid: bool = True              # closed, consistently oriented surface
    _edges: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32).reshape(-1, 3)
        self.normals = np.ascontiguousarray(self.normals, dtype=np.float32).reshape(-1, 3)
        self.indices = np.ascontiguousarray(self.indices, dtype=np.uint32).reshape(-1, 3)
        if self.colors is not None:
            self.colors = np.ascontiguousarray(self.colors, dtype=np.float32).reshape(-1, 3)

    @property
    def triangle_count(self) -> int:
        return int(len(self.indices))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.vertices):
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0).astype(float), self.vertices.max(axis=0).astype(float)

    def feature_edges(self, angle_deg: float = 30.0) -> np.ndarray:
        """Line-segment endpoints (2K, 3) of silhouette-independent feature edges."""
        if self._edges is None:
            span = float(np.ptp(self.vertices, axis=0).max()) if len(self.vertices) else 1.0
            self._edges = numerics.feature_edges(self.vertices, self.indices, angle_deg, max(span, 1e-6) * 1e-5)
        return self._edges

    def transformed(self, matrix: np.ndarray) -> "Mesh":
        m = np.asarray(matrix, dtype=np.float64)
        v = numerics.transform_points(m, self.vertices.astype(np.float64))
        nm = np.linalg.inv(m[:3, :3]).T
        n = self.normals.astype(np.float64) @ nm.T
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln == 0] = 1
        flip = np.linalg.det(m[:3, :3]) < 0
        idx = self.indices[:, ::-1] if flip else self.indices
        return Mesh(v, n / ln, idx, self.colors, self.solid)

    def scaled(self, scale) -> "Mesh":
        return self.transformed(np.diag([*np.asarray(scale, dtype=float), 1.0]))

    def with_color(self, rgb) -> "Mesh":
        return Mesh(self.vertices, self.normals, self.indices, np.tile(np.asarray(rgb, np.float32), (len(self.vertices), 1)), self.solid)


def merge(meshes: list[Mesh], solid: bool | None = None) -> Mesh:
    verts, norms, idx, cols = [], [], [], []
    offset = 0
    any_colors = any(m.colors is not None for m in meshes)
    for m in meshes:
        verts.append(m.vertices)
        norms.append(m.normals)
        idx.append(m.indices + offset)
        if any_colors:
            cols.append(m.colors if m.colors is not None else np.full((len(m.vertices), 3), 0.8, np.float32))
        offset += len(m.vertices)
    return Mesh(np.vstack(verts), np.vstack(norms), np.vstack(idx), np.vstack(cols) if any_colors else None,
                all(m.solid for m in meshes) if solid is None else solid)


# --------------------------------------------------------------------------
# Primitive builders
# --------------------------------------------------------------------------

def _segments(n: int, lo: int = 3, hi: int = 256) -> int:
    return int(min(max(int(n), lo), hi))


def box(lx: float, ly: float, lz: float) -> Mesh:
    hx, hy, hz = lx / 2, ly / 2, lz / 2
    faces = [  # (normal, 4 corners CCW seen from outside)
        ((1, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
        ((-1, 0, 0), [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)]),
        ((0, 1, 0), [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)]),
    ]
    v, n, f = [], [], []
    for normal, quad in faces:
        b = len(v)
        v.extend(quad)
        n.extend([normal] * 4)
        f.extend([(b, b + 1, b + 2), (b, b + 2, b + 3)])
    return Mesh(np.array(v), np.array(n), np.array(f))


def lathe(sections: list[tuple[np.ndarray, np.ndarray]], segments: int = 48, solid: bool = True) -> Mesh:
    """Surface of revolution about Z.

    Each section is (points (K,2) of (r, z), normals (K,2) of (n_r, n_z)), traversed
    counter-clockwise in the meridian half-plane (bottom-centre -> outer -> top-centre)
    so that triangles face outward. Sections are separate smooth strips, which lets
    hard creases (e.g. cylinder rims) be expressed by starting a new section.
    """
    seg = _segments(segments)
    theta = np.linspace(0.0, 2 * np.pi, seg + 1)
    c, s = np.cos(theta), np.sin(theta)
    meshes = []
    for pts, nrm in sections:
        pts = np.asarray(pts, float)
        nrm = np.asarray(nrm, float)
        k = len(pts)
        r, z = pts[:, 0][:, None], pts[:, 1][:, None]
        verts = np.stack([r * c, r * s, np.broadcast_to(z, (k, seg + 1))], axis=-1).reshape(-1, 3)
        nr, nz = nrm[:, 0][:, None], nrm[:, 1][:, None]
        norms = np.stack([nr * c, nr * s, np.broadcast_to(nz, (k, seg + 1))], axis=-1).reshape(-1, 3)
        i, j = np.meshgrid(np.arange(k - 1), np.arange(seg), indexing="ij")
        a = (i * (seg + 1) + j).ravel()
        b = a + 1
        cc = a + (seg + 1)
        d = cc + 1
        tris = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, cc], 1)])
        meshes.append(Mesh(verts, norms, tris, solid=solid))
    return merge(meshes, solid=solid)


def _flat_section(p0, p1, n=None):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if n is None:
        d = p1 - p0
        n = np.array([d[1], -d[0]]) / max(np.hypot(*d), 1e-12)
    return np.array([p0, p1]), np.array([n, n])


def frustum(r_bottom: float, r_top: float, height: float, segments: int = 48) -> Mesh:
    h = height / 2
    secs = []
    if r_bottom > 0:
        secs.append(_flat_section((0, -h), (r_bottom, -h)))
    secs.append(_flat_section((r_bottom, -h), (r_top, h)))
    if r_top > 0:
        secs.append(_flat_section((r_top, h), (0, h)))
    return lathe(secs, segments)


def cylinder(radius: float, height: float, segments: int = 48) -> Mesh:
    return frustum(radius, radius, height, segments)


def cone(radius: float, height: float, segments: int = 48) -> Mesh:
    return frustum(radius, 0.0, height, segments)


def _arc(radius: float, phi0: float, phi1: float, steps: int, z0: float = 0.0, r0: float = 0.0):
    phi = np.linspace(phi0, phi1, steps)
    pts = np.stack([r0 + radius * np.cos(phi), z0 + radius * np.sin(phi)], 1)
    nrm = np.stack([np.cos(phi), np.sin(phi)], 1)
    pts[np.abs(pts[:, 0]) < 1e-12, 0] = 0.0
    return pts, nrm


def sphere(radius: float, segments: int = 48, rings: int = 24) -> Mesh:
    return lathe([_arc(radius, -np.pi / 2, np.pi / 2, _segments(rings, 4) + 1)], segments)


def ellipsoid(rx: float, ry: float, rz: float, segments: int = 48, rings: int = 24) -> Mesh:
    unit = sphere(1.0, segments, rings)
    v = unit.vertices * np.array([rx, ry, rz], np.float32)
    n = unit.vertices / np.array([rx, ry, rz], np.float32)  # gradient of the implicit surface
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return Mesh(v, n / ln, unit.indices)


def hemisphere(radius: float, segments: int = 48, rings: int = 12) -> Mesh:
    """Dome with a flat base; centred on its bounding box (base at z = -r/2)."""
    z0 = -radius / 2
    base = _flat_section((0, z0), (radius, z0))
    dome = _arc(radius, 0.0, np.pi / 2, _segments(rings, 3) + 1, z0=z0)
    return lathe([base, dome], segments)


def capsule(radius: float, length: float, segments: int = 48, rings: int = 12) -> Mesh:
    """Cylinder of `length` (between hemisphere centres) capped by two hemispheres."""
    h = length / 2
    n = _segments(rings, 3) + 1
    bottom = _arc(radius, -np.pi / 2, 0.0, n, z0=-h)
    top = _arc(radius, 0.0, np.pi / 2, n, z0=h)
    pts = np.vstack([bottom[0], top[0]])
    nrm = np.vstack([bottom[1], top[1]])
    return lathe([(pts, nrm)], segments)


def torus(major: float, minor: float, segments: int = 64, tube_segments: int = 24) -> Mesh:
    phi = np.linspace(-np.pi, np.pi, _segments(tube_segments) + 1)
    pts = np.stack([major + minor * np.cos(phi), minor * np.sin(phi)], 1)
    nrm = np.stack([np.cos(phi), np.sin(phi)], 1)
    return lathe([(pts, nrm)], segments)


def disk(radius: float, segments: int = 48) -> Mesh:
    return lathe([_flat_section((radius, 0), (0, 0))], segments, solid=False)


def plane(lx: float, ly: float, divisions: int = 1) -> Mesh:
    d = _segments(divisions, 1, 64)
    xs = np.linspace(-lx / 2, lx / 2, d + 1)
    ys = np.linspace(-ly / 2, ly / 2, d + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    v = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    n = np.tile([0.0, 0.0, 1.0], (len(v), 1))
    i, j = np.meshgrid(np.arange(d), np.arange(d), indexing="ij")
    a = (i * (d + 1) + j).ravel()
    b = a + (d + 1)
    tris = np.concatenate([np.stack([a, b, b + 1], 1), np.stack([a, b + 1, a + 1], 1)])
    return Mesh(v, n, tris, solid=False)


# ---- polygon extrusion -----------------------------------------------------

def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def triangulate_polygon(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple CCW polygon."""
    n = len(poly)
    idx = list(range(n))
    tris: list[tuple[int, int, int]] = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= -1e-12 and cross(b, c, p) >= -1e-12 and cross(c, a, p) >= -1e-12

    guard = 0
    while len(idx) > 3 and guard < 10 * n * n:
        guard += 1
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 1e-14:
                continue  # reflex or degenerate
            if any(inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break  # numerical failure; fall back to a fan for the rest
    if len(idx) >= 3:
        tris.extend((idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1))
    return tris


def extrude(poly, height: float, smooth_sides: bool = False) -> Mesh:
    """Extrude a simple polygon in the XY plane along Z, centred at z = 0."""
    poly = np.asarray(poly, float)
    if _polygon_area(poly) < 0:
        poly = poly[::-1]
    n = len(poly)
    h = height / 2
    tris = triangulate_polygon(poly)
    v, nn, f = [], [], []
    # caps
    for z, normal, flip in ((h, (0, 0, 1), False), (-h, (0, 0, -1), True)):
        b = len(v)
        v.extend((p[0], p[1], z) for p in poly)
        nn.extend([normal] * n)
        f.extend(((b + t[0], b + t[2], b + t[1]) if flip else (b + t[0], b + t[1], b + t[2])) for t in tris)
    # sides (one flat quad per edge)
    for k in range(n):
        p0, p1 = poly[k], poly[(k + 1) % n]
        d = p1 - p0
        normal = np.array([d[1], -d[0], 0.0]) / max(np.hypot(*d), 1e-12)
        b = len(v)
        v.extend([(p0[0], p0[1], -h), (p1[0], p1[1], -h), (p1[0], p1[1], h), (p0[0], p0[1], h)])
        nn.extend([normal] * 4)
        f.extend([(b, b + 1, b + 2), (b, b + 2, b + 3)])
    mesh = Mesh(np.array(v), np.array(nn), np.array(f))
    if smooth_sides:
        mesh.normals = numerics.compute_vertex_normals(mesh.vertices, mesh.indices)
    return mesh


def regular_polygon(sides: int, circumradius: float) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, _segments(sides, 3, 128), endpoint=False) + np.pi / 2
    return np.stack([circumradius * np.cos(a), circumradius * np.sin(a)], 1)


def prism(sides: int, circumradius: float, height: float) -> Mesh:
    return extrude(regular_polygon(sides, circumradius), height)


def pyramid(lx: float, ly: float, height: float) -> Mesh:
    """Rectangular-base pyramid centred on its bounding box."""
    h = height / 2
    base = [(-lx / 2, -ly / 2, -h), (lx / 2, -ly / 2, -h), (lx / 2, ly / 2, -h), (-lx / 2, ly / 2, -h)]
    apex = (0.0, 0.0, h)
    v, n, f = [], [], []
    b = 0
    v.extend(base[::-1])
    n.extend([(0, 0, -1)] * 4)
    f.extend([(0, 1, 2), (0, 2, 3)])
    for k in range(4):
        p0, p1 = np.array(base[k]), np.array(base[(k + 1) % 4])
        normal = np.cross(p1 - p0, np.array(apex) - p0)
        normal /= np.linalg.norm(normal)
        b = len(v)
        v.extend([p0, p1, apex])
        n.extend([normal] * 3)
        f.append((b, b + 1, b + 2))
    return Mesh(np.array(v), np.array(n), np.array(f))


def gear_profile(teeth: int, outer_radius: float, root_radius: float, samples_per_tooth: int = 8) -> np.ndarray:
    teeth = _segments(teeth, 4, 200)
    pts = []
    for t in range(teeth):
        a0 = 2 * np.pi * t / teeth
        w = 2 * np.pi / teeth
        # root land, flank up, tip land, flank down (trapezoidal teeth)
        for frac, r in ((0.0, root_radius), (0.30, root_radius), (0.42, outer_radius), (0.70, outer_radius),
                        (0.82, root_radius)):
            pts.append((r * np.cos(a0 + frac * w), r * np.sin(a0 + frac * w)))
    return np.array(pts)


def i_beam_profile(width: float, height: float, web: float, flange: float) -> np.ndarray:
    w, h, tw, tf = width / 2, height / 2, web / 2, flange
    return np.array([(-w, -h), (w, -h), (w, -h + tf), (tw, -h + tf), (tw, h - tf), (w, h - tf), (w, h),
                     (-w, h), (-w, h - tf), (-tw, h - tf), (-tw, -h + tf), (-w, -h + tf)])


# ---- swept tubes -------------------------------------------------------------

def tube(path: np.ndarray, radius: float, sides: int = 16, caps: bool = True) -> Mesh:
    """Sweep a circle along a polyline using parallel-transport frames."""
    path = np.asarray(path, float)
    k = len(path)
    sides = _segments(sides, 3, 64)
    tangents = np.gradient(path, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(tangents[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    nrm = np.cross(tangents[0], ref)
    nrm /= np.linalg.norm(nrm)
    frames = []
    for i in range(k):
        if i > 0:
            t0, t1 = tangents[i - 1], tangents[i]
            axis = np.cross(t0, t1)
            s = np.linalg.norm(axis)
            if s > 1e-12:
                ang = math.atan2(s, float(np.dot(t0, t1)))
                q = numerics.quat_from_axis_angle(axis / s, ang)
                nrm = numerics.quat_rotate(q, nrm)
        nrm = nrm - tangents[i] * np.dot(nrm, tangents[i])
        nrm /= max(np.linalg.norm(nrm), 1e-12)
        frames.append((nrm, np.cross(tangents[i], nrm)))
    ang = np.linspace(0, 2 * np.pi, sides + 1)
    v, n = [], []
    for i in range(k):
        a, b = frames[i]
        dirs = np.outer(np.cos(ang), a) + np.outer(np.sin(ang), b)
        v.append(path[i] + radius * dirs)
        n.append(dirs)
    v = np.vstack(v)
    n = np.vstack(n)
    i, j = np.meshgrid(np.arange(k - 1), np.arange(sides), indexing="ij")
    a = (i * (sides + 1) + j).ravel()
    tris = np.concatenate([np.stack([a, a + 1, a + sides + 2], 1), np.stack([a, a + sides + 2, a + sides + 1], 1)])
    mesh = Mesh(v, n, tris, solid=caps)
    if not caps:
        return mesh
    parts = [mesh]
    for end, sign in ((0, -1.0), (k - 1, 1.0)):
        t = tangents[end] * sign
        a, b = frames[end]
        ring = path[end] + radius * (np.outer(np.cos(ang[:-1]), a) + np.outer(np.sin(ang[:-1]), b))
        cv = np.vstack([path[end], ring])
        cn = np.tile(t, (len(cv), 1))
        ct = [(0, 1 + (m + 1) % sides, 1 + m) for m in range(sides)]
        cap = Mesh(cv, cn, np.array(ct))
        # orient the fan so it faces along t
        p0, p1, p2 = cap.vertices[cap.indices[0]]
        if np.dot(np.cross(p1 - p0, p2 - p0), t) < 0:
            cap.indices = cap.indices[:, ::-1].copy()
        parts.append(cap)
    return merge(parts, solid=True)


def helix_path(coil_radius: float, length: float, turns: float, samples_per_turn: int = 32) -> np.ndarray:
    m = max(int(turns * samples_per_turn), 8)
    t = np.linspace(0, 1, m + 1)
    a = 2 * np.pi * turns * t
    return np.stack([coil_radius * np.cos(a), coil_radius * np.sin(a), length * (t - 0.5)], 1)


def spring(coil_radius: float, wire_radius: float, length: float, turns: float) -> Mesh:
    return tube(helix_path(coil_radius, length, turns), wire_radius, sides=12)


def rope(length: float, radius: float, sag: float, samples: int = 40) -> Mesh:
    """Cable between (-L/2, 0, 0) and (L/2, 0, 0) with a parabolic sag along -Z."""
    x = np.linspace(-length / 2, length / 2, samples)
    z = -sag * (1 - (2 * x / length) ** 2) if length > 0 else np.zeros_like(x)
    z += sag / 2  # centre the bounding box
    return tube(np.stack([x, np.zeros_like(x), z], 1), radius, sides=12)


def arrow(length: float, shaft_radius: float, head_radius: float, head_length: float, segments: int = 24) -> Mesh:
    """Arrow along +X from the origin (local origin at the tail)."""
    head_length = min(head_length, length)
    shaft_len = max(length - head_length, 1e-6)
    secs = [
        _flat_section((0, 0), (shaft_radius, 0)),
        _flat_section((shaft_radius, 0), (shaft_radius, shaft_len)),
        _flat_section((shaft_radius, shaft_len), (head_radius, shaft_len)),
        _flat_section((head_radius, shaft_len), (0, length)),
    ]
    m = lathe(secs, segments)
    # lathe builds along +Z; rotate +Z -> +X
    rot = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], float)
    return m.transformed(rot)


AXIS_COLORS = ((0.90, 0.30, 0.28), (0.40, 0.78, 0.30), (0.30, 0.52, 0.95))


def axis_triad(length: float, radius: float) -> Mesh:
    base = arrow(length, radius, radius * 2.6, length * 0.22)
    rots = [np.eye(4),
            np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float),
            np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)]
    parts = [base.transformed(r).with_color(c) for r, c in zip(rots, AXIS_COLORS)]
    parts.append(sphere(radius * 2.2, 20, 10).with_color((0.85, 0.85, 0.85)))
    return merge(parts, solid=False)


def coordinate_marker(size: float) -> Mesh:
    r = size * 0.06
    bars = [box(size, r * 2, r * 2), box(r * 2, size, r * 2), box(r * 2, r * 2, size)]
    parts = [b.with_color(c) for b, c in zip(bars, AXIS_COLORS)]
    parts.append(sphere(size * 0.16, 20, 10).with_color((0.95, 0.75, 0.25)))
    return merge(parts, solid=False)


def line_segment(length: float, radius: float) -> Mesh:
    """Thin cylinder from (-L/2,0,0) to (L/2,0,0) with spherical end markers."""
    path = np.array([[-length / 2, 0, 0], [length / 2, 0, 0]], float)
    body = tube(path, radius, sides=12)
    ends = [sphere(radius * 2.2, 16, 8).transformed(_translation(p)) for p in path]
    return merge([body, *ends], solid=False)


def _translation(t) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def translated(mesh: Mesh, t) -> Mesh:
    return mesh.transformed(_translation(t))


# ---- OBJ import ----------------------------------------------------------------

class MeshImportError(ValueError):
    pass


def load_obj(text: str, max_vertices: int = 2_000_000) -> Mesh:
    """Parse a Wavefront OBJ (positions + polygon faces; normals recomputed).

    Polygons are fan-triangulated. The result is recentred on its bounding box.
    """
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            if parts[0] == "v":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(verts) > max_vertices:
                    raise MeshImportError("Mesh exceeds the vertex limit")
            elif parts[0] == "f":
                ids = []
                for tok in parts[1:]:
                    i = int(tok.split("/")[0])
                    ids.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(ids) - 1):
                    faces.append((ids[0], ids[k], ids[k + 1]))
        except (IndexError, ValueError) as exc:
            raise MeshImportError(f"Line {lineno}: {exc}") from None
    if not verts or not faces:
        raise MeshImportError("No geometry found in OBJ file")
    v = np.array(verts, float)
    f = np.array(faces, np.int64)
    if f.min() < 0 or f.max() >= len(v):
        raise MeshImportError("Face references a missing vertex")
    lo, hi = v.min(axis=0), v.max(axis=0)
    v -= (lo + hi) / 2
    return mesh_from_arrays(v, f)


def mesh_from_arrays(vertices, indices, flat: bool = True) -> Mesh:
    """Build a mesh from raw positions/triangles. `flat` splits vertices for faceted shading."""
    v = np.asarray(vertices, np.float32).reshape(-1, 3)
    f = np.asarray(indices, np.uint32).reshape(-1, 3)
    if flat:
        v = v[f.reshape(-1)]
        f = np.arange(len(v), dtype=np.uint32).reshape(-1, 3)
    n = numerics.compute_vertex_normals(v, f)
    m = Mesh(v, n, f)
    mp = numerics.mass_properties_solid(m.vertices, m.indices, 1.0)
    m.solid = _is_closed(v, f) and mp["measure"] > 0
    return m


def _is_closed(v: np.ndarray, f: np.ndarray) -> bool:
    keys = np.round(v.astype(np.float64) / max(float(np.ptp(v, axis=0).max()), 1e-9) * 1e6).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    c = inv.reshape(-1)[f]
    c = c[(c[:, 0] != c[:, 1]) & (c[:, 1] != c[:, 2]) & (c[:, 0] != c[:, 2])]  # drop collapsed pole triangles
    edges = np.sort(np.concatenate([c[:, [0, 1]], c[:, [1, 2]], c[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(len(counts)) and bool(np.all(counts == 2))


def mesh_to_obj(mesh: Mesh) -> str:
    lines = ["# Exported by PSim Studio"]
    lines += [f"v {x:.7g} {y:.7g} {z:.7g}" for x, y, z in mesh.vertices]
    lines += [f"vn {x:.5g} {y:.5g} {z:.5g}" for x, y, z in mesh.normals]
    lines += [f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}" for a, b, c in mesh.indices]
    return "\n".join(lines) + "\n"
