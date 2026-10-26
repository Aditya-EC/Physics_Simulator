"""Collision detection: world-space shapes, broadphase and narrowphase contact generation.

Every contact has a world point, a unit normal pointing from shape A into shape B
(B must move along +n to separate) and a penetration depth (> 0 when overlapping).

Pairs supported: sphere–{sphere, box, capsule, plane}, capsule–{capsule, box, plane},
box–{box, plane}. Box–box uses the separating-axis test (15 axes) with face
clipping for face contacts (up to 8 points) and closest points for edge–edge.
Capsule–box samples the capsule core (ends and the point closest to the box) and is
approximate for deep edge contacts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colliders import ColliderShape


def _cross(a, b) -> np.ndarray:
    """Cross product of two 3-vectors (np.cross has a large per-call overhead for single vectors)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


@dataclass
class WorldShape:
    kind: str
    center: np.ndarray            # world
    rot: np.ndarray               # (3, 3) world rotation of the collider
    radius: float = 0.0
    half: np.ndarray | None = None
    half_length: float = 0.0

    @staticmethod
    def place(shape: ColliderShape, origin: np.ndarray, rot: np.ndarray) -> "WorldShape":
        return WorldShape(shape.kind, origin + rot @ shape.center, rot, shape.radius, shape.half_extents,
                          shape.half_length)

    def segment(self) -> tuple[np.ndarray, np.ndarray]:
        axis = self.rot[:, 2] * self.half_length
        return self.center - axis, self.center + axis

    def bounding_radius(self) -> float:
        if self.kind == "sphere":
            return self.radius
        if self.kind == "box":
            return float(np.linalg.norm(self.half))
        if self.kind == "capsule":
            return self.radius + self.half_length
        return float("inf")


@dataclass
class Contact:
    point: np.ndarray
    normal: np.ndarray
    depth: float


def _contact(point, normal, depth) -> Contact:
    return Contact(np.asarray(point, float), np.asarray(normal, float), float(depth))


# ---- helpers ------------------------------------------------------------------------------------------------------

def closest_point_segment(p, a, b) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-18 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return a + t * ab


def closest_points_segments(p1, q1, p2, q2) -> tuple[np.ndarray, np.ndarray]:
    """Closest points between segments p1q1 and p2q2 (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a < 1e-18 and e < 1e-18:
        return p1, p2
    if a < 1e-18:
        s, t = 0.0, float(np.clip(f / e, 0, 1))
    else:
        c = float(d1 @ r)
        if e < 1e-18:
            s, t = float(np.clip(-c / a, 0, 1)), 0.0
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0, 1)) if denom > 1e-18 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, float(np.clip(-c / a, 0, 1))
            elif t > 1:
                t, s = 1.0, float(np.clip((b - c) / a, 0, 1))
    return p1 + d1 * s, p2 + d2 * t


def closest_point_box(p, box: WorldShape) -> tuple[np.ndarray, bool]:
    """Closest point on (or in) the box to p, and whether p is inside."""
    local = box.rot.T @ (p - box.center)
    clamped = np.clip(local, -box.half, box.half)
    inside = bool(np.all(np.abs(local) <= box.half))
    return box.center + box.rot @ clamped, inside


def _box_vertices(box: WorldShape) -> np.ndarray:
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return box.center + (signs * box.half) @ box.rot.T


# ---- narrowphase -----------------------------------------------------------------------------------------------------

def sphere_sphere(c1, r1, c2, r2) -> list[Contact]:
    d = c2 - c1
    dist = float(np.linalg.norm(d))
    if dist >= r1 + r2:
        return []
    n = d / dist if dist > 1e-12 else np.array([0.0, 0.0, 1.0])
    return [_contact(c1 + n * (r1 - (r1 + r2 - dist) / 2), n, r1 + r2 - dist)]


def sphere_plane(c, r, plane: WorldShape, sphere_is_a: bool) -> list[Contact]:
    n = plane.rot[:, 2]
    dist = float((c - plane.center) @ n)
    if dist >= r:
        return []
    point = c - n * dist          # the sphere centre projected onto the plane
    depth = r - dist
    # plane normal points out of the solid half-space, i.e. from the plane into the sphere
    return [_contact(point, -n if sphere_is_a else n, depth)]


def sphere_box(c, r, box: WorldShape) -> list[Contact]:
    """Normal from the box into the sphere."""
    q, inside = closest_point_box(c, box)
    if not inside:
        d = c - q
        dist = float(np.linalg.norm(d))
        if dist >= r:
            return []
        return [_contact(q, d / dist, r - dist)]
    local = box.rot.T @ (c - box.center)
    gaps = box.half - np.abs(local)
    k = int(np.argmin(gaps))
    n_local = np.zeros(3)
    n_local[k] = 1.0 if local[k] >= 0 else -1.0
    n = box.rot @ n_local
    return [_contact(c - n * gaps[k], n, r + gaps[k])]


def box_plane(box: WorldShape, plane: WorldShape, box_is_a: bool) -> list[Contact]:
    n = plane.rot[:, 2]
    verts = _box_vertices(box)
    dist = (verts - plane.center) @ n
    out = [_contact(v - n * d, -n if box_is_a else n, -d) for v, d in zip(verts, dist) if d < 0]
    out.sort(key=lambda c: -c.depth)
    return out[:4]


def capsule_plane(cap: WorldShape, plane: WorldShape, cap_is_a: bool) -> list[Contact]:
    out = []
    for p in cap.segment():
        out += sphere_plane(p, cap.radius, plane, cap_is_a)
    return out


def capsule_sphere(cap: WorldShape, c, r) -> list[Contact]:
    """Normal from the capsule into the sphere."""
    a, b = cap.segment()
    return sphere_sphere(closest_point_segment(c, a, b), cap.radius, c, r)


def capsule_capsule(c1: WorldShape, c2: WorldShape) -> list[Contact]:
    a1, b1 = c1.segment()
    a2, b2 = c2.segment()
    d1, d2 = b1 - a1, b2 - a2
    out = []
    parallel = np.linalg.norm(_cross(d1, d2)) < 1e-6 * max(1e-12, np.linalg.norm(d1) * np.linalg.norm(d2))
    if parallel and np.linalg.norm(d1) > 1e-9:
        # overlapping parallel cores: contacts at both ends of the overlap
        for p in (a2, b2):
            out += sphere_sphere(closest_point_segment(p, a1, b1), c1.radius, p, c2.radius)
        for p in (a1, b1):
            out += [Contact(c.point, -c.normal, c.depth)
                    for c in sphere_sphere(closest_point_segment(p, a2, b2), c2.radius, p, c1.radius)]
        return out[:2] if out else []
    p, q = closest_points_segments(a1, b1, a2, b2)
    return sphere_sphere(p, c1.radius, q, c2.radius)


def capsule_box(cap: WorldShape, box: WorldShape) -> list[Contact]:
    """Normal from the box into the capsule (approximate: samples the capsule core)."""
    a, b = cap.segment()
    samples = [a, b, closest_point_segment(box.center, a, b)]
    out: list[Contact] = []
    for p in samples:
        for c in sphere_box(p, cap.radius, box):
            if all(np.linalg.norm(c.point - o.point) > 1e-6 for o in out):
                out.append(c)
    return out


def _clip(poly: list[np.ndarray], n: np.ndarray, d: float) -> list[np.ndarray]:
    """Keep the part of a polygon with n·x ≤ d (Sutherland–Hodgman)."""
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        dp, dq = float(n @ p) - d, float(n @ q) - d
        if dp <= 0:
            out.append(p)
        if dp * dq < 0:
            out.append(p + (q - p) * (dp / (dp - dq)))
    return out


def _face(box: WorldShape, axis: int, sign: float) -> list[np.ndarray]:
    i, j = [k for k in range(3) if k != axis]
    c = box.center + box.rot[:, axis] * box.half[axis] * sign
    u, v = box.rot[:, i] * box.half[i], box.rot[:, j] * box.half[j]
    return [c + u + v, c - u + v, c - u - v, c + u - v]


def box_box(b1: WorldShape, b2: WorldShape) -> list[Contact]:
    """Normal from box 1 into box 2."""
    d = b2.center - b1.center
    axes, kinds = [], []
    for k in range(3):
        axes.append(b1.rot[:, k])
        kinds.append(("face1", k))
    for k in range(3):
        axes.append(b2.rot[:, k])
        kinds.append(("face2", k))
    for i in range(3):
        for j in range(3):
            ax = _cross(b1.rot[:, i], b2.rot[:, j])
            n = np.linalg.norm(ax)
            if n > 1e-6:
                axes.append(ax / n)
                kinds.append(("edge", (i, j)))
    best, best_axis, best_kind = np.inf, None, None
    for idx, (ax, kind) in enumerate(zip(axes, kinds)):
        r1 = float(np.sum(b1.half * np.abs(b1.rot.T @ ax)))
        r2 = float(np.sum(b2.half * np.abs(b2.rot.T @ ax)))
        overlap = r1 + r2 - abs(float(d @ ax))
        if overlap <= 0:
            return []
        # prefer face axes over nearly equal edge axes (more stable manifolds)
        bias = 1.0 if kind[0] != "edge" else 1.05
        if overlap * bias < best:
            best, best_axis, best_kind = overlap * bias, ax if d @ ax >= 0 else -ax, kind
            best_overlap = overlap
    n = best_axis
    if best_kind[0] == "edge":
        i, j = best_kind[1]
        # the edge of each box most aligned with the separating direction
        def support_edge(box, k, direction):
            s = np.sign(box.rot.T @ direction)
            s[s == 0] = 1.0
            s[k] = 0.0
            mid = box.center + box.rot @ (s * box.half)
            e = box.rot[:, k] * box.half[k]
            return mid - e, mid + e
        p1, q1 = support_edge(b1, i, n)
        p2, q2 = support_edge(b2, j, -n)
        c1, c2 = closest_points_segments(p1, q1, p2, q2)
        return [_contact((c1 + c2) / 2, n, best_overlap)]
    # face contact: reference face on the box owning the axis, incident face on the other
    if best_kind[0] == "face1":
        ref, inc, ref_axis, flip = b1, b2, best_kind[1], False
        ref_n = n
    else:
        ref, inc, ref_axis, flip = b2, b1, best_kind[1], True
        ref_n = -n
    sign = 1.0 if ref.rot[:, ref_axis] @ ref_n >= 0 else -1.0
    ref_face_n = ref.rot[:, ref_axis] * sign
    # incident face: the face of `inc` most anti-parallel to the reference normal
    dots = inc.rot.T @ ref_face_n
    inc_axis = int(np.argmax(np.abs(dots)))
    inc_poly = _face(inc, inc_axis, -np.sign(dots[inc_axis]) or -1.0)
    for k in [k for k in range(3) if k != ref_axis]:
        side = ref.rot[:, k]
        for s in (1.0, -1.0):
            nn = side * s
            inc_poly = _clip(inc_poly, nn, float(nn @ ref.center) + ref.half[k])
            if not inc_poly:
                return []
    plane_d = float(ref_face_n @ ref.center) + ref.half[ref_axis]
    out = []
    for p in inc_poly:
        depth = plane_d - float(ref_face_n @ p)
        if depth > 0:
            point = p + ref_face_n * depth / 2
            out.append(_contact(point, -ref_face_n if flip else ref_face_n, depth))
    out.sort(key=lambda c: -c.depth)
    return out[:8]


def collide(a: WorldShape, b: WorldShape) -> list[Contact]:
    """Contacts between two shapes with normals from a into b."""
    ka, kb = a.kind, b.kind
    if ka == "plane" and kb == "plane":
        return []
    if kb == "plane":
        if ka == "sphere":
            return sphere_plane(a.center, a.radius, b, True)
        if ka == "box":
            return box_plane(a, b, True)
        return capsule_plane(a, b, True)
    if ka == "plane":
        return [Contact(c.point, -c.normal, c.depth) for c in collide(b, a)]
    if ka == "sphere" and kb == "sphere":
        return sphere_sphere(a.center, a.radius, b.center, b.radius)
    if ka == "box" and kb == "sphere":
        return sphere_box(b.center, b.radius, a)
    if ka == "capsule" and kb == "sphere":
        return capsule_sphere(a, b.center, b.radius)
    if ka == "capsule" and kb == "capsule":
        return capsule_capsule(a, b)
    if ka == "box" and kb == "capsule":
        return capsule_box(b, a)
    if ka == "box" and kb == "box":
        return box_box(a, b)
    # remaining orders: swap
    return [Contact(c.point, -c.normal, c.depth) for c in collide(b, a)]


def broadphase(shapes: list[WorldShape | None], margin: float = 0.0) -> list[tuple[int, int]]:
    """Candidate pairs from bounding spheres (planes overlap everything)."""
    idx = [i for i, s in enumerate(shapes) if s is not None]
    pairs = []
    for k, i in enumerate(idx):
        si = shapes[i]
        for j in idx[k + 1:]:
            sj = shapes[j]
            if si.kind == "plane" or sj.kind == "plane":
                pairs.append((i, j))
                continue
            if np.linalg.norm(si.center - sj.center) <= si.bounding_radius() + sj.bounding_radius() + margin:
                pairs.append((i, j))
    return pairs
