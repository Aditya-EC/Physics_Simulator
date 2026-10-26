"""Contact generation and contact constraint rows for all colliders of one step (array interface).

This is the NumPy reference of the native routine ``contact_rows`` (native/src/collision.cpp): the same
broadphase pair order, narrowphase and row layout, so both backends feed the solver identically.

Inputs (per shape, in world space): kind (0 sphere, 1 box, 2 capsule, 3 plane, 4 convex hull), the solver
body it belongs to, centre, rotation (row-major 3×3), radius, half extents, capsule half length, restitution, friction.
Per solver body: dynamic flag, centre of mass, velocity, angular velocity. ``excluded`` lists body pairs
that never collide (connected by a joint). Planes are static half-spaces (normal = local +Z). Hull shapes
name their hull in ``hull_id``; ``hulls`` holds the hulls in local coordinates, packed by ``hull.pack_hulls``.

Speculative contacts (continuous collision detection): ``sweep`` gives, per shape, how far its surface can
move during the step. When the combined sweep of a pair exceeds half the thickness of the thinner shape —
the only case in which it could pass through in one step — the pair is tested at the *start* of the step
(``center0``/``rot0``/``com0``: the solver runs after integration, and the end-of-step positions may already
be past a thin obstacle) with shapes inflated by that amount. A contact found this way has a real gap
g ≥ 0, and the step ends at x₀ + v·Δt: its row lets the bodies approach by at most g (rhs = −g/Δt), or,
when they would meet within the step faster than the restitution threshold, asks for the bounce velocity.
Slow pairs take the ordinary path, at the end-of-step positions, unchanged.

Output dict:
  pairs:    (P, 2) shape pairs that touch, in order
  contacts: shape_a, shape_b, point (C, 3), normal (C, 3) from a into b, depth, row (its normal row)
  rows:     ia, ib (solver bodies), J (R, 12), rhs, lo, hi, parent (row index or -1), mu, err,
            mode (1 = inequality with position correction, 2 = friction, no position correction)
"""
from __future__ import annotations

import math

import numpy as np

from .collision import WorldShape, collide
from .hull import unpack_hulls

KINDS = ("sphere", "box", "capsule", "plane", "hull")
COLLIDER_CODES = {"sphere": 0, "box": 1, "capsule": 2, "plane": 3, "convex": 4}
INF = 1e30


def _cross(a, b) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def friction_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = _cross(n, a)
    t1 /= math.sqrt(float(t1 @ t1))
    return t1, _cross(n, t1)


def bounding_radius(kind: int, radius: float, half, half_len: float) -> float:
    if kind == 0:
        return radius
    if kind in (1, 4):          # boxes, and hulls through their bounding box
        return math.sqrt(float(half @ half))
    if kind == 2:
        return radius + half_len
    return math.inf


def candidate_pairs(kind, center, radius, half, half_len, margin: float, sweep=None) -> list[tuple[int, int]]:
    """Pairs (i < j) whose bounding spheres, grown by their sweep, overlap (planes overlap everything)."""
    n = len(kind)
    rad = [bounding_radius(int(kind[i]), float(radius[i]), half[i], float(half_len[i]))
           + (float(sweep[i]) if sweep is not None else 0.0) for i in range(n)]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if kind[i] == 3 or kind[j] == 3:
                if not (kind[i] == 3 and kind[j] == 3):
                    out.append((i, j))
                continue
            d = center[j] - center[i]
            if math.sqrt(float(d @ d)) <= rad[i] + rad[j] + margin:
                out.append((i, j))
    return out


def thickness(kind: int, radius: float, half) -> float:
    """Smallest extent through the shape (an object thinner than its step can tunnel)."""
    if kind in (0, 2):
        return 2.0 * radius
    if kind in (1, 4):
        return 2.0 * float(min(half))
    return math.inf


def inflated(s: WorldShape, m: float) -> WorldShape:
    if s.kind in ("sphere", "capsule"):
        return WorldShape(s.kind, s.center, s.rot, s.radius + m, s.half, s.half_length)
    if s.kind in ("box", "hull"):   # a fast hull is tested speculatively through its bounding box
        return WorldShape("box", s.center, s.rot, s.radius, s.half + m, s.half_length)
    return s


def contact_rows(kind, body, center, rot, radius, half, half_len, restitution, friction, dynamic, excluded,
                 com, v, w, margin: float, restitution_threshold: float, sweep=None, dt: float = 0.0,
                 center0=None, rot0=None, com0=None, hull_id=None, hulls=None) -> dict:
    kind = np.asarray(kind, np.int64)
    body = np.asarray(body, np.int64)
    center = np.asarray(center, float).reshape(-1, 3)
    rot = np.asarray(rot, float).reshape(-1, 3, 3)
    half = np.asarray(half, float).reshape(-1, 3)
    com, v, w = (np.asarray(x, float).reshape(-1, 3) for x in (com, v, w))
    excl = {(int(a), int(b)) for a, b in np.asarray(excluded, np.int64).reshape(-1, 2)}
    excl |= {(b, a) for a, b in excl}
    sweep = np.zeros(len(kind)) if sweep is None else np.asarray(sweep, float)
    hull_list = unpack_hulls(hulls) if hulls is not None else []
    hid = np.full(len(kind), -1, np.int64) if hull_id is None else np.asarray(hull_id, np.int64)

    def hull_of(i):
        return hull_list[hid[i]] if kind[i] == 4 else None
    shapes = [WorldShape(KINDS[int(kind[i])], center[i], rot[i], float(radius[i]), half[i], float(half_len[i]),
                         hull_of(i)) for i in range(len(kind))]
    if center0 is None:
        shapes0, com0 = shapes, com
    else:
        center0 = np.asarray(center0, float).reshape(-1, 3)
        rot0 = np.asarray(rot0, float).reshape(-1, 3, 3)
        com0 = np.asarray(com0, float).reshape(-1, 3)
        shapes0 = [WorldShape(KINDS[int(kind[i])], center0[i], rot0[i], float(radius[i]), half[i],
                              float(half_len[i]), hull_of(i)) for i in range(len(kind))]
    rows = {k: [] for k in ("ia", "ib", "J", "rhs", "lo", "hi", "parent", "mu", "err", "mode")}
    con = {k: [] for k in ("shape_a", "shape_b", "point", "normal", "depth", "row")}
    pairs = []

    def add_row(a, b, j, rhs, lo, hi, parent, mu, err, mode) -> int:
        for k, x in zip(("ia", "ib", "J", "rhs", "lo", "hi", "parent", "mu", "err", "mode"),
                        (a, b, j, rhs, lo, hi, parent, mu, err, mode)):
            rows[k].append(x)
        return len(rows["ia"]) - 1

    for p, q in candidate_pairs(kind, center, radius, half, half_len, margin, sweep):
        a, b = int(body[p]), int(body[q])
        if a == b or not (dynamic[a] or dynamic[b]) or (a, b) in excl:
            continue
        m = float(sweep[p] + sweep[q])
        speculative = dt > 0 and m > 0.5 * min(thickness(int(kind[p]), float(radius[p]), half[p]),
                                                thickness(int(kind[q]), float(radius[q]), half[q]))
        if speculative:
            found = collide(inflated(shapes0[p], float(sweep[p])), inflated(shapes0[q], float(sweep[q])))
        else:
            found = collide(shapes[p], shapes[q])
        if not found:
            continue
        pairs.append((p, q))
        e = max(float(restitution[p]), float(restitution[q]))
        mu = math.sqrt(max(float(friction[p]), 0.0) * max(float(friction[q]), 0.0))
        for c in found:
            n = c.normal
            cc = com0 if speculative else com
            ra, rb = c.point - cc[a], c.point - cc[b]
            vn0 = float(((v[b] + _cross(w[b], rb)) - (v[a] + _cross(w[a], ra))) @ n)
            depth = c.depth - m if speculative else c.depth      # < 0: still separated by a gap
            gap = max(-depth, 0.0)
            if vn0 < -restitution_threshold and gap + vn0 * dt <= 0.0:
                target = -e * vn0                                  # they meet within this step: bounce
            else:
                target = -gap / dt if gap > 0.0 else 0.0           # may close the gap, not cross it
            # position error at the end of the step (speculative pairs were measured at its start)
            err = -depth + vn0 * dt if speculative else -depth
            k = add_row(a, b, np.concatenate([-n, -_cross(ra, n), n, _cross(rb, n)]), target, 0.0, INF, -1, 0.0,
                        err, 1)
            if mu > 0:
                for t in friction_basis(n):
                    add_row(a, b, np.concatenate([-t, -_cross(ra, t), t, _cross(rb, t)]), 0.0, -INF, INF, k, mu,
                            0.0, 2)
            for key, x in zip(("shape_a", "shape_b", "point", "normal", "depth", "row"),
                              (p, q, c.point, n, depth, k)):
                con[key].append(x)
    m, c = len(rows["ia"]), len(con["row"])
    return {"pairs": np.array(pairs, np.int64).reshape(-1, 2),
            "shape_a": np.array(con["shape_a"], np.int64), "shape_b": np.array(con["shape_b"], np.int64),
            "point": np.array(con["point"], float).reshape(c, 3),
            "normal": np.array(con["normal"], float).reshape(c, 3),
            "depth": np.array(con["depth"], float), "row": np.array(con["row"], np.int64),
            "ia": np.array(rows["ia"], np.int64), "ib": np.array(rows["ib"], np.int64),
            "J": np.array(rows["J"], float).reshape(m, 12), "rhs": np.array(rows["rhs"], float),
            "lo": np.array(rows["lo"], float), "hi": np.array(rows["hi"], float),
            "parent": np.array(rows["parent"], np.int64), "mu": np.array(rows["mu"], float),
            "err": np.array(rows["err"], float), "mode": np.array(rows["mode"], np.int64)}
