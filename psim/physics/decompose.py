"""Approximate convex decomposition of a closed triangle mesh (compound colliders for concave objects).

The solid is split recursively by planes into pieces whose convex hulls stay close to the piece itself:
at each step the piece whose hull adds the most volume ("concavity" = hull volume − piece volume) is cut by
the candidate plane that minimises the concavity of the two halves. It stops when the remaining concavity is a
small fraction of the solid's volume, or at a maximum number of parts. When no cut reduces a piece's concavity
noticeably (a ring needs two cuts before its pieces become convex), the cuts are ranked by what one further cut
of their halves achieves. Costs within 1e-9 of the volume count as equal and the first candidate wins, so
symmetric shapes are cut the same way by both backends.

This is the NumPy reference of the native routine ``convex_decomposition`` (native/src/decompose.cpp).

Cuts are exact. A piece is a closed triangle chain: the clipped surface plus a cap on each cut plane. A cap is
fanned from a point of its boundary over the oriented cut segments, which gives the correct (signed) volume
even for caps with several loops or holes, and every vertex lies in the piece, so the hull of a piece's
vertices is exactly the convex hull of the piece.
"""
from __future__ import annotations

import numpy as np

from .hull import HULL_EPS, HullError, _dedupe, _fibonacci_directions, _triangle_hull

MAX_PARTS = 16
TOLERANCE = 0.03                    # stop when total concavity ≤ this fraction of the solid's volume
POSITIONS = (0.25, 0.5, 0.75)       # candidate cut positions along each axis of a piece's bounds
LOOKAHEAD_POSITIONS = (0.5,)
MIN_SLICE = 1e-4                    # cuts leave no piece thinner than this fraction of the piece's extent
MAX_VERTEX_CUTS = 8                 # pieces with at most this many distinct inner coordinates along an axis are
                                    # also cut beside those coordinates (exact parts for blocky polyhedra)
IMPROVEMENT = 0.9                   # a cut must remove 10 % of a piece's concavity, or cuts are ranked by lookahead


def chain_volume(tris: np.ndarray) -> float:
    """Signed volume enclosed by a closed (signed) triangle chain (T, 3, 3)."""
    if len(tris) == 0:
        return 0.0
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)


def _lerp(a, b, sa, sb):
    t = (sa / (sa - sb))[:, None]
    return a + t * (b - a)


def clip(tris: np.ndarray, n, d: float) -> np.ndarray:
    """The part of a closed chain with n·x ≤ d, closed again by a cap on the plane."""
    n = np.asarray(n, float)
    if len(tris) == 0:
        return tris
    s = tris @ n - d                          # (T, 3) signed distances
    inside = s <= 0.0
    count = inside.sum(axis=1)
    keep = [tris[count == 3]]
    seg_a, seg_b = [], []                     # cap boundary segments a → b (cap orientation)
    # one vertex inside: (Pi, Pij, Pik); the cut edge Pij → Pik, the cap gets Pik → Pij
    one = np.flatnonzero(count == 1)
    if len(one):
        i = np.argmax(inside[one], axis=1)
        order = (i[:, None] + np.arange(3)) % 3
        t = tris[one[:, None], order]
        st = s[one[:, None], order]
        pij = _lerp(t[:, 0], t[:, 1], st[:, 0], st[:, 1])
        pik = _lerp(t[:, 0], t[:, 2], st[:, 0], st[:, 2])
        keep.append(np.stack([t[:, 0], pij, pik], 1))
        seg_a.append(pik)
        seg_b.append(pij)
    # two vertices inside (k outside): quad Pi, Pj, Pjk, Pki; the cut edge Pjk → Pki, the cap gets Pki → Pjk
    two = np.flatnonzero(count == 2)
    if len(two):
        k = np.argmin(inside[two], axis=1)
        order = (k[:, None] + 1 + np.arange(3)) % 3      # i, j, k
        t = tris[two[:, None], order]
        st = s[two[:, None], order]
        pjk = _lerp(t[:, 1], t[:, 2], st[:, 1], st[:, 2])
        pki = _lerp(t[:, 2], t[:, 0], st[:, 2], st[:, 0])
        keep.append(np.stack([t[:, 0], t[:, 1], pjk], 1))
        keep.append(np.stack([t[:, 0], pjk, pki], 1))
        seg_a.append(pki)
        seg_b.append(pjk)
    if seg_a:
        a, b = np.concatenate(seg_a), np.concatenate(seg_b)
        o = np.broadcast_to(a[0], a.shape)
        keep.append(np.stack([o, a, b], 1))
    out = np.concatenate(keep)
    area = np.linalg.norm(np.cross(out[:, 1] - out[:, 0], out[:, 2] - out[:, 0]), axis=1)
    return out[area > 0.0]


def _points(tris: np.ndarray) -> np.ndarray:
    return np.unique(tris.reshape(-1, 3), axis=0)


def hull_volume(pts: np.ndarray, max_points: int = 0) -> float:
    """Volume of the convex hull of `pts` (0 for flat sets). With max_points > 0, larger point sets are first
    reduced to their support points in max_points directions (faster, slightly smaller). Points closer than
    1e-7 of the set's size are merged first (cuts near vertices leave such near-duplicates)."""
    if 0 < max_points < len(pts):
        pts = pts[np.unique(np.argmax(pts @ _fibonacci_directions(max_points).T, axis=0))]
    if len(pts) < 4:
        return 0.0
    size = float(np.max(np.ptp(pts, axis=0))) or 1.0
    pts = _dedupe(pts, 1e-7 * size)
    if len(pts) < 4:
        return 0.0
    try:
        tris = np.asarray(_triangle_hull(pts, HULL_EPS * size), np.int64)
    except HullError:
        return 0.0
    c = pts.mean(axis=0)
    t = pts[tris] - c
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def _concavity(tris: np.ndarray, max_points: int) -> tuple[float, float]:
    """(hull volume − volume, volume) of a piece."""
    vol = chain_volume(tris)
    if vol <= 0.0:
        return 0.0, 0.0
    return max(hull_volume(_points(tris), max_points) - vol, 0.0), vol


def _first_min(costs, tie: float) -> int:
    """Index of the first cost within `tie` of the minimum (rounding never decides between equal cuts)."""
    m = min(costs)
    return next(i for i, c in enumerate(costs) if c <= m + tie)


def _cuts(piece: np.ndarray, total: float, max_points: int, positions=POSITIONS, vertex_cuts: bool = True):
    """(cost, left, right, concavity left, concavity right) for every candidate cut of a piece: at fixed fractions
    of its extent along each axis, and just beside its vertex coordinates when it has only a few distinct ones."""
    pts = piece.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    out = []
    for axis in range(3):
        if hi[axis] - lo[axis] <= 1e-9 * max(float(np.max(hi - lo)), 1e-12):
            continue
        n = np.zeros(3)
        n[axis] = 1.0
        planes = [lo[axis] + f * (hi[axis] - lo[axis]) for f in positions]
        if vertex_cuts:
            inner = np.unique(pts[:, axis])
            inner = inner[(inner > lo[axis]) & (inner < hi[axis])]
            if len(inner) <= MAX_VERTEX_CUTS:
                # just beside each coordinate, never through it: faces lying in a cut plane would otherwise
                # belong to both sides
                delta = 1e-6 * (hi[axis] - lo[axis])
                planes += [float(x) + sgn * delta for x in inner for sgn in (-1.0, 1.0)]
        for d in planes:
            if min(d - lo[axis], hi[axis] - d) < MIN_SLICE * (hi[axis] - lo[axis]):
                continue                        # no slivers
            left, right = clip(piece, n, d), clip(piece, -n, -d)
            cl, vl = _concavity(left, max_points)
            cr, vr = _concavity(right, max_points)
            if vl > 1e-12 * total and vr > 1e-12 * total:
                out.append((cl + cr, left, right, cl, cr))
    return out


def _lookahead(cut, total: float, max_points: int) -> float:
    """Concavity left after a cut and the best further cut (through the middle) of each of its halves."""
    _, left, right, cl, cr = cut
    cost = 0.0
    for piece, c in ((left, cl), (right, cr)):
        further = [x[0] for x in _cuts(piece, total, max_points, LOOKAHEAD_POSITIONS, False)] if c > 0.0 else []
        cost += min([c, *further])
    return cost


def decompose(vertices, triangles, max_parts: int = MAX_PARTS, tolerance: float = TOLERANCE,
              max_hull_points: int = 0) -> list[np.ndarray]:
    """Vertex sets of the convex parts of a closed mesh (the convex hull of each set is one part).
    `max_hull_points` > 0 compares pieces through reduced hulls (see hull_volume); the native core uses 0."""
    v = np.asarray(vertices, float)
    tris = v[np.asarray(triangles, np.int64)]
    total = chain_volume(tris)
    if total <= 0.0:
        raise ValueError("the mesh is not a closed solid with outward normals")
    tie = 1e-9 * total
    pieces = [tris]
    conc = [_concavity(tris, max_hull_points)[0]]
    while len(pieces) < max_parts and sum(conc) > tolerance * total:
        k = next(i for i, c in enumerate(conc) if c >= max(conc) - tie)
        cuts = _cuts(pieces[k], total, max_hull_points)
        if not cuts:
            conc[k] = 0.0                     # nothing to cut (flat piece): leave it as one part
            continue
        best = cuts[_first_min([c[0] for c in cuts], tie)]
        if best[0] > IMPROVEMENT * conc[k]:
            # no cut pays off at once (a ring needs two cuts before its pieces become convex): rank the cuts by
            # what one further cut of their halves can achieve
            best = cuts[_first_min([_lookahead(c, total, max_hull_points) for c in cuts], tie)]
        _, left, right, cl, cr = best
        pieces[k:k + 1] = [left, right]
        conc[k:k + 1] = [cl, cr]
    return [_points(p) for p in pieces]
