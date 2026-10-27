"""Convex hulls for collision shapes (computed once per run, when the solver is set up).

A ``Hull`` holds vertices relative to the collider centre, polygonal faces (coplanar triangles merged,
vertices counter-clockwise around the outward normal, n·x ≤ d inside) and the edges between faces with
their two adjacent faces (for the Gauss-map pruning of edge pairs in the separating-axis test).
``pack_hulls`` flattens hulls into the arrays taken by ``numerics.contact_rows``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HULL_EPS = 1e-9           # coplanarity tolerance relative to the size
MAX_VERTICES = 100         # meshes with more distinct vertices are reduced to support points in REDUCED_DIRECTIONS
REDUCED_DIRECTIONS = 64


class HullError(ValueError):
    pass


@dataclass
class Hull:
    verts: np.ndarray        # (V, 3)
    normals: np.ndarray      # (F, 3) outward unit normals
    offsets: np.ndarray      # (F,) n·x ≤ d inside
    faces: list              # F lists of vertex indices, counter-clockwise around the normal
    edges: np.ndarray        # (E, 2) vertex indices (a, b)
    edge_faces: np.ndarray   # (E, 2) the face with directed edge a→b, and the face with b→a
    reduced: bool = False    # built from support points of a detailed mesh (approximate)

    @property
    def volume(self) -> float:
        vol = 0.0
        for f in self.faces:
            p0 = self.verts[f[0]]
            for a, b in zip(f[1:-1], f[2:]):
                vol += float(np.dot(p0, np.cross(self.verts[a], self.verts[b]))) / 6.0
        return vol


def _fibonacci_directions(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = math.pi * (1 + 5 ** 0.5) * k
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def _dedupe(points: np.ndarray, tol: float) -> np.ndarray:
    key = np.round(points / tol).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return points[np.sort(idx)]


def _horizon(faces, owner, region) -> list[tuple[int, int]] | None:
    """Boundary edges of a set of faces, or None unless they form one simple loop."""
    out = []
    for k in sorted(region):
        a, b, c = faces[k]
        for e in ((a, b), (b, c), (c, a)):
            if owner[(e[1], e[0])] not in region:
                out.append(e)
    nxt = dict(out)
    if len(nxt) != len(out) or not out:
        return None                                   # pinched: a vertex starts two boundary edges
    v, steps = out[0][0], 0
    while True:
        v = nxt.get(v)
        steps += 1
        if v is None or steps > len(out):
            return None
        if v == out[0][0]:
            return out if steps == len(out) else None  # several loops: a hole in the region


def _triangle_hull(p: np.ndarray, eps: float) -> list[tuple[int, int, int]]:
    """Quickhull: outward-oriented triangles over indices of p (≥ 4 non-coplanar points).

    Each face keeps the points above it; the farthest point above the first face that has any is added next,
    replacing the faces it sees (reached from that face across edges, distance > eps). Points within eps of the
    hull are dropped. Adding far points first keeps the hull robust with many (nearly) coplanar points. If the
    replaced faces do not form a disc (a pinch or hole, only with noise near eps), nearly coplanar neighbours are
    taken in as well; a point for which even that fails is left out."""
    n = len(p)
    # initial tetrahedron from extreme points
    i0 = int(np.argmin(p[:, 0]))
    i1 = int(np.argmax(np.linalg.norm(p - p[i0], axis=1)))
    line = p[i1] - p[i0]
    d2 = np.linalg.norm(np.cross(p - p[i0], line), axis=1)
    i2 = int(np.argmax(d2))
    nrm = np.cross(p[i1] - p[i0], p[i2] - p[i0])
    d3 = np.abs((p - p[i0]) @ nrm)
    i3 = int(np.argmax(d3))
    if d2[i2] <= eps * np.linalg.norm(line) or d3[i3] <= eps * np.linalg.norm(nrm):
        raise HullError("points are coplanar")
    centre = p[[i0, i1, i2, i3]].mean(axis=0)
    faces: list[tuple[int, int, int]] = []        # by face id; removed faces stay, marked dead
    normal: list[np.ndarray] = []
    offset: list[float] = []
    alive: list[bool] = []
    outside: dict[int, np.ndarray] = {}           # live face → indices of the points above it
    owner: dict[tuple[int, int], int] = {}        # directed edge → id of the live face that has it

    def add(a, b, c) -> int:
        nn = np.cross(p[b] - p[a], p[c] - p[a])
        ln = float(np.linalg.norm(nn))
        nn = nn / ln if ln > 0 else nn
        k = len(faces)
        faces.append((a, b, c))
        normal.append(nn)
        offset.append(float(nn @ p[a]))
        alive.append(True)
        owner[(a, b)] = owner[(b, c)] = owner[(c, a)] = k
        return k

    def assign(points: np.ndarray, new: list[int]) -> None:
        """Give each point to the new face it is farthest above (points within eps are inside)."""
        if not len(points) or not new:
            return
        d = p[points] @ np.array([normal[k] for k in new]).T - np.array([offset[k] for k in new])
        best = np.argmax(d, axis=1)
        above = d[np.arange(len(points)), best] > eps
        for j, k in enumerate(new):
            sel = points[above & (best == j)]
            if len(sel):
                outside[k] = sel

    tet = []
    for a, b, c in ((i0, i1, i2), (i0, i1, i3), (i0, i2, i3), (i1, i2, i3)):
        if np.cross(p[b] - p[a], p[c] - p[a]) @ (p[a] - centre) < 0:
            b, c = c, b
        tet.append(add(a, b, c))
    assign(np.array([i for i in range(n) if i not in (i0, i1, i2, i3)], np.int64), tet)
    while outside:
        f0 = min(outside)
        pts = outside[f0]
        d = p[pts] @ normal[f0] - offset[f0]
        i = int(pts[int(np.argmax(d))])

        def dist(k):
            return float(normal[k] @ p[i] - offset[k])
        region = {f0}
        stack = [f0]
        while stack:                              # the faces the point sees, connected to f0
            a, b, c = faces[stack.pop()]
            for e in ((b, a), (c, b), (a, c)):
                g = owner[e]
                if g not in region and dist(g) > eps:
                    region.add(g)
                    stack.append(g)
        horizon = _horizon(faces, owner, region)
        for attempt in range(2):                  # repair a pinch or hole: nearly coplanar neighbours, then all
            if horizon is not None:
                break
            near = {owner[(e[1], e[0])] for k in region for e in ((faces[k][0], faces[k][1]),
                                                                   (faces[k][1], faces[k][2]),
                                                                   (faces[k][2], faces[k][0]))}
            region |= {g for g in near if g not in region and (attempt == 1 or dist(g) > -eps)}
            horizon = _horizon(faces, owner, region)
        freed = [outside.pop(k) for k in sorted(region) if k in outside]
        freed = np.concatenate(freed) if freed else np.zeros(0, np.int64)
        freed = freed[freed != i]
        if horizon is None:                       # could not be repaired: leave the point out
            assign(freed, sorted(region))
            continue
        for k in region:
            alive[k] = False
            a, b, c = faces[k]
            for e in ((a, b), (b, c), (c, a)):
                if owner.get(e) == k:
                    del owner[e]
        new = [add(a, b, i) for a, b in horizon]   # keeps the orientation of the removed faces
        assign(freed, new)
    return [f for f, ok in zip(faces, alive) if ok]


def convex_hull(points, scale: float | None = None, max_vertices: int = MAX_VERTICES) -> Hull:
    """Hull of a point cloud. Flat point sets are given a small thickness (1e-3 of their size)."""
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) < 1:
        raise HullError("no points")
    size = scale or float(np.max(np.ptp(pts, axis=0))) or 1.0
    eps = HULL_EPS * size
    pts = _dedupe(pts, 1e-7 * size)
    reduced = False
    if len(pts) > max_vertices:
        dirs = _fibonacci_directions(min(REDUCED_DIRECTIONS, max_vertices))
        idx = np.unique(np.argmax(pts @ dirs.T, axis=0))
        pts, reduced = pts[idx], True
    c = pts.mean(axis=0)
    centred = pts - c
    _, sv, vt = np.linalg.svd(centred, full_matrices=False) if len(pts) >= 3 else (None, np.zeros(3), np.eye(3))
    if len(pts) < 4 or sv[-1] <= 1e-7 * max(sv[0], 1e-300) * max(len(pts), 1) ** 0.5:
        n = vt[-1] if len(pts) >= 3 else np.array([0.0, 0.0, 1.0])
        t = 0.5e-3 * size
        pts = np.r_[pts + n * t, pts - n * t]
    tris = _triangle_hull(pts, eps)
    used = sorted({i for t in tris for i in t})
    remap = {old: new for new, old in enumerate(used)}
    verts = pts[used]
    tris = [tuple(remap[i] for i in t) for t in tris]
    # merge coplanar neighbours into polygons
    tn, td = [], []
    for a, b, cc in tris:
        n_ = np.cross(verts[b] - verts[a], verts[cc] - verts[a])
        n_ /= np.linalg.norm(n_)
        tn.append(n_)
        td.append(float(n_ @ verts[a]))
    parent = list(range(len(tris)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edge_tri = {}
    for k, (a, b, cc) in enumerate(tris):
        for e in ((a, b), (b, cc), (cc, a)):
            edge_tri[e] = k
    tol_d = 1e-7 * size
    for (a, b), k in edge_tri.items():
        k2 = edge_tri.get((b, a))
        if k2 is not None and tn[k] @ tn[k2] > 1 - 1e-9 and abs(td[k] - td[k2]) <= tol_d:
            parent[find(k)] = find(k2)
    groups: dict[int, list[int]] = {}
    for k in range(len(tris)):
        groups.setdefault(find(k), []).append(k)
    loops, loop_normals = [], []
    for g in sorted(groups.values(), key=lambda g: min(g)):
        boundary = {}
        for k in g:
            a, b, cc = tris[k]
            for e in ((a, b), (b, cc), (cc, a)):
                if (e[1], e[0]) in boundary:
                    del boundary[(e[1], e[0])]
                else:
                    boundary[e] = True
        nxt = {a: b for a, b in boundary}
        start = min(nxt)
        loop = [start]
        while nxt[loop[-1]] != start:
            loop.append(nxt[loop[-1]])
            if len(loop) > len(nxt):
                raise HullError("could not order a hull face")
        n_ = np.mean([tn[k] for k in g], axis=0)
        loops.append(loop)
        loop_normals.append(n_ / np.linalg.norm(n_))
    # drop vertices that lie on an edge (merged triangles leave them), but only when they are collinear in every
    # face that contains them, so that neighbouring faces keep matching edges
    keep_vertex: dict[int, bool] = {}
    for loop in loops:
        for i, v in enumerate(loop):
            pa, pb = verts[loop[i - 1]], verts[loop[(i + 1) % len(loop)]]
            corner = bool(np.linalg.norm(np.cross(verts[v] - pa, pb - verts[v])) > 1e-9 * size * size)
            keep_vertex[v] = keep_vertex.get(v, False) or corner
    normals, offsets, faces = [], [], []
    face_of_edge = {}
    for loop, n_ in zip(loops, loop_normals):
        clean = [v for v in loop if keep_vertex[v]]
        if len(clean) < 3:
            clean = loop
        fi = len(faces)
        faces.append(clean)
        normals.append(n_)
        offsets.append(float(max(verts[v] @ n_ for v in clean)))
        for i, v in enumerate(clean):
            face_of_edge[(v, clean[(i + 1) % len(clean)])] = fi
    edges, edge_faces = [], []
    for (a, b), fi in sorted(face_of_edge.items()):
        if a > b:
            continue
        other = face_of_edge.get((b, a))
        if other is None:
            raise HullError("open hull edge")
        edges.append((a, b))
        edge_faces.append((fi, other))
    # keep only the vertices of the polygons (merging coplanar triangles leaves interior and edge points out)
    used = sorted({v for f in faces for v in f})
    remap = {old: new for new, old in enumerate(used)}
    faces = [[remap[v] for v in f] for f in faces]
    edges = [(remap[a], remap[b]) for a, b in edges]
    return Hull(verts[used], np.array(normals), np.array(offsets), faces, np.array(edges, np.int64).reshape(-1, 2),
                np.array(edge_faces, np.int64).reshape(-1, 2), reduced)


def translated(h: Hull, t) -> Hull:
    """The same hull moved by t (face offsets follow)."""
    t = np.asarray(t, float)
    return Hull(h.verts + t, h.normals, h.offsets + h.normals @ t, h.faces, h.edges, h.edge_faces, h.reduced)


_HULL_CACHE: dict[bytes, Hull] = {}


def mesh_hull(vertices) -> Hull:
    """Hull of a mesh's vertices, cached by their exact values (identical objects share one hull)."""
    v = np.ascontiguousarray(np.asarray(vertices, float).reshape(-1, 3))
    key = v.tobytes()
    h = _HULL_CACHE.get(key)
    if h is None:
        if len(_HULL_CACHE) > 256:
            _HULL_CACHE.clear()
        h = _HULL_CACHE[key] = convex_hull(v)
    return h


_BOX_CACHE: dict[tuple, Hull] = {}


def box_hull(half) -> Hull:
    key = tuple(np.round(np.asarray(half, float), 12))
    h = _BOX_CACHE.get(key)
    if h is None:
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * np.asarray(half)
        h = _BOX_CACHE[key] = convex_hull(corners)
    return h


def pack_hulls(hulls: list[Hull]) -> dict:
    """Flatten hulls for numerics.contact_rows (offset arrays have one more entry than items)."""
    def offsets(sizes):
        return np.r_[0, np.cumsum(sizes)].astype(np.int64)
    face_sizes = [len(f) for h in hulls for f in h.faces]
    return {
        "verts": np.concatenate([h.verts for h in hulls]) if hulls else np.zeros((0, 3)),
        "vert_off": offsets([len(h.verts) for h in hulls]),
        "normals": np.concatenate([h.normals for h in hulls]) if hulls else np.zeros((0, 3)),
        "offsets": np.concatenate([h.offsets for h in hulls]) if hulls else np.zeros(0),
        "face_off": offsets([len(h.faces) for h in hulls]),
        "face_verts": np.array([v for h in hulls for f in h.faces for v in f], np.int64),
        "face_vert_off": offsets(face_sizes),
        "edges": np.concatenate([h.edges for h in hulls]) if hulls else np.zeros((0, 2), np.int64),
        "edge_faces": np.concatenate([h.edge_faces for h in hulls]) if hulls else np.zeros((0, 2), np.int64),
        "edge_off": offsets([len(h.edges) for h in hulls]),
    }


def unpack_hulls(packed: dict) -> list[Hull]:
    """Inverse of pack_hulls (used by the NumPy contact kernel)."""
    out = []
    vo, fo, eo = packed["vert_off"], packed["face_off"], packed["edge_off"]
    fvo = packed["face_vert_off"]
    for h in range(len(vo) - 1):
        f0, f1 = int(fo[h]), int(fo[h + 1])
        faces = [packed["face_verts"][fvo[f]:fvo[f + 1]].tolist() for f in range(f0, f1)]
        out.append(Hull(packed["verts"][vo[h]:vo[h + 1]], packed["normals"][f0:f1], packed["offsets"][f0:f1], faces,
                        packed["edges"][eo[h]:eo[h + 1]], packed["edge_faces"][eo[h]:eo[h + 1]]))
    return out
