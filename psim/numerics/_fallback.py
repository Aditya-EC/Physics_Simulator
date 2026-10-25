"""Pure NumPy implementation of the native numerical core API.

Used only when the compiled `_psim_native` module is unavailable. Every
function mirrors the signature and conventions of the C++ implementation
(see native/include/psim/vecmath.hpp); tests verify that both agree.
"""
from __future__ import annotations

import math

import numpy as np


def _v(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(3)


def _q(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(4)


def quat_normalize(q):
    q = _q(q)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    return -q if q[0] < 0 else q


def quat_multiply(a, b):
    aw, ax, ay, az = _q(a)
    bw, bx, by, bz = _q(b)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conjugate(q):
    q = _q(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_axis_angle(axis, angle):
    a = _v(axis)
    n = np.linalg.norm(a)
    a = a / n if n > 0 else a
    h = 0.5 * angle
    return quat_normalize(np.concatenate([[math.cos(h)], a * math.sin(h)]))


def quat_from_euler(e):
    rx, ry, rz = _v(e)
    qx = quat_from_axis_angle([1, 0, 0], rx)
    qy = quat_from_axis_angle([0, 1, 0], ry)
    qz = quat_from_axis_angle([0, 0, 1], rz)
    return quat_normalize(quat_multiply(qz, quat_multiply(qy, qx)))


def quat_to_matrix(q):
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_euler(q):
    m = quat_to_matrix(q)
    sy = -m[2, 0]
    if abs(sy) < 0.9999999:
        ry = math.asin(sy)
        rx = math.atan2(m[2, 1], m[2, 2])
        rz = math.atan2(m[1, 0], m[0, 0])
    else:
        ry = math.pi / 2 if sy > 0 else -math.pi / 2
        rx = 0.0
        rz = math.atan2(-m[0, 1], m[1, 1])
    return np.array([rx, ry, rz])


def matrix_to_quat(r):
    m = np.asarray(r, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return quat_normalize(q)


def quat_rotate(q, v):
    return quat_to_matrix(q) @ _v(v)


def pose_compose(p1, q1, p2, q2):
    return _v(p1) + quat_rotate(q1, p2), quat_normalize(quat_multiply(quat_normalize(q1), quat_normalize(q2)))


def pose_inverse(p, q):
    qi = quat_conjugate(quat_normalize(q))
    return -quat_rotate(qi, p), quat_normalize(qi)


def trs_matrix(t, q, s):
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(q) * _v(s)[None, :]
    m[:3, 3] = _v(t)
    return m


def look_at(eye, target, up):
    eye, target, up = _v(eye), _v(target), _v(up)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-12:
        s = np.cross(f, [0.0, 1.0, 0.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[0, 3] = s, -s @ eye
    m[1, :3], m[1, 3] = u, -u @ eye
    m[2, :3], m[2, 3] = -f, f @ eye
    return m


def perspective(fovy, aspect, n, f):
    t = 1.0 / math.tan(0.5 * fovy)
    m = np.zeros((4, 4))
    m[0, 0] = t / aspect
    m[1, 1] = t
    m[2, 2] = (f + n) / (n - f)
    m[2, 3] = 2 * f * n / (n - f)
    m[3, 2] = -1.0
    return m


def orthographic(l, r, b, t, n, f):
    m = np.eye(4)
    m[0, 0] = 2 / (r - l)
    m[1, 1] = 2 / (t - b)
    m[2, 2] = -2 / (f - n)
    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = -(f + n) / (f - n)
    return m


def transform_points(mat, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(mat, dtype=np.float64).T
    w = h[:, 3:4].copy()
    w[w == 0] = 1.0
    return h[:, :3] / w


def _check(v, f):
    v = np.asarray(v, dtype=np.float32).reshape(-1, 3)
    f = np.asarray(f, dtype=np.uint32).reshape(-1, 3)
    if f.size and int(f.max()) >= len(v):
        raise IndexError("triangle index out of range")
    return v, f


def ray_mesh_intersect(origin, direction, v, f):
    v, f = _check(v, f)
    if not len(f):
        return False, -1.0, -1
    o, d = _v(origin), _v(direction)
    p0 = v[f[:, 0]].astype(np.float64)
    e1 = v[f[:, 1]] - p0
    e2 = v[f[:, 2]] - p0
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) >= 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - p0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    vv = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (u <= 1) & (vv >= 0) & (u + vv <= 1) & (t > 1e-9)
    if not hit.any():
        return False, -1.0, -1
    idx = np.where(hit)[0]
    k = idx[np.argmin(t[idx])]
    return True, float(t[k]), int(k)


def compute_vertex_normals(v, f):
    v, f = _check(v, f)
    v64 = v.astype(np.float64)
    fn = np.cross(v64[f[:, 1]] - v64[f[:, 0]], v64[f[:, 2]] - v64[f[:, 0]])
    acc = np.zeros_like(v64)
    for k in range(3):
        np.add.at(acc, f[:, k], fn)
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (acc / n).astype(np.float32)


def feature_edges(v, f, angle_deg=30.0, weld_tolerance=1e-5):
    v, f = _check(v, f)
    tol = weld_tolerance if weld_tolerance > 0 else 1e-6
    keys = np.round(v.astype(np.float64) / tol).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    canon = first[inverse.reshape(-1)]
    c = canon[f]
    v64 = v.astype(np.float64)
    fn = np.cross(v64[c[:, 1]] - v64[c[:, 0]], v64[c[:, 2]] - v64[c[:, 0]])
    nn = np.linalg.norm(fn, axis=1)
    valid = nn >= 1e-14
    fn[valid] /= nn[valid, None]
    edges: dict[tuple[int, int], list[int]] = {}
    for t in np.where(valid)[0]:
        for k in range(3):
            a, b = int(c[t, k]), int(c[t, (k + 1) % 3])
            if a == b:
                continue
            edges.setdefault((min(a, b), max(a, b)), []).append(int(t))
    cos_limit = math.cos(math.radians(angle_deg))
    out = []
    for (a, b), faces in edges.items():
        if len(faces) == 1 or (len(faces) == 2 and fn[faces[0]] @ fn[faces[1]] < cos_limit):
            out.extend([v[a], v[b]])
    return np.asarray(out, dtype=np.float32).reshape(-1, 3)


def mass_properties_solid(v, f, density):
    v, f = _check(v, f)
    p0, p1, p2 = (v[f[:, k]].astype(np.float64) for k in range(3))
    d = np.cross(p1 - p0, p2 - p0)

    def sub(w0, w1, w2):
        temp0 = w0 + w1
        f1 = temp0 + w2
        temp1 = w0 * w0
        temp2 = temp1 + w1 * temp0
        f2 = temp2 + w2 * f1
        f3 = w0 * temp1 + w1 * temp2 + w2 * f2
        return f1, f2, f3, f2 + w0 * (f1 + w0), f2 + w1 * (f1 + w1), f2 + w2 * (f1 + w2)

    f1x, f2x, f3x, g0x, g1x, g2x = sub(p0[:, 0], p1[:, 0], p2[:, 0])
    f1y, f2y, f3y, g0y, g1y, g2y = sub(p0[:, 1], p1[:, 1], p2[:, 1])
    f1z, f2z, f3z, g0z, g1z, g2z = sub(p0[:, 2], p1[:, 2], p2[:, 2])
    i = np.array([
        (d[:, 0] * f1x).sum() / 6,
        (d[:, 0] * f2x).sum() / 24, (d[:, 1] * f2y).sum() / 24, (d[:, 2] * f2z).sum() / 24,
        (d[:, 0] * f3x).sum() / 60, (d[:, 1] * f3y).sum() / 60, (d[:, 2] * f3z).sum() / 60,
        (d[:, 0] * (p0[:, 1] * g0x + p1[:, 1] * g1x + p2[:, 1] * g2x)).sum() / 120,
        (d[:, 1] * (p0[:, 2] * g0y + p1[:, 2] * g1y + p2[:, 2] * g2y)).sum() / 120,
        (d[:, 2] * (p0[:, 0] * g0z + p1[:, 0] * g1z + p2[:, 0] * g2z)).sum() / 120,
    ])
    vol = i[0]
    out = {"measure": float(vol), "mass": 0.0, "centroid": np.zeros(3), "inertia": np.zeros((3, 3))}
    if abs(vol) < 1e-300:
        return out
    cm = i[1:4] / vol
    ixx = i[5] + i[6] - vol * (cm[1] ** 2 + cm[2] ** 2)
    iyy = i[4] + i[6] - vol * (cm[2] ** 2 + cm[0] ** 2)
    izz = i[4] + i[5] - vol * (cm[0] ** 2 + cm[1] ** 2)
    ixy = -(i[7] - vol * cm[0] * cm[1])
    iyz = -(i[8] - vol * cm[1] * cm[2])
    ixz = -(i[9] - vol * cm[2] * cm[0])
    out.update(mass=float(vol * density), centroid=cm,
               inertia=density * np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]))
    return out


def mass_properties_shell(v, f, areal_density):
    v, f = _check(v, f)
    p = [v[f[:, k]].astype(np.float64) for k in range(3)]
    a = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]), axis=1)
    area = float(a.sum())
    out = {"measure": area, "mass": 0.0, "centroid": np.zeros(3), "inertia": np.zeros((3, 3))}
    if area <= 0:
        return out
    s = p[0] + p[1] + p[2]
    cm = (s * (a / 3.0)[:, None]).sum(axis=0) / area
    second = np.einsum("t,ti,tj->ij", a / 12.0, s, s)
    for k in range(3):
        second += np.einsum("t,ti,tj->ij", a / 12.0, p[k], p[k])
    c = second - area * np.outer(cm, cm)
    out.update(mass=area * areal_density, centroid=cm,
               inertia=areal_density * (np.trace(c) * np.eye(3) - c))
    return out


def _require_inplace(a, cols: int, name: str) -> None:
    if not (isinstance(a, np.ndarray) and a.dtype == np.float64 and a.flags.c_contiguous and a.flags.writeable
            and a.ndim == 2 and a.shape[1] == cols):
        raise TypeError(f"{name} must be a writable C-contiguous float64 array of shape (N, {cols})")


def integrate_linear(method, x, v, a, dt):
    _require_inplace(x, 3, "x")
    _require_inplace(v, 3, "v")
    a = np.asarray(a, dtype=np.float64).reshape(x.shape)
    if method == "explicit_euler":
        x += v * dt
        v += a * dt
    elif method == "semi_implicit_euler":
        v += a * dt
        x += v * dt
    else:
        raise ValueError(f"unknown integration method '{method}'")


def quat_integrate(q, w, dt):
    _require_inplace(q, 4, "q")
    w = np.asarray(w, dtype=np.float64).reshape(len(q), 3)
    speed = np.linalg.norm(w, axis=1)
    angle = speed * dt
    moving = angle > 0
    axis = np.zeros_like(w)
    axis[moving] = w[moving] / speed[moving, None]
    h = 0.5 * angle
    dq = np.concatenate([np.cos(h)[:, None], axis * np.sin(h)[:, None]], axis=1)
    aw, ax, ay, az = dq.T
    bw, bx, by, bz = q.T.copy()
    out = np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)
    q[:] = out / np.linalg.norm(out, axis=1, keepdims=True)


def _quat_matrices(q):
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def quat_rotate_batch(q, v):
    return np.einsum("nij,nj->ni", _quat_matrices(q), np.asarray(v, dtype=np.float64).reshape(-1, 3))


def rotate_tensors(q, t):
    r = _quat_matrices(q)
    return r @ np.asarray(t, dtype=np.float64).reshape(-1, 3, 3) @ np.transpose(r, (0, 2, 1))
