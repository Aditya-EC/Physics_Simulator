"""Line geometry for global axes, object-local frames, bounding boxes and COM markers."""
from __future__ import annotations

import numpy as np

from .. import numerics
from ..scene import geometry as g
from .view_settings import AxesSettings


def global_axes(ax: AxesSettings) -> tuple[np.ndarray, np.ndarray]:
    o = np.zeros(3)
    pts, cols = [], []
    for i, c in enumerate((ax.x_color, ax.y_color, ax.z_color)):
        e = np.zeros(3)
        e[i] = ax.length * 0.86  # leave room for the arrow head
        pts += [o, o + e]
        cols += [[*c, 1.0]] * 2
    return np.array(pts), np.array(cols)


def axis_heads(ax: AxesSettings) -> list[tuple[g.Mesh, np.ndarray, list[float]]]:
    """Cone meshes (local +Z) and model matrices for the arrow heads of the global axes."""
    head_len = ax.length * 0.14
    cone = g.cone(head_len * 0.36, head_len, 24)
    out = []
    rots = [numerics.quat_from_axis_angle([0, 1, 0], np.pi / 2), numerics.quat_from_axis_angle([1, 0, 0], -np.pi / 2),
            np.array([1.0, 0, 0, 0])]
    for i, (q, c) in enumerate(zip(rots, (ax.x_color, ax.y_color, ax.z_color))):
        p = np.zeros(3)
        p[i] = ax.length - head_len / 2
        out.append((cone, np.asarray(numerics.trs_matrix(p, q, [1, 1, 1])), list(c)))
    return out


def frame_lines(position, quaternion, length: float, alpha: float = 1.0,
                colors=((0.91, 0.30, 0.28), (0.42, 0.80, 0.32), (0.32, 0.55, 0.97))) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(numerics.quat_to_matrix(quaternion), float)
    p = np.asarray(position, float)
    pts, cols = [], []
    for i in range(3):
        pts += [p, p + r[:, i] * length]
        cols += [[*colors[i], alpha]] * 2
    return np.array(pts), np.array(cols)


def box_lines(model: np.ndarray, lo, hi, color) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = numerics.transform_points(model, c)
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    pts = np.array([w[i] for e in edges for i in e])
    return pts, np.tile([*color, 1.0], (len(pts), 1))


def cross_lines(center, size: float, color) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(center, float)
    pts = []
    for i in range(3):
        e = np.zeros(3)
        e[i] = size
        pts += [c - e, c + e]
    return np.array(pts), np.tile([*color, 1.0], (len(pts), 1))
