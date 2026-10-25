"""Software-rendered 3D thumbnails of object meshes for the object library.

Thumbnails are produced from the *actual* object geometry: triangles are
projected with a perspective camera, shaded with a key/fill light model and
painted back-to-front with QPainter. No OpenGL context is required, so
thumbnails can be generated before the viewport exists.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPolygonF

from ..scene.geometry import Mesh

_MAX_TRIANGLES = 6000


def render_thumbnail(mesh: Mesh, size: int, color=(0.45, 0.62, 0.88), azimuth_deg: float = 35.0,
                     elevation_deg: float = 25.0, vertex_colors: bool = False, dpr: float = 1.0) -> QImage:
    px = int(size * dpr)
    img = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    img.setDevicePixelRatio(dpr)
    v = mesh.vertices.astype(np.float64)
    idx = mesh.indices
    if len(idx) == 0:
        return img
    if len(idx) > _MAX_TRIANGLES:
        idx = idx[:: int(math.ceil(len(idx) / _MAX_TRIANGLES))]
    lo, hi = v.min(axis=0), v.max(axis=0)
    center = (lo + hi) / 2
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-9)
    v = (v - center) / radius

    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    eye_dir = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    fwd = -eye_dir
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    dist = 3.2
    cam = eye_dir * dist
    rel = v - cam
    x = rel @ right
    y = rel @ up
    z = rel @ fwd
    f = 1.0 / math.tan(math.radians(17))
    sx = (x / z) * f
    sy = (y / z) * f
    span = max(np.abs(sx).max(), np.abs(sy).max(), 1e-9)
    scale = size * 0.44 / span
    X = size / 2 + sx * scale
    Y = size / 2 - sy * scale

    tri = idx
    p0, p1, p2 = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    valid = nl[:, 0] > 1e-12
    n = n / np.where(nl > 0, nl, 1)
    centroid = (p0 + p1 + p2) / 3
    view_vec = cam - centroid
    facing = np.einsum("ij,ij->i", n, view_vec)
    n[facing < 0] *= -1  # two-sided shading
    depth = (centroid - cam) @ fwd
    order = np.argsort(-depth)

    key = np.array([0.45, -0.35, 0.82])
    key /= np.linalg.norm(key)
    key = key[0] * right + key[1] * fwd * -1 + key[2] * up
    key /= np.linalg.norm(key)
    fill = eye_dir
    diff = np.clip(n @ key, 0, 1) * 0.75 + np.clip(n @ fill, 0, 1) * 0.30 + 0.18 + 0.08 * n[:, 2]
    if vertex_colors and mesh.colors is not None:
        base = (mesh.colors[tri[:, 0]] + mesh.colors[tri[:, 1]] + mesh.colors[tri[:, 2]]) / 3
    else:
        base = np.tile(np.asarray(color, float), (len(tri), 1))
    rgb = np.clip(base * diff[:, None], 0, 1)
    # gamma-ish lift for readability on dark backgrounds
    rgb = np.power(rgb, 0.85)

    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for k in order:
        if not valid[k]:
            continue
        a, b, c = tri[k]
        col = QColor.fromRgbF(*rgb[k])
        painter.setPen(col)
        painter.setBrush(col)
        painter.drawPolygon(QPolygonF([QPointF(X[a], Y[a]), QPointF(X[b], Y[b]), QPointF(X[c], Y[c])]))
    painter.end()
    return img
