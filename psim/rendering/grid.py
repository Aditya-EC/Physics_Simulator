"""Grid line geometry for the XY, YZ and XZ reference planes."""
from __future__ import annotations

import numpy as np

from .view_settings import GridSettings

# (u axis, v axis, normal axis) indices for each plane
PLANE_AXES = {"XY": (0, 1, 2), "YZ": (1, 2, 0), "XZ": (0, 2, 1)}
AXIS_RGB = {0: (0.91, 0.30, 0.28), 1: (0.42, 0.80, 0.32), 2: (0.32, 0.55, 0.97)}


def _plane_points(u_vals, v_vals, plane: str, offset: float = 0.0) -> np.ndarray:
    ui, vi, ni = PLANE_AXES[plane]
    pts = np.zeros((len(u_vals), 3))
    pts[:, ui] = u_vals
    pts[:, vi] = v_vals
    pts[:, ni] = offset
    return pts


def grid_lines(gs: GridSettings) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {"minor": (points, colors), "major": ..., "origin": ...} as GL_LINES vertex arrays."""
    spacing = max(gs.major_spacing, 1e-6)
    extent = max(gs.extent, spacing)
    n_major = int(np.floor(extent / spacing))
    sub = max(1, int(gs.subdivisions))
    minor_step = spacing / sub
    n_minor = int(np.floor(extent / minor_step))
    # keep the line count bounded for extreme settings
    if n_minor > 2000:
        n_minor, sub = 0, 1
    color = np.asarray(gs.color, float)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def lines(values, kind):
        segs = []
        for c in values:
            segs.append(_plane_points([c, c], [-extent, extent], gs.plane))
            segs.append(_plane_points([-extent, extent], [c, c], gs.plane))
        if not segs:
            return np.zeros((0, 3)), np.zeros((0, 4))
        pts = np.vstack(segs)
        alpha = {"major": 1.0, "minor": 0.45}[kind] * gs.opacity
        cols = np.tile([*color, alpha], (len(pts), 1))
        return pts, cols

    majors = [k * spacing for k in range(-n_major, n_major + 1) if not (gs.emphasize_origin and k == 0)]
    minors = [k * minor_step for k in range(-n_minor, n_minor + 1) if k % sub != 0]
    out["major"] = lines(majors, "major")
    out["minor"] = lines(minors, "minor")
    if gs.emphasize_origin:
        ui, vi, _ = PLANE_AXES[gs.plane]
        pts = np.vstack([_plane_points([0, 0], [-extent, extent], gs.plane),
                         _plane_points([-extent, extent], [0, 0], gs.plane)])
        cu, cv = np.asarray(AXIS_RGB[vi]), np.asarray(AXIS_RGB[ui])  # line along v is the v axis
        a = min(1.0, gs.opacity * 1.4)
        cols = np.array([[*cu * 0.85, a]] * 2 + [[*cv * 0.85, a]] * 2)
        out["origin"] = (pts, cols)
    return out


def grid_labels(gs: GridSettings, max_labels: int = 41) -> list[tuple[np.ndarray, str]]:
    """World positions and texts for major-line labels along the plane's two axes."""
    spacing = max(gs.major_spacing, 1e-6)
    n = int(np.floor(max(gs.extent, spacing) / spacing))
    step = max(1, int(np.ceil((2 * n + 1) / max_labels)))
    out = []
    for k in range(-n, n + 1, step):
        if k == 0:
            continue
        val = k * spacing
        text = f"{val:g}"
        out.append((_plane_points([val], [0.0], gs.plane)[0], text))
        out.append((_plane_points([0.0], [val], gs.plane)[0], text))
    return out


def ground_quad(extent: float) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[-extent, -extent, 0], [extent, -extent, 0], [extent, extent, 0], [-extent, extent, 0]], np.float32)
    return v, np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
