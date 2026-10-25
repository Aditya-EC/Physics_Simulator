"""Orbit camera with perspective/orthographic projection (Z-up world).

The eye position is target + distance * (cos(el)cos(az), cos(el)sin(az), sin(el)).
Standard views: Front looks along +Y (eye on -Y), Right looks along -X,
Top looks down -Z with +X to the right and +Y up on screen.
"""
from __future__ import annotations

import math

import numpy as np

from .. import numerics
from .view_settings import CameraState

STANDARD_VIEWS = {
    "front": (-90.0, 0.0), "back": (90.0, 0.0), "right": (0.0, 0.0), "left": (180.0, 0.0),
    "top": (-90.0, 90.0), "bottom": (-90.0, -90.0), "isometric": (-45.0, math.degrees(math.atan(1 / math.sqrt(2)))),
}


class OrbitCamera:
    def __init__(self, state: CameraState | None = None) -> None:
        self.state = state or CameraState()
        self.scene_radius = 10.0

    # ---- basic geometry --------------------------------------------------------------
    @property
    def target(self) -> np.ndarray:
        return np.asarray(self.state.target, float)

    @target.setter
    def target(self, value) -> None:
        self.state.target = [float(c) for c in value]

    def _angles(self) -> tuple[float, float]:
        return math.radians(self.state.azimuth_deg), math.radians(self.state.elevation_deg)

    def forward(self) -> np.ndarray:
        az, el = self._angles()
        return -np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])

    def up(self) -> np.ndarray:
        az, el = self._angles()
        return np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])

    def right(self) -> np.ndarray:
        return np.cross(self.forward(), self.up())

    def eye(self) -> np.ndarray:
        return self.target - self.forward() * self.state.distance

    def view_matrix(self) -> np.ndarray:
        return np.asarray(numerics.look_at(self.eye(), self.target, self.up()), float)

    def ortho_half_height(self) -> float:
        return self.state.distance * math.tan(math.radians(self.state.fov_deg) / 2)

    def projection_matrix(self, aspect: float) -> np.ndarray:
        aspect = max(aspect, 1e-6)
        d = self.state.distance
        r = max(self.scene_radius, 1.0)
        if self.state.orthographic:
            h = self.ortho_half_height()
            depth = d + 4 * r + 100.0
            return np.asarray(numerics.orthographic(-h * aspect, h * aspect, -h, h, -depth, depth), float)
        near = max(d * 0.005, 1e-4)
        far = d + 4 * r + 1000.0
        return np.asarray(numerics.perspective(math.radians(self.state.fov_deg), aspect, near, far), float)

    def view_projection(self, aspect: float) -> np.ndarray:
        return self.projection_matrix(aspect) @ self.view_matrix()

    # ---- interaction -------------------------------------------------------------------
    def orbit(self, d_azimuth_deg: float, d_elevation_deg: float) -> None:
        self.state.azimuth_deg = (self.state.azimuth_deg + d_azimuth_deg + 180.0) % 360.0 - 180.0
        self.state.elevation_deg = float(np.clip(self.state.elevation_deg + d_elevation_deg, -89.9, 89.9))

    def pan_pixels(self, dx: float, dy: float, viewport_height: float) -> None:
        """Move the target so the scene follows the cursor."""
        world_per_px = 2 * self.ortho_half_height() / max(viewport_height, 1.0)
        self.target = self.target - self.right() * dx * world_per_px + self.up() * dy * world_per_px

    def zoom(self, factor: float, toward: np.ndarray | None = None) -> None:
        """factor < 1 moves closer. If `toward` is given, zoom about that world point."""
        old = self.state.distance
        new = float(np.clip(old * factor, 1e-3, 1e6))
        if toward is not None:
            self.target = self.target + (np.asarray(toward, float) - self.target) * (1 - new / old)
        self.state.distance = new

    def set_view(self, name: str) -> None:
        az, el = STANDARD_VIEWS[name]
        self.state.azimuth_deg, self.state.elevation_deg = az, el

    def fit_bounds(self, lo: np.ndarray, hi: np.ndarray, aspect: float = 1.6, margin: float = 1.15) -> None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        center = (lo + hi) / 2
        radius = max(float(np.linalg.norm(hi - lo)) / 2, 0.25)
        half_fov = math.radians(self.state.fov_deg) / 2
        fov_limit = min(half_fov, math.atan(math.tan(half_fov) * max(aspect, 1e-3)))
        self.target = center
        self.state.distance = radius * margin / math.sin(fov_limit)

    # ---- projection helpers ------------------------------------------------------------------
    def screen_ray(self, x: float, y: float, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
        """World-space ray (origin, unit direction) through logical pixel (x, y)."""
        ndc_x = 2 * x / max(width, 1) - 1
        ndc_y = 1 - 2 * y / max(height, 1)
        inv = np.linalg.inv(self.view_projection(width / max(height, 1)))
        p0 = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0])
        p1 = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0])
        p0 = p0[:3] / p0[3]
        p1 = p1[:3] / p1[3]
        d = p1 - p0
        d /= np.linalg.norm(d)
        if not self.state.orthographic:
            p0 = self.eye()
        return p0, d

    def project(self, points: np.ndarray, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to logical pixels. Returns (xy (N,2), visible mask)."""
        pts = np.asarray(points, float).reshape(-1, 3)
        vp = self.view_projection(width / max(height, 1))
        h = np.hstack([pts, np.ones((len(pts), 1))]) @ vp.T
        w = h[:, 3]
        ok = w > 1e-9
        ndc = h[:, :3] / np.where(ok, w, 1.0)[:, None]
        xy = np.stack([(ndc[:, 0] + 1) / 2 * width, (1 - ndc[:, 1]) / 2 * height], 1)
        ok &= (ndc[:, 2] >= -1) & (ndc[:, 2] <= 1)
        return xy, ok

    def world_per_pixel(self, point: np.ndarray, viewport_height: float) -> float:
        if self.state.orthographic:
            return 2 * self.ortho_half_height() / max(viewport_height, 1)
        depth = float(np.dot(np.asarray(point, float) - self.eye(), self.forward()))
        depth = max(depth, 1e-4)
        return 2 * depth * math.tan(math.radians(self.state.fov_deg) / 2) / max(viewport_height, 1)
