"""Serializable viewport configuration: grids, axes, snapping, environment, camera.

Pure data (no OpenGL) so it can be saved in projects and unit tested.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class GridSettings:
    plane: str = "XY"                 # XY, YZ or XZ
    visible: bool = True
    color: list[float] = field(default_factory=lambda: [0.55, 0.58, 0.64])
    opacity: float = 0.55
    major_spacing: float = 1.0        # metres
    subdivisions: int = 5             # minor lines per major cell
    extent: float = 20.0              # half-size, metres
    line_width: float = 1.0           # pixels (major lines; minor lines are thinner)
    show_labels: bool = False
    emphasize_origin: bool = True


@dataclass
class AxesSettings:
    visible: bool = True
    # where the global axes are drawn (display only: global coordinates are always measured from (0, 0, 0))
    origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    length: float = 1.5
    arrow_size: float = 1.0           # arrow-head size relative to the default (14 % of the length)
    line_width: float = 3.0
    x_color: list[float] = field(default_factory=lambda: [0.91, 0.30, 0.28])
    y_color: list[float] = field(default_factory=lambda: [0.42, 0.80, 0.32])
    z_color: list[float] = field(default_factory=lambda: [0.32, 0.55, 0.97])
    show_labels: bool = True
    show_origin_marker: bool = True
    show_view_triad: bool = True


@dataclass
class SnapSettings:
    grid_snap: bool = True
    translate_increment: float = 0.25
    rotate_snap: bool = False
    rotate_increment_deg: float = 15.0
    scale_snap: bool = False
    scale_increment: float = 0.1
    surface_snap: bool = True
    work_plane: str = "XY"            # placement plane for drag-and-drop
    work_plane_offset: float = 0.0


@dataclass
class EnvironmentSettings:
    background_top: list[float] = field(default_factory=lambda: [0.17, 0.18, 0.20])
    background_bottom: list[float] = field(default_factory=lambda: [0.10, 0.105, 0.115])
    shadows: bool = True
    ground_shadow: bool = True
    ambient: float = 0.35
    light_azimuth_deg: float = 35.0
    light_elevation_deg: float = 55.0
    light_intensity: float = 2.6
    show_labels: bool = True          # master switch for object overlays
    show_local_axes: bool = False     # show local axes on every object
    highlight_color: list[float] = field(default_factory=lambda: [0.98, 0.62, 0.18])
    show_stats: bool = True
    msaa_samples: int = 4


@dataclass
class CameraState:
    target: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    distance: float = 9.0
    azimuth_deg: float = -55.0        # around +Z, 0 = looking along +Y from -Y side... see camera.py
    elevation_deg: float = 28.0
    fov_deg: float = 45.0
    orthographic: bool = False


@dataclass
class ViewSettings:
    grids: dict[str, GridSettings] = field(default_factory=lambda: {
        "XY": GridSettings("XY", True),
        "YZ": GridSettings("YZ", False, [0.62, 0.55, 0.55]),
        "XZ": GridSettings("XZ", False, [0.55, 0.62, 0.55]),
    })
    axes: AxesSettings = field(default_factory=AxesSettings)
    snap: SnapSettings = field(default_factory=SnapSettings)
    environment: EnvironmentSettings = field(default_factory=EnvironmentSettings)
    camera: CameraState = field(default_factory=CameraState)
    bookmarks: dict[str, CameraState] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "grids": {k: asdict(v) for k, v in self.grids.items()},
            "axes": asdict(self.axes), "snap": asdict(self.snap), "environment": asdict(self.environment),
            "camera": asdict(self.camera), "bookmarks": {k: asdict(v) for k, v in self.bookmarks.items()},
        }

    @staticmethod
    def from_dict(data: dict) -> "ViewSettings":
        vs = ViewSettings()
        for k, g in dict(data.get("grids", {})).items():
            if k in vs.grids:
                vs.grids[k] = _load(GridSettings, g, vs.grids[k])
                vs.grids[k].plane = k
        vs.axes = _load(AxesSettings, data.get("axes", {}), vs.axes)
        vs.snap = _load(SnapSettings, data.get("snap", {}), vs.snap)
        vs.environment = _load(EnvironmentSettings, data.get("environment", {}), vs.environment)
        vs.camera = _load(CameraState, data.get("camera", {}), vs.camera)
        vs.bookmarks = {str(k): _load(CameraState, v, CameraState()) for k, v in dict(data.get("bookmarks", {})).items()}
        return vs


def _load(cls, data: Any, default):
    """Copy known fields with type checks; unknown or malformed values keep their defaults."""
    out = cls(**asdict(default))
    if not isinstance(data, dict):
        return out
    for f in fields(cls):
        if f.name not in data:
            continue
        cur = getattr(out, f.name)
        val = data[f.name]
        try:
            if isinstance(cur, bool):
                val = bool(val)
            elif isinstance(cur, int):
                val = int(val)
            elif isinstance(cur, float):
                val = float(val)
            elif isinstance(cur, str):
                val = str(val)
            elif isinstance(cur, list):
                val = [float(x) for x in val]
                if len(val) != len(cur):
                    continue
            setattr(out, f.name, val)
        except (TypeError, ValueError):
            continue
    return out
