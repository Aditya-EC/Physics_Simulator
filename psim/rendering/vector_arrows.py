"""Vector arrows drawn from an object's centre of mass (velocity, acceleration, force, angular velocity)."""
from __future__ import annotations

import numpy as np

from ..scene.object_base import SceneObject
from ..scene.scene_manager import Scene

ARROW_COLORS = {"velocity": (0.35, 0.85, 0.45), "acceleration": (1.0, 0.62, 0.2), "force": (0.95, 0.3, 0.3),
                "angular_velocity": (0.72, 0.5, 1.0)}


def arrow_vectors(scene: Scene, obj: SceneObject) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(quantity, start point, vector in world units) for every enabled arrow overlay of the object.

    Acceleration is the current total acceleration while simulating (the input property otherwise); the
    force arrow shows the net force m·a while simulating and the applied force property otherwise.
    """
    ov = obj.overlays
    if not any(ov.get(f"{q}_arrow") for q in ARROW_COLORS):
        return []
    m = scene.world_matrix(obj.id)
    com = (m @ np.append(obj.mass_properties()["com"] if obj.typedef.physical else np.zeros(3), 1.0))[:3]
    runtime = scene.runtime.get(obj.id, {})
    acc = runtime.get("acceleration")
    out = []
    for q in ARROW_COLORS:
        if not ov.get(f"{q}_arrow") or f"{q}_arrow" not in obj.typedef.overlays:
            continue
        if q == "velocity":
            v = np.asarray(obj.linear_velocity, float)
        elif q == "acceleration":
            v = np.asarray(acc if acc is not None else obj.linear_acceleration, float)
        elif q == "force":
            mass = obj.mass() or 0.0
            v = np.asarray(acc, float) * mass if acc is not None and mass > 0 else np.asarray(obj.force, float)
        else:
            v = np.asarray(obj.angular_velocity, float)
        vec = v * obj.arrow_scale(q)
        if np.all(np.isfinite(vec)) and np.linalg.norm(vec) > 1e-9:
            out.append((q, com, vec))
    return out


def arrow_lines(start: np.ndarray, vec: np.ndarray, view_dir: np.ndarray, color) -> tuple[np.ndarray, np.ndarray]:
    """Shaft and a two-line head in the plane facing the camera."""
    end = start + vec
    length = float(np.linalg.norm(vec))
    d = vec / length
    side = np.cross(d, view_dir)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(d, [0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0.0, 0.0])
    side = side / np.linalg.norm(side)
    head = min(0.25 * length, 0.15 + 0.05 * length)
    pts = [start, end, end, end - d * head + side * head * 0.45, end, end - d * head - side * head * 0.45]
    return np.array(pts), np.tile([*color, 1.0], (len(pts), 1))
