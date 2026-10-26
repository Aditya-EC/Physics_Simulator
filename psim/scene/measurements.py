"""Measurements: live distances and angles between points of scene objects.

A measurement point is an object's origin or centre of mass (world frame, at the current pose — so the
values follow a running simulation). Kinds:

* ``distance`` — two points A, B: |B − A| and the components of B − A
* ``angle``    — three points A, V, B: the angle AVB at the vertex V (0…180°)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .. import numerics

MEASUREMENT_ID = re.compile(r"^MSR-(\d{4,})$")
KINDS = {"distance": 2, "angle": 3}
POINT_AT = ("origin", "com")
KEYS = {"id", "name", "kind", "points", "visible"}


class MeasurementError(ValueError):
    pass


@dataclass
class Measurement:
    id: str
    name: str
    kind: str
    points: list            # [{"object": id, "at": "origin" | "com"}]
    visible: bool = True

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "points": [dict(p) for p in self.points], "visible": self.visible}

    @property
    def objects(self) -> list[str]:
        return [p["object"] for p in self.points]


def validate_measurements(data) -> list[Measurement]:
    if not isinstance(data, list):
        raise MeasurementError("Measurements must be a list")
    out, ids, names = [], set(), set()
    for d in data:
        if not isinstance(d, dict):
            raise MeasurementError("A measurement must be an object")
        unknown = set(d) - KEYS
        if unknown:
            raise MeasurementError(f"Measurement has unknown field(s) {', '.join(sorted(unknown))}")
        mid = str(d.get("id", ""))
        if not MEASUREMENT_ID.match(mid):
            raise MeasurementError(f"Invalid measurement id '{mid}'")
        name = str(d.get("name", "")).strip()
        if not name:
            raise MeasurementError(f"{mid}: a measurement needs a name")
        kind = str(d.get("kind", ""))
        if kind not in KINDS:
            raise MeasurementError(f"{mid}: kind must be distance or angle")
        pts = d.get("points")
        if not isinstance(pts, list) or len(pts) != KINDS[kind]:
            raise MeasurementError(f"{mid}: a {kind} needs {KINDS[kind]} points")
        clean = []
        for p in pts:
            if not isinstance(p, dict) or not isinstance(p.get("object"), str) or not p["object"]:
                raise MeasurementError(f"{mid}: every point needs an object id")
            at = str(p.get("at", "origin"))
            if at not in POINT_AT:
                raise MeasurementError(f"{mid}: a point is an object's origin or com")
            clean.append({"object": p["object"], "at": at})
        if mid in ids:
            raise MeasurementError(f"Duplicate measurement id '{mid}'")
        if name.lower() in names:
            raise MeasurementError(f"Duplicate measurement name '{name}'")
        ids.add(mid)
        names.add(name.lower())
        out.append(Measurement(mid, name, kind, clean, bool(d.get("visible", True))))
    return out


def point_position(scene, point: dict) -> np.ndarray:
    """World position of a measurement point (raises SceneError for a missing object)."""
    oid = point["object"]
    p, q = scene.world_pose(oid)
    p = np.asarray(p, float)
    if point.get("at") == "com":
        obj = scene.get(oid)
        if obj.typedef.physical:
            com = np.asarray(obj.mass_properties()["com"], float)
            return p + np.asarray(numerics.quat_rotate(q, com), float)
    return p


@dataclass
class MeasurementValue:
    value: float            # metres (distance) or radians (angle)
    delta: np.ndarray | None = None   # B − A for distances
    points: list | None = None        # world positions used


def evaluate(scene, m: Measurement) -> MeasurementValue:
    pts = [point_position(scene, p) for p in m.points]
    if m.kind == "distance":
        d = pts[1] - pts[0]
        return MeasurementValue(float(np.linalg.norm(d)), d, pts)
    a, v, b = pts
    u, w = a - v, b - v
    nu, nw = float(np.linalg.norm(u)), float(np.linalg.norm(w))
    if nu < 1e-12 or nw < 1e-12:
        return MeasurementValue(math.nan, None, pts)     # undefined: a point coincides with the vertex
    c = float(np.clip(u @ w / (nu * nw), -1.0, 1.0))
    # atan2 form keeps full precision near 0° and 180°
    angle = math.atan2(float(np.linalg.norm(np.cross(u, w))) / (nu * nw), c)
    return MeasurementValue(angle, None, pts)


def new_measurement(scene, kind: str, object_ids: list[str], at: str = "origin", name: str | None = None) -> dict:
    """Data for a new measurement over the given objects (in order; the vertex of an angle is the second)."""
    if kind not in KINDS:
        raise MeasurementError("kind must be distance or angle")
    if len(object_ids) != KINDS[kind]:
        raise MeasurementError(f"A {kind} needs {KINDS[kind]} objects (selected: {len(object_ids)})")
    if name is None:
        names = {m.name.lower() for m in scene.measurements}
        base = "Distance" if kind == "distance" else "Angle"
        k = 1
        while f"{base} {k}".lower() in names:
            k += 1
        name = f"{base} {k}"
    return {"id": scene.allocate_measurement_id(), "name": name, "kind": kind,
            "points": [{"object": oid, "at": at} for oid in object_ids], "visible": True}
