"""Named, user-defined reference frames.

A frame is either fixed in the global frame (``parent`` is None) or attached to
an object or assembly. An attached frame's origin is a point fixed on the
parent at ``position`` (parent axes); its axes either rotate with the parent
(``follow_rotation``) or keep the fixed global orientation ``orientation``.
Frames carry no physics of their own: their motion is derived from the parent.

Frames are identified by persistent ids ``FRM-0001``…, never reused within a
project, and referenced by that id from the inspector, formulas
(``frame("FRM-0001")``) and the relative-motion panel.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

FRAME_ID = re.compile(r"^FRM-(\d{4,})$")
FRAME_KEYS = {"id", "name", "parent", "position", "orientation", "follow_rotation", "visible", "show_label",
              "axis_length", "color", "description"}


class FrameError(ValueError):
    pass


def _vec(value, n: int, what: str) -> list[float]:
    try:
        out = [float(x) for x in value]
    except (TypeError, ValueError):
        raise FrameError(f"{what} must be a list of {n} numbers") from None
    if len(out) != n or not all(math.isfinite(x) for x in out):
        raise FrameError(f"{what} must be a list of {n} finite numbers")
    return out


@dataclass
class ReferenceFrame:
    id: str
    name: str
    parent: str | None = None
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    follow_rotation: bool = True
    visible: bool = True
    show_label: bool = True
    axis_length: float = 0.6
    color: list[float] = field(default_factory=lambda: [0.98, 0.78, 0.32])
    description: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "parent": self.parent, "position": list(self.position),
                "orientation": list(self.orientation), "follow_rotation": self.follow_rotation,
                "visible": self.visible, "show_label": self.show_label, "axis_length": self.axis_length,
                "color": list(self.color), "description": self.description}

    @staticmethod
    def from_dict(d: dict) -> "ReferenceFrame":
        if not isinstance(d, dict):
            raise FrameError("Reference frame entry must be an object")
        unknown = set(d) - FRAME_KEYS
        if unknown:
            raise FrameError(f"Reference frame has unknown field(s) {', '.join(sorted(unknown))}")
        fid = str(d.get("id", ""))
        if not FRAME_ID.match(fid):
            raise FrameError(f"Invalid reference frame id '{fid}'")
        name = str(d.get("name", "")).strip()
        if not name:
            raise FrameError(f"{fid}: a reference frame needs a name")
        parent = d.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise FrameError(f"{fid}: parent must be an object id or null")
        q = np.asarray(_vec(d.get("orientation", [1, 0, 0, 0]), 4, f"{fid}: orientation"))
        n = float(np.linalg.norm(q))
        if n < 1e-9:
            raise FrameError(f"{fid}: orientation quaternion must be non-zero")
        length = float(d.get("axis_length", 0.6))
        if not (math.isfinite(length) and length > 0):
            raise FrameError(f"{fid}: axis length must be positive")
        color = _vec(d.get("color", [0.98, 0.78, 0.32]), 3, f"{fid}: color")
        return ReferenceFrame(fid, name, parent or None, _vec(d.get("position", [0, 0, 0]), 3, f"{fid}: position"),
                              [float(x) for x in q / n], bool(d.get("follow_rotation", True)),
                              bool(d.get("visible", True)), bool(d.get("show_label", True)), length,
                              [min(max(c, 0.0), 1.0) for c in color], str(d.get("description", "")))


def validate_frames(data: list) -> list[ReferenceFrame]:
    if not isinstance(data, list):
        raise FrameError("Reference frames must be a list")
    frames = [ReferenceFrame.from_dict(d) for d in data]
    ids, names = set(), set()
    for f in frames:
        if f.id in ids:
            raise FrameError(f"Duplicate reference frame id '{f.id}'")
        if f.name.lower() in names:
            raise FrameError(f"Duplicate reference frame name '{f.name}'")
        ids.add(f.id)
        names.add(f.name.lower())
    return frames


def is_frame_id(ref: str | None) -> bool:
    return bool(ref) and bool(FRAME_ID.match(ref))
