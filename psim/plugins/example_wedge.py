"""Example plugin: registers a 'Wedge' (triangular ramp) object type.

Demonstrates the object extension interface: parameters, mesh generator,
validation and defaults, with no changes to the GUI or the scene code.
"""
from __future__ import annotations

import numpy as np

from ..properties.property_system import PropertyDef, ValueType
from ..scene import geometry as g
from ..scene.object_registry import ObjectTypeDef


def _wedge(p: dict) -> g.Mesh:
    lx, ly, h = p["length"], p["width"], p["height"]
    # Right-triangle profile in the XZ plane, extruded along Y.
    profile = np.array([(-lx / 2, -h / 2), (lx / 2, -h / 2), (-lx / 2, h / 2)])
    m = g.extrude(profile, ly)  # extruded along Z: profile (x, z') -> rotate so extrusion is along Y
    rot = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    return m.transformed(rot)


def register(registry) -> None:
    registry.register(ObjectTypeDef(
        "plugin.wedge", "Wedge (Ramp)", "Basic Solids",
        "Right-angled wedge, useful as an inclined plane. Provided by the example plugin.",
        [PropertyDef("length", "Length (X)", ValueType.FLOAT, 1.2, quantity="length", minimum=1e-6),
         PropertyDef("width", "Width (Y)", ValueType.FLOAT, 0.8, quantity="length", minimum=1e-6),
         PropertyDef("height", "Height (Z)", ValueType.FLOAT, 0.6, quantity="length", minimum=1e-6)],
        _wedge,
        derived=lambda p: {"incline": ("Incline angle", float(np.arctan2(p["height"], p["length"])), "angle")},
    ))
