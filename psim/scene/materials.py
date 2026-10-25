"""Surface (appearance) material presets and physical material densities."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurfaceMaterial:
    name: str
    metallic: float
    roughness: float


SURFACE_MATERIALS: dict[str, SurfaceMaterial] = {m.name: m for m in [
    SurfaceMaterial("Matte Plastic", 0.0, 0.75),
    SurfaceMaterial("Glossy Plastic", 0.0, 0.25),
    SurfaceMaterial("Rubber", 0.0, 0.95),
    SurfaceMaterial("Ceramic", 0.0, 0.15),
    SurfaceMaterial("Brushed Metal", 1.0, 0.45),
    SurfaceMaterial("Polished Metal", 1.0, 0.15),
    SurfaceMaterial("Painted Metal", 0.3, 0.4),
    SurfaceMaterial("Glass", 0.0, 0.05),
    SurfaceMaterial("Wood", 0.0, 0.65),
    SurfaceMaterial("Custom", 0.0, 0.5),
]}

# Physical materials (density in kg/m^3) used to fill in the density field.
PHYSICAL_MATERIALS: dict[str, float] = {
    "Steel": 7850.0,
    "Stainless Steel": 8000.0,
    "Aluminium": 2700.0,
    "Copper": 8960.0,
    "Brass": 8500.0,
    "Titanium": 4506.0,
    "Lead": 11340.0,
    "Concrete": 2400.0,
    "Glass": 2500.0,
    "Water": 1000.0,
    "Ice": 917.0,
    "Oak": 750.0,
    "Pine": 500.0,
    "ABS Plastic": 1050.0,
    "Rubber": 1100.0,
    "Foam": 30.0,
}

PALETTE = [
    (0.36, 0.58, 0.86), (0.89, 0.53, 0.29), (0.47, 0.72, 0.45), (0.84, 0.38, 0.42),
    (0.62, 0.50, 0.82), (0.86, 0.75, 0.36), (0.36, 0.74, 0.76), (0.72, 0.72, 0.74),
]
