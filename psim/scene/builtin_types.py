"""Built-in object type definitions."""
from __future__ import annotations

import hashlib

import numpy as np

from ..properties.property_system import PropertyDef, ValueType
from . import geometry as g
from .object_registry import ObjectRegistry, ObjectTypeDef


def L(key: str, label: str, default: float, minimum: float = 1e-6, description: str = "") -> PropertyDef:
    return PropertyDef(key, label, ValueType.FLOAT, default, quantity="length", minimum=minimum, maximum=1e6,
                       step=0.1, description=description)


def RES(default: int = 48, minimum: int = 8, maximum: int = 192) -> PropertyDef:
    return PropertyDef("segments", "Resolution", ValueType.INT, default, minimum=minimum, maximum=maximum,
                       description="Number of facets around curved surfaces (affects display and computed "
                                   "volume/inertia, which are integrated from the faceted mesh)")


def _diameter(p: dict) -> dict:
    return {"diameter": ("Diameter", 2 * p["radius"], "length")}


# ---- user mesh assets (content addressed) ------------------------------------

class MeshAssetStore:
    """Imported meshes, keyed by a content hash so identical imports share storage."""

    def __init__(self) -> None:
        self._meshes: dict[str, g.Mesh] = {}
        self._names: dict[str, str] = {}

    def add(self, mesh: g.Mesh, name: str) -> str:
        h = hashlib.sha256(mesh.vertices.tobytes() + mesh.indices.tobytes()).hexdigest()[:16]
        key = f"mesh-{h}"
        self._meshes.setdefault(key, mesh)
        self._names.setdefault(key, name)
        return key

    def get(self, key: str) -> g.Mesh | None:
        return self._meshes.get(key)

    def name(self, key: str) -> str:
        return self._names.get(key, key)

    def keys(self) -> list[str]:
        return list(self._meshes)

    def clear(self) -> None:
        self._meshes.clear()
        self._names.clear()

    def to_dict(self, keys: set[str]) -> dict:
        out = {}
        for k in sorted(keys):
            m = self._meshes.get(k)
            if m is not None:
                out[k] = {"name": self._names.get(k, k), "vertices": m.vertices.round(7).tolist(),
                          "indices": m.indices.tolist()}
        return out

    def load_dict(self, data: dict) -> list[str]:
        warnings = []
        for k, d in data.items():
            try:
                v = np.asarray(d["vertices"], np.float32).reshape(-1, 3)
                f = np.asarray(d["indices"], np.int64).reshape(-1, 3)
                if len(v) == 0 or len(f) == 0 or f.min() < 0 or f.max() >= len(v):
                    raise ValueError("invalid mesh arrays")
                m = g.Mesh(v, g.numerics.compute_vertex_normals(v, f.astype(np.uint32)), f)
                m.solid = g._is_closed(m.vertices, m.indices) and \
                    g.numerics.mass_properties_solid(m.vertices, m.indices, 1.0)["measure"] > 0
                self._meshes[k] = m
                self._names[k] = str(d.get("name", k))
            except (KeyError, TypeError, ValueError) as exc:
                warnings.append(f"Mesh asset '{k}' could not be loaded: {exc}")
        return warnings


mesh_assets = MeshAssetStore()
_MISSING_MESH = g.box(0.5, 0.5, 0.5)


def _user_mesh(p: dict) -> g.Mesh:
    m = mesh_assets.get(p["asset"])
    return m if m is not None else _MISSING_MESH


# ---- definitions -----------------------------------------------------------------

def _pulley(p: dict) -> g.Mesh:
    r, w, d, n = p["radius"], p["width"], p["groove_depth"], p["segments"]
    h = w / 2
    gr = max(r - d, 1e-4)
    groove_pts = np.array([(r, -h * 0.55), (gr, -h * 0.15), (gr, h * 0.15), (r, h * 0.55)])
    secs = [g._flat_section((0, -h), (r, -h)), g._flat_section((r, -h), (r, -h * 0.55))]
    for a, b in zip(groove_pts[:-1], groove_pts[1:]):
        secs.append(g._flat_section(a, b))
    secs += [g._flat_section((r, h * 0.55), (r, h)), g._flat_section((r, h), (0, h))]
    return g.lathe(secs, n)


def register_all(reg: ObjectRegistry) -> None:
    B = "Basic Solids"
    defs = [
        ObjectTypeDef("basic.cube", "Cube", B, "Cube with equal edge lengths.",
                      [L("size", "Edge length", 1.0)], lambda p: g.box(p["size"], p["size"], p["size"])),
        ObjectTypeDef("basic.cuboid", "Rectangular Cuboid", B, "Box with independent length, width and height.",
                      [L("length", "Length (X)", 1.6), L("width", "Width (Y)", 1.0), L("height", "Height (Z)", 0.6)],
                      lambda p: g.box(p["length"], p["width"], p["height"])),
        ObjectTypeDef("basic.sphere", "Sphere", B, "Sphere centred on its local origin.",
                      [L("radius", "Radius", 0.5), RES()], lambda p: g.sphere(p["radius"], p["segments"], p["segments"] // 2),
                      derived=_diameter),
        ObjectTypeDef("basic.ellipsoid", "Ellipsoid", B, "Ellipsoid with semi-axes along local X, Y and Z.",
                      [L("rx", "Semi-axis X", 0.8), L("ry", "Semi-axis Y", 0.5), L("rz", "Semi-axis Z", 0.35), RES()],
                      lambda p: g.ellipsoid(p["rx"], p["ry"], p["rz"], p["segments"], p["segments"] // 2)),
        ObjectTypeDef("basic.cylinder", "Cylinder", B, "Right circular cylinder along local Z.",
                      [L("radius", "Radius", 0.4), L("height", "Height", 1.0), RES()],
                      lambda p: g.cylinder(p["radius"], p["height"], p["segments"]), derived=_diameter),
        ObjectTypeDef("basic.cone", "Cone", B, "Right circular cone along local Z (apex at +Z).",
                      [L("radius", "Base radius", 0.5), L("height", "Height", 1.0), RES()],
                      lambda p: g.cone(p["radius"], p["height"], p["segments"]), derived=_diameter),
        ObjectTypeDef("basic.frustum", "Frustum", B, "Truncated cone along local Z.",
                      [L("bottom_radius", "Bottom radius", 0.5), L("top_radius", "Top radius", 0.25, minimum=0.0),
                       L("height", "Height", 0.8), RES()],
                      lambda p: g.frustum(p["bottom_radius"], p["top_radius"], p["height"], p["segments"])),
        ObjectTypeDef("basic.torus", "Torus", B, "Ring torus in the local XY plane.",
                      [L("major_radius", "Major radius", 0.5), L("minor_radius", "Tube radius", 0.15), RES(64)],
                      lambda p: g.torus(p["major_radius"], p["minor_radius"], p["segments"], max(12, p["segments"] // 2)),
                      validate=lambda p: [] if p["minor_radius"] < p["major_radius"] else
                      ["Tube radius must be smaller than the major radius"]),
        ObjectTypeDef("basic.plane", "Plane", B, "Zero-thickness rectangular surface (thin shell) facing +Z.",
                      [L("length", "Length (X)", 2.0), L("width", "Width (Y)", 2.0)],
                      lambda p: g.plane(p["length"], p["width"]), mass_model="shell", default_material="Matte Plastic",
                      default_density=5.0, thumbnail_view=(35.0, 40.0)),
        ObjectTypeDef("basic.disk", "Disk", B, "Zero-thickness circular surface (thin shell) facing +Z.",
                      [L("radius", "Radius", 0.6), RES()], lambda p: g.disk(p["radius"], p["segments"]),
                      mass_model="shell", default_density=5.0, derived=_diameter, thumbnail_view=(35.0, 40.0)),
        ObjectTypeDef("basic.capsule", "Capsule", B, "Cylinder with hemispherical caps along local Z.",
                      [L("radius", "Radius", 0.3), L("length", "Cylinder length", 0.8, minimum=0.0), RES()],
                      lambda p: g.capsule(p["radius"], p["length"], p["segments"], max(4, p["segments"] // 4)),
                      derived=lambda p: {"total_length": ("Total length", p["length"] + 2 * p["radius"], "length")}),
        ObjectTypeDef("basic.pyramid", "Pyramid", B, "Rectangular-base pyramid along local Z.",
                      [L("base_length", "Base length (X)", 1.0), L("base_width", "Base width (Y)", 1.0),
                       L("height", "Height", 1.0)],
                      lambda p: g.pyramid(p["base_length"], p["base_width"], p["height"])),
        ObjectTypeDef("basic.prism", "Prism", B, "Right prism with a regular polygonal cross-section.",
                      [PropertyDef("sides", "Sides", ValueType.INT, 6, minimum=3, maximum=64),
                       L("circumradius", "Circumradius", 0.5), L("height", "Height", 1.0)],
                      lambda p: g.prism(p["sides"], p["circumradius"], p["height"])),
        ObjectTypeDef("basic.hemisphere", "Hemisphere", B, "Half sphere with a flat base (base towards -Z).",
                      [L("radius", "Radius", 0.5), RES()],
                      lambda p: g.hemisphere(p["radius"], p["segments"], max(4, p["segments"] // 4)), derived=_diameter),
        # ---- rigid bodies & point mass ---------------------------------------
        ObjectTypeDef("body.point_mass", "Point Mass", "Rigid Bodies",
                      "Mass concentrated at a single point. The sphere is a display marker only.",
                      [L("display_radius", "Display radius", 0.12)],
                      lambda p: g.sphere(p["display_radius"], 32, 16), mass_model="point",
                      default_color=(0.95, 0.72, 0.25), default_material="Glossy Plastic", default_mass=1.0),
        ObjectTypeDef("body.rigid_box", "Rigid Body (Box)", "Rigid Bodies",
                      "Box rigid body: collides (automatic box collider) and can be connected by joints.",
                      [L("length", "Length (X)", 1.0), L("width", "Width (Y)", 0.6), L("height", "Height (Z)", 0.4)],
                      lambda p: g.box(p["length"], p["width"], p["height"]), rigid_body=True,
                      default_material="Painted Metal", default_density=7850.0, default_color=(0.55, 0.60, 0.68)),
        ObjectTypeDef("body.rigid_cylinder", "Rigid Body (Cylinder)", "Rigid Bodies",
                      "Cylinder rigid body: collides (bounding-box collider unless changed) and can be connected by joints.",
                      [L("radius", "Radius", 0.3), L("height", "Height", 0.8), RES()],
                      lambda p: g.cylinder(p["radius"], p["height"], p["segments"]), rigid_body=True,
                      default_material="Brushed Metal", default_density=2700.0, default_color=(0.70, 0.72, 0.76)),
        # ---- reference frames -------------------------------------------------
        ObjectTypeDef("frame.reference", "Reference Frame", "Reference Frames",
                      "Named coordinate frame (X red, Y green, Z blue). Has no mass.",
                      [L("axis_length", "Axis length", 0.8)],
                      lambda p: g.axis_triad(p["axis_length"], p["axis_length"] * 0.025), mass_model="none",
                      uses_vertex_colors=True, axes_scale=0.0, default_material="Glossy Plastic",
                      overlays=("name", "id", "position", "orientation", "custom")),
        ObjectTypeDef("frame.marker", "Coordinate Marker", "Reference Frames",
                      "Small marker identifying a point of interest. Has no mass.",
                      [L("size", "Size", 0.3)], lambda p: g.coordinate_marker(p["size"]), mass_model="none",
                      uses_vertex_colors=True, axes_scale=0.0, default_material="Glossy Plastic",
                      overlays=("name", "id", "position", "custom")),
        ObjectTypeDef("group.assembly", "Assembly", "Composite Shapes",
                      "Empty node that groups child objects. Children move with the assembly.",
                      [L("marker_size", "Marker size", 0.3)],
                      lambda p: g.axis_triad(p["marker_size"], p["marker_size"] * 0.04), mass_model="none",
                      uses_vertex_colors=True, axes_scale=0.0, default_material="Glossy Plastic",
                      overlays=("name", "id", "position", "orientation", "custom")),
        # ---- vectors & measurement ----------------------------------------------
        ObjectTypeDef("measure.vector_arrow", "Vector Arrow", "Vectors & Measurement",
                      "Arrow along local +X starting at the local origin. Has no mass.",
                      [L("length", "Length", 1.0), L("shaft_radius", "Shaft radius", 0.025)],
                      lambda p: g.arrow(p["length"], p["shaft_radius"], p["shaft_radius"] * 2.6, min(p["length"] * 0.25,
                                        p["shaft_radius"] * 8)),
                      mass_model="none", default_color=(0.95, 0.60, 0.20), default_material="Glossy Plastic",
                      overlays=("name", "id", "position", "orientation", "custom", "local_axes")),
        ObjectTypeDef("measure.line_segment", "Line Segment", "Vectors & Measurement",
                      "Straight segment along local X with end markers. Has no mass.",
                      [L("length", "Length", 1.0), L("radius", "Line radius", 0.012)],
                      lambda p: g.line_segment(p["length"], p["radius"]), mass_model="none",
                      default_color=(0.85, 0.85, 0.90), default_material="Glossy Plastic",
                      overlays=("name", "id", "position", "orientation", "custom", "local_axes")),
        # ---- mechanical components ----------------------------------------------
        ObjectTypeDef("mech.spring", "Spring", "Mechanical Components",
                      "Helical coil spring along local Z (geometry only; spring force is a Phase 4 process).",
                      [L("coil_radius", "Coil radius", 0.2), L("wire_radius", "Wire radius", 0.025),
                       L("length", "Free length", 1.0), PropertyDef("turns", "Turns", ValueType.FLOAT, 8.0,
                                                                    minimum=0.5, maximum=200)],
                      lambda p: g.spring(p["coil_radius"], p["wire_radius"], p["length"], p["turns"]),
                      default_material="Polished Metal", default_density=7850.0, default_color=(0.75, 0.76, 0.80),
                      validate=lambda p: [] if p["wire_radius"] < p["coil_radius"] else
                      ["Wire radius must be smaller than the coil radius"]),
        ObjectTypeDef("mech.rope", "Rope / Cable", "Mechanical Components",
                      "Flexible cable between two end points with a static sag (geometry only).",
                      [L("length", "Span", 2.0), L("radius", "Radius", 0.02), L("sag", "Sag", 0.3, minimum=0.0)],
                      lambda p: g.rope(p["length"], p["radius"], p["sag"]), default_material="Rubber",
                      default_density=1100.0, default_color=(0.55, 0.45, 0.35)),
        ObjectTypeDef("mech.gear", "Spur Gear", "Mechanical Components", "Spur gear with trapezoidal teeth.",
                      [PropertyDef("teeth", "Teeth", ValueType.INT, 16, minimum=4, maximum=200),
                       L("outer_radius", "Outer radius", 0.5), L("root_radius", "Root radius", 0.42),
                       L("thickness", "Thickness", 0.15)],
                      lambda p: g.extrude(g.gear_profile(p["teeth"], p["outer_radius"], p["root_radius"]), p["thickness"]),
                      default_material="Brushed Metal", default_density=7850.0, default_color=(0.68, 0.70, 0.74),
                      validate=lambda p: [] if p["root_radius"] < p["outer_radius"] else
                      ["Root radius must be smaller than the outer radius"], thumbnail_view=(30.0, 45.0)),
        ObjectTypeDef("mech.i_beam", "I-Beam", "Mechanical Components", "Structural I-beam along local Z.",
                      [L("width", "Flange width", 0.3), L("height", "Section height", 0.4),
                       L("web", "Web thickness", 0.03), L("flange", "Flange thickness", 0.04),
                       L("length", "Length", 2.0)],
                      lambda p: g.extrude(g.i_beam_profile(p["width"], p["height"], p["web"], p["flange"]), p["length"]),
                      default_material="Painted Metal", default_density=7850.0, default_color=(0.80, 0.45, 0.25),
                      validate=lambda p: (["Web must be thinner than the flange width"] if p["web"] >= p["width"] else []) +
                      (["Flanges must be thinner than half the section height"] if 2 * p["flange"] >= p["height"] else [])),
        ObjectTypeDef("mech.pulley", "Pulley", "Mechanical Components", "Grooved pulley wheel about local Z.",
                      [L("radius", "Radius", 0.4), L("width", "Width", 0.15), L("groove_depth", "Groove depth", 0.06),
                       RES(64)],
                      _pulley, default_material="Brushed Metal", default_density=2700.0,
                      default_color=(0.66, 0.68, 0.72), derived=_diameter,
                      validate=lambda p: [] if p["groove_depth"] < p["radius"] else
                      ["Groove depth must be smaller than the radius"]),
        # ---- user-defined meshes ----------------------------------------------------
        ObjectTypeDef("user.mesh", "Custom Polygonal Mesh", "User-Defined Objects",
                      "Polygonal mesh imported from a Wavefront OBJ file.",
                      [PropertyDef("asset", "Mesh asset", ValueType.STRING, "", read_only=True)],
                      _user_mesh, user_creatable=False),
    ]
    for d in defs:
        reg.register(d)
