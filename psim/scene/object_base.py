"""SceneObject: identity, transform, appearance, physical state and custom data.

Values are addressed by *property paths* so the inspector, undo system,
multi-object editing and (later) the process engine use a single access API:

    "name", "position", "color", "density", ...   standard properties
    "params.<key>"                                  geometry parameters
    "custom.<key>"                                  custom properties
    "overlay.<key>"                                 overlay visibility flags
    "trajectory.<key>"                              trajectory marker settings (TRAJECTORY_PROPERTIES)
    "label_unit.<quantity>"                         display unit of that quantity in this object's label
    "arrow_scale.<quantity>"                        length of vector arrows per SI unit (ARROW_QUANTITIES)
    "physics.<key>"                                 collision geometry and contact material (PHYSICS_PROPERTIES)

Transform convention: `position` / `orientation` are expressed in the parent's
frame (the global frame for root objects). Scale is a per-object geometry
scale and is NOT inherited by children, so every object frame is rigid.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import numerics
from ..properties.custom_properties import CustomProperty
from ..properties.property_system import Origin, PropertyDef, PropertyError, ValueType
from .geometry import Mesh
from .materials import SURFACE_MATERIALS
from .object_registry import ALL_OVERLAYS, ObjectTypeDef, registry

P = PropertyDef
V3 = ValueType.VEC3

STANDARD_PROPERTIES: dict[str, PropertyDef] = {p.key: p for p in [
    # identity
    P("name", "Name", ValueType.STRING, "Object", pattern=r".*\S.*", description="Display name (need not be unique)"),
    P("description", "Description", ValueType.STRING, ""),
    P("visible", "Visible", ValueType.BOOL, True),
    P("locked", "Locked", ValueType.BOOL, False, description="Locked objects cannot be moved or edited in the viewport"),
    P("tags", "Tags", ValueType.STRING, "", description="Comma-separated tags"),
    # transform
    P("position", "Position", V3, [0, 0, 0], quantity="length", minimum=-1e7, maximum=1e7, step=0.1),
    P("orientation_euler", "Orientation (Euler XYZ)", V3, [0, 0, 0], quantity="angle", step=0.0872664626,
      description="Extrinsic X-then-Y-then-Z rotation (R = Rz·Ry·Rx)"),
    P("scale", "Scale", V3, [1, 1, 1], minimum=1e-4, maximum=1e4, step=0.1, precision=4,
      description="Geometry scale factors along local axes (not inherited by children)"),
    # appearance
    P("color", "Color", ValueType.COLOR, [0.6, 0.6, 0.6]),
    P("opacity", "Opacity", ValueType.FLOAT, 1.0, minimum=0.05, maximum=1.0, step=0.05, precision=2),
    P("material", "Material", ValueType.ENUM, "Matte Plastic", choices=tuple(SURFACE_MATERIALS)),
    P("metallic", "Metallic", ValueType.FLOAT, 0.0, minimum=0.0, maximum=1.0, step=0.05, precision=2),
    P("roughness", "Roughness", ValueType.FLOAT, 0.5, minimum=0.02, maximum=1.0, step=0.05, precision=2),
    P("wireframe", "Wireframe", ValueType.BOOL, False),
    P("show_surface", "Surface visible", ValueType.BOOL, True),
    P("show_edges", "Edges visible", ValueType.BOOL, False),
    P("label_color", "Label color", ValueType.COLOR, [0.92, 0.93, 0.95]),
    # physical (initial conditions are user-controlled; the simulation will update them later)
    P("mass_mode", "Mass from", ValueType.ENUM, "density", choices=("density", "mass"),
      description="Compute mass from density × volume, or specify mass and derive density"),
    P("density", "Density", ValueType.FLOAT, 1000.0, quantity="density", minimum=1e-9, maximum=1e9, precision=4),
    P("mass_input", "Mass", ValueType.FLOAT, 1.0, quantity="mass", minimum=1e-12, maximum=1e15, precision=6),
    P("linear_velocity", "Linear velocity", V3, [0, 0, 0], quantity="velocity", minimum=-1e9, maximum=1e9),
    P("linear_acceleration", "Linear acceleration", V3, [0, 0, 0], quantity="acceleration", minimum=-1e9, maximum=1e9),
    P("angular_velocity", "Angular velocity", V3, [0, 0, 0], quantity="angular_velocity", minimum=-1e9, maximum=1e9),
    P("angular_acceleration", "Angular acceleration", V3, [0, 0, 0], quantity="angular_acceleration",
      minimum=-1e9, maximum=1e9),
    P("force", "Applied force", V3, [0, 0, 0], quantity="force", minimum=-1e12, maximum=1e12),
    P("torque", "Applied torque", V3, [0, 0, 0], quantity="torque", minimum=-1e12, maximum=1e12),
    # display
    P("local_axes_size", "Local axes size", ValueType.FLOAT, 0.0, quantity="length", minimum=0.0, maximum=1e6,
      description="0 = automatic (based on object size)"),
    P("label_font_size", "Label font size", ValueType.INT, 9, minimum=6, maximum=48),
    P("label_precision", "Label precision", ValueType.INT, 3, minimum=0, maximum=10),
    P("label_background", "Label background", ValueType.BOOL, True),
    P("label_placement", "Label placement", ValueType.ENUM, "above", choices=("above", "below", "left", "right"),
      description="Where the information label is drawn relative to the object"),
]}

TRAJECTORY_PROPERTIES: dict[str, PropertyDef] = {p.key: p for p in [
    P("enabled", "Show trajectory", ValueType.BOOL, False),
    P("color", "Path colour", ValueType.COLOR, [1.0, 0.72, 0.25]),
    P("width", "Line thickness", ValueType.FLOAT, 2.0, minimum=0.5, maximum=12.0, step=0.5, precision=1),
    P("opacity", "Opacity", ValueType.FLOAT, 0.9, minimum=0.05, maximum=1.0, step=0.05, precision=2),
    P("interval", "Sampling interval", ValueType.FLOAT, 0.0, quantity="time", minimum=0.0, maximum=1e6,
      description="Simulated time between path points; 0 = every recorded sample (the recording interval is "
                  "the finest possible spacing)"),
    P("max_points", "Maximum points", ValueType.INT, 5000, minimum=10, maximum=1_000_000,
      description="Complete history is thinned evenly to this many points; recent history keeps the newest"),
    P("history", "History", ValueType.ENUM, "complete", choices=("complete", "recent")),
    P("recent_duration", "Recent history length", ValueType.FLOAT, 5.0, quantity="time", minimum=1e-6, maximum=1e9),
    P("point", "Tracked point", ValueType.ENUM, "com", choices=("com", "origin"),
      description="Centre of mass or object origin"),
    P("frame", "Recorded in frame", ValueType.STRING, "",
      description="Empty = global. Relative trajectories are drawn attached to the frame's current pose"),
    P("arrows", "Direction arrows", ValueType.BOOL, False),
    P("arrow_spacing", "Arrow spacing", ValueType.FLOAT, 1.0, quantity="time", minimum=1e-6, maximum=1e9),
    P("time_labels", "Time labels", ValueType.BOOL, False),
    P("label_spacing", "Time label spacing", ValueType.FLOAT, 1.0, quantity="time", minimum=1e-6, maximum=1e9),
    P("preserve", "Keep after stopping", ValueType.BOOL, True,
      description="Keep showing the path of the last run after the simulation stops"),
]}

PHYSICS_PROPERTIES: dict[str, PropertyDef] = {p.key: p for p in [
    P("collider", "Collision shape", ValueType.ENUM, "none",
      choices=("none", "auto", "sphere", "box", "capsule", "plane", "convex"),
      description="Collision geometry (separate from the visual mesh). 'auto' uses the exact shape where the "
                  "type has one, else the convex hull of the mesh. Rigid-body types default to 'auto'"),
    P("radius", "Collider radius", ValueType.FLOAT, 0.0, quantity="length", minimum=0.0, maximum=1e6,
      description="0 = fitted to the mesh"),
    P("length", "Capsule core length", ValueType.FLOAT, 0.0, quantity="length", minimum=0.0, maximum=1e6,
      description="0 = fitted to the mesh"),
    P("half_extents", "Box half extents", V3, [0.0, 0.0, 0.0], quantity="length", minimum=0.0, maximum=1e6,
      description="0 = fitted to the mesh (per axis)"),
    P("offset", "Collider offset", V3, [0.0, 0.0, 0.0], quantity="length", minimum=-1e6, maximum=1e6,
      description="Shift of the collider centre in the object's local frame"),
    P("restitution", "Restitution", ValueType.FLOAT, 0.3, minimum=0.0, maximum=1.0, step=0.05, precision=3,
      description="Bounciness of contacts (the larger of the two bodies' values is used)"),
    P("friction", "Friction coefficient", ValueType.FLOAT, 0.5, minimum=0.0, maximum=10.0, step=0.05, precision=3,
      description="Coulomb friction coefficient (geometric mean of the two bodies' values)"),
    P("show_collider", "Show collider", ValueType.BOOL, False, description="Draw the collision shape in the viewport"),
]}

# vector arrows: quantity -> (label, length per SI unit, quantity of the scale)
ARROW_QUANTITIES = {"velocity": ("Velocity arrow scale", 0.25), "acceleration": ("Acceleration arrow scale", 0.05),
                    "force": ("Force arrow scale", 0.02), "angular_velocity": ("Angular velocity arrow scale", 0.25)}

# Physical quantities whose values the simulation will later own (badge "S" at runtime).
DYNAMIC_STATE = ("position", "orientation_euler", "linear_velocity", "linear_acceleration", "angular_velocity",
                 "angular_acceleration")

OVERLAY_LABELS = {
    "name": "Name", "id": "Unique ID", "position": "Position", "orientation": "Orientation",
    "velocity": "Velocity", "acceleration": "Acceleration", "angular_velocity": "Angular velocity",
    "mass": "Mass", "force": "Force", "torque": "Torque", "custom": "Custom properties",
    "local_axes": "Local axes", "bbox": "Bounding box", "com": "Center of mass",
    "angular_acceleration": "Angular acceleration", "velocity_arrow": "Velocity arrow",
    "acceleration_arrow": "Acceleration arrow", "force_arrow": "Force arrow (net while simulating)",
    "angular_velocity_arrow": "Angular velocity arrow",
}

_SIMPLE_ATTRS = [k for k in STANDARD_PROPERTIES if k not in ("orientation_euler", "tags")]


@dataclass
class SceneObject:
    id: str
    type_id: str
    name: str = "Object"
    description: str = ""
    visible: bool = True
    locked: bool = False
    parent_id: str | None = None
    tags: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    color: list[float] = field(default_factory=lambda: [0.6, 0.6, 0.6])
    opacity: float = 1.0
    material: str = "Matte Plastic"
    metallic: float = 0.0
    roughness: float = 0.75
    wireframe: bool = False
    show_surface: bool = True
    show_edges: bool = False
    label_color: list[float] = field(default_factory=lambda: [0.92, 0.93, 0.95])
    mass_mode: str = "density"
    density: float = 1000.0
    mass_input: float = 1.0
    linear_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    linear_acceleration: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_acceleration: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    force: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    torque: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    local_axes_size: float = 0.0
    label_font_size: int = 9
    label_precision: int = 3
    label_background: bool = True
    label_placement: str = "above"
    overlays: dict[str, bool] = field(default_factory=lambda: {"name": False})
    trajectory: dict[str, Any] = field(default_factory=dict)     # only non-default settings are stored
    label_units: dict[str, str] = field(default_factory=dict)    # quantity -> unit symbol
    arrow_scales: dict[str, float] = field(default_factory=dict)  # quantity -> length per SI unit
    physics: dict[str, Any] = field(default_factory=dict)        # only non-default settings are stored
    custom: dict[str, CustomProperty] = field(default_factory=dict)

    _mesh_cache: tuple[tuple, Mesh] | None = field(default=None, repr=False, compare=False)
    _mass_cache: tuple[tuple, dict] | None = field(default=None, repr=False, compare=False)

    # ---- type information ------------------------------------------------
    @property
    def typedef(self) -> ObjectTypeDef:
        return registry().get(self.type_id)

    @staticmethod
    def new(oid: str, typedef: ObjectTypeDef, name: str | None = None, params: dict | None = None) -> "SceneObject":
        obj = SceneObject(id=oid, type_id=typedef.type_id, name=name or typedef.name)
        obj.params = typedef.coerce_params(params or {})
        mat = SURFACE_MATERIALS.get(typedef.default_material, SURFACE_MATERIALS["Matte Plastic"])
        obj.material, obj.metallic, obj.roughness = mat.name, mat.metallic, mat.roughness
        if typedef.default_color is not None:
            obj.color = list(typedef.default_color)
        obj.density = typedef.default_density
        obj.mass_input = typedef.default_mass
        if typedef.mass_model == "point":
            obj.mass_mode = "mass"
        return obj

    # ---- geometry ----------------------------------------------------------
    def mesh(self) -> Mesh:
        """Unscaled local-frame mesh for the current geometry parameters."""
        key = (self.type_id, tuple(sorted((k, _freeze(v)) for k, v in self.params.items())))
        if self._mesh_cache is None or self._mesh_cache[0] != key:
            self._mesh_cache = (key, self.typedef.build_mesh(self.params))
            self._mass_cache = None
        return self._mesh_cache[1]

    def local_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.mesh().bounds()
        s = np.asarray(self.scale)
        a, b = lo * s, hi * s
        return np.minimum(a, b), np.maximum(a, b)

    def bounding_radius(self) -> float:
        lo, hi = self.local_bounds()
        return float(max(np.linalg.norm(np.maximum(np.abs(lo), np.abs(hi))), 1e-6))

    def axes_length(self) -> float:
        if self.local_axes_size > 0:
            return self.local_axes_size
        lo, hi = self.local_bounds()
        return float(max(np.max(hi - lo) * 0.5 * 1.35, 0.2))

    def derived_geometry(self) -> dict[str, tuple[str, float, str | None]]:
        d = self.typedef.derived
        return d(self.params) if d else {}

    # ---- mass properties ------------------------------------------------------
    def mass_properties(self) -> dict:
        """Mass, density, volume/area, centre of mass (local frame) and inertia about the COM.

        Inertia is expressed in the object's local frame and includes geometry scale.
        """
        td = self.typedef
        self.mesh()  # refresh the geometry cache
        key = (self._mesh_cache[0], tuple(self.scale), self.mass_mode, self.density, self.mass_input)
        if self._mass_cache is not None and self._mass_cache[0] == key:
            return self._mass_cache[1]
        model = td.mass_model
        out: dict[str, Any] = {"model": model, "volume": None, "area": None, "mass": None, "density": None,
                               "com": np.zeros(3), "inertia": np.zeros((3, 3)), "approximate": False,
                               "density_quantity": "density"}
        if model == "none":
            pass
        elif model == "point":
            out.update(mass=self.mass_input)
        else:
            mesh = self.mesh().scaled(self.scale)
            solid = model == "solid" and mesh.solid
            if solid:
                unit = numerics.mass_properties_solid(mesh.vertices, mesh.indices, 1.0)
                measure = unit["measure"]
                out["volume"] = measure
            else:
                unit = numerics.mass_properties_shell(mesh.vertices, mesh.indices, 1.0)
                measure = unit["measure"]
                out["area"] = measure
                out["density_quantity"] = "areal_density"
                out["approximate"] = model == "solid"  # open mesh treated as a shell
            if measure > 0:
                if self.mass_mode == "density":
                    mass = self.density * measure
                else:
                    mass = self.mass_input
                out.update(mass=mass, density=mass / measure, com=np.asarray(unit["centroid"], float),
                           inertia=np.asarray(unit["inertia"], float) * (mass / measure))
        self._mass_cache = (key, out)
        return out

    def mass(self) -> float | None:
        return self.mass_properties()["mass"]

    # ---- orientation helpers -----------------------------------------------------
    def euler(self) -> list[float]:
        return [float(a) for a in numerics.quat_to_euler(self.orientation)]

    def set_euler(self, angles) -> None:
        self.orientation = [float(c) for c in numerics.quat_from_euler(np.asarray(angles, float))]

    def set_orientation(self, q) -> None:
        q = numerics.quat_normalize(np.asarray(q, float))
        self.orientation = [float(c) for c in q]

    # ---- property path access ------------------------------------------------------
    def property_def(self, path: str) -> PropertyDef:
        if path.startswith("params."):
            key = path[7:]
            d = next((p for p in self.typedef.geometry if p.key == key), None)
            if d is None:
                raise PropertyError(f"Unknown geometry parameter '{key}'")
            return d
        if path.startswith("custom."):
            cp = self.custom.get(path[7:])
            if cp is None:
                raise PropertyError(f"Unknown custom property '{path[7:]}'")
            return cp.definition
        if path.startswith("overlay."):
            key = path[8:]
            if key not in ALL_OVERLAYS:
                raise PropertyError(f"Unknown overlay '{key}'")
            return PropertyDef(path, OVERLAY_LABELS.get(key, key), ValueType.BOOL, False)
        if path.startswith("trajectory."):
            d = TRAJECTORY_PROPERTIES.get(path[11:])
            if d is None:
                raise PropertyError(f"Unknown trajectory setting '{path[11:]}'")
            return d
        if path.startswith("label_unit."):
            return PropertyDef(path, f"Label unit ({path[11:]})", ValueType.STRING, "")
        if path.startswith("physics."):
            d = PHYSICS_PROPERTIES.get(path[8:])
            if d is None:
                raise PropertyError(f"Unknown physics setting '{path[8:]}'")
            return d
        if path.startswith("arrow_scale."):
            q = path[12:]
            if q not in ARROW_QUANTITIES:
                raise PropertyError(f"Unknown arrow quantity '{q}'")
            return PropertyDef(path, ARROW_QUANTITIES[q][0], ValueType.FLOAT, ARROW_QUANTITIES[q][1], minimum=0.0,
                               maximum=1e9, precision=6)
        if path == "orientation":
            return PropertyDef("orientation", "Quaternion", ValueType.STRING)
        d = STANDARD_PROPERTIES.get(path)
        if d is None:
            raise PropertyError(f"Unknown property '{path}'")
        return d

    def get(self, path: str) -> Any:
        if path.startswith("params."):
            return copy.deepcopy(self.params[path[7:]])
        if path.startswith("custom."):
            return copy.deepcopy(self.custom[path[7:]].value)
        if path.startswith("overlay."):
            return bool(self.overlays.get(path[8:], False))
        if path.startswith("trajectory."):
            return copy.deepcopy(self.trajectory_setting(path[11:]))
        if path.startswith("label_unit."):
            return self.label_units.get(path[11:], "")
        if path.startswith("physics."):
            return copy.deepcopy(self.physics_setting(path[8:]))
        if path.startswith("arrow_scale."):
            return self.arrow_scale(path[12:])
        if path == "orientation_euler":
            return self.euler()
        if path == "orientation":
            return list(self.orientation)
        if path == "tags":
            return ", ".join(self.tags)
        self.property_def(path)
        return copy.deepcopy(getattr(self, path))

    def set(self, path: str, value: Any, *, force: bool = False) -> None:
        """Validated assignment. Read-only custom properties require force=True."""
        if path.startswith("params."):
            key = path[7:]
            new = dict(self.params)
            new[key] = value
            self.params = self.typedef.coerce_params(new)
            return
        if path.startswith("custom."):
            cp = self.custom.get(path[7:])
            if cp is None:
                raise PropertyError(f"Unknown custom property '{path[7:]}'")
            if cp.definition.read_only and not force:
                raise PropertyError(f"'{cp.definition.label}' is read-only")
            cp.set_value(value)
            return
        if path.startswith("overlay."):
            self.property_def(path)
            self.overlays[path[8:]] = bool(value)
            return
        if path.startswith("trajectory."):
            d = self.property_def(path)
            v = d.coerce(value)
            if v == d.default:
                self.trajectory.pop(d.key, None)
            else:
                self.trajectory[d.key] = v
            return
        if path.startswith("label_unit."):
            v = str(value or "").strip()
            if v:
                self.label_units[path[11:]] = v
            else:
                self.label_units.pop(path[11:], None)
            return
        if path.startswith("arrow_scale."):
            self.arrow_scales[path[12:]] = self.property_def(path).coerce(value)
            return
        if path.startswith("physics."):
            d = self.property_def(path)
            v = d.coerce(value)
            if v == self._physics_default(d.key):
                self.physics.pop(d.key, None)
            else:
                self.physics[d.key] = v
            return
        if path == "orientation":
            self.set_orientation(value)
            return
        d = self.property_def(path)
        v = d.coerce(value)
        if path == "orientation_euler":
            self.set_euler(v)
        elif path == "tags":
            self.tags = _parse_tags(v)
        elif path == "material":
            self.material = v
            m = SURFACE_MATERIALS[v]
            if v != "Custom":
                self.metallic, self.roughness = m.metallic, m.roughness
        elif path in ("metallic", "roughness"):
            setattr(self, path, v)
            self.material = "Custom"
        elif path == "mass_mode" and self.typedef.mass_model == "point":
            self.mass_mode = "mass"
        else:
            setattr(self, path, v)

    def origin_of(self, path: str) -> Origin:
        return self.property_def(path).origin

    # ---- display settings -------------------------------------------------------------
    def trajectory_setting(self, key: str):
        d = TRAJECTORY_PROPERTIES[key]
        return self.trajectory.get(key, d.default)

    def _physics_default(self, key: str):
        if key == "collider" and self.typedef.rigid_body:
            return "auto"
        return PHYSICS_PROPERTIES[key].default

    def physics_setting(self, key: str):
        return self.physics.get(key, self._physics_default(key))

    def arrow_scale(self, quantity: str) -> float:
        return float(self.arrow_scales.get(quantity, ARROW_QUANTITIES[quantity][1]))

    # ---- custom properties ----------------------------------------------------------
    def add_custom(self, prop: CustomProperty) -> None:
        if prop.key in self.custom:
            raise PropertyError(f"A custom property with key '{prop.key}' already exists")
        self.custom[prop.key] = prop

    def remove_custom(self, key: str) -> None:
        self.custom.pop(key, None)

    # ---- serialization ------------------------------------------------------------
    def to_dict(self) -> dict:
        d = {"id": self.id, "type": self.type_id, "parent": self.parent_id, "tags": list(self.tags),
             "params": copy.deepcopy(self.params), "orientation": list(self.orientation),
             "overlays": {k: v for k, v in self.overlays.items() if v},
             "custom": [cp.to_dict() for cp in self.custom.values()]}
        if self.trajectory:
            d["trajectory"] = copy.deepcopy(self.trajectory)
        if self.label_units:
            d["label_units"] = dict(self.label_units)
        if self.arrow_scales:
            d["arrow_scales"] = dict(self.arrow_scales)
        if self.physics:
            d["physics"] = copy.deepcopy(self.physics)
        for k in _SIMPLE_ATTRS:
            d[k] = copy.deepcopy(getattr(self, k))
        return d

    @staticmethod
    def from_dict(data: dict) -> "SceneObject":
        """Create an object from serialized data, validating every field."""
        try:
            typedef = registry().get(str(data["type"]))
            obj = SceneObject(id=str(data["id"]), type_id=typedef.type_id)
            obj.params = typedef.coerce_params(dict(data.get("params", {})))
            for k in _SIMPLE_ATTRS:
                if k in data:
                    setattr(obj, k, STANDARD_PROPERTIES[k].coerce(data[k]))
            obj.parent_id = data.get("parent") or None
            obj.tags = _parse_tags(",".join(str(t) for t in data.get("tags", [])))
            q = np.asarray(data.get("orientation", [1, 0, 0, 0]), float).reshape(4)
            if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-9:
                raise PropertyError("invalid orientation quaternion")
            obj.set_orientation(q)
            obj.overlays = {str(k): bool(v) for k, v in dict(data.get("overlays", {})).items() if k in ALL_OVERLAYS}
            for k, v in dict(data.get("trajectory", {})).items():
                if k in TRAJECTORY_PROPERTIES:
                    obj.set(f"trajectory.{k}", v)
            obj.label_units = {str(k): str(v) for k, v in dict(data.get("label_units", {})).items() if v}
            for k, v in dict(data.get("arrow_scales", {})).items():
                if k in ARROW_QUANTITIES:
                    obj.set(f"arrow_scale.{k}", v)
            for k, v in dict(data.get("physics", {})).items():
                if k in PHYSICS_PROPERTIES:
                    obj.set(f"physics.{k}", v)
            obj.custom = {}
            for cd in data.get("custom", []):
                obj.add_custom(CustomProperty.from_dict(cd))
            if typedef.mass_model == "point":
                obj.mass_mode = "mass"
            return obj
        except (KeyError, TypeError, ValueError) as exc:
            raise PropertyError(f"Object '{data.get('id', '?')}': {exc}") from None

    def clone_state(self) -> dict:
        return self.to_dict()


def _parse_tags(text: str) -> list[str]:
    seen: list[str] = []
    for t in text.split(","):
        t = t.strip()
        if t and t not in seen:
            seen.append(t[:64])
    return seen


def _freeze(v):
    return tuple(v) if isinstance(v, list) else v
