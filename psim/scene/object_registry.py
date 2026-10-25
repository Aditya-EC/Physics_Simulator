"""Object type registry (internal extension API).

An ObjectTypeDef bundles everything the application needs to create and
display a kind of object: geometry parameters and their mesh generator, the
mass model, default appearance, local-frame display size, supported overlays
and validation rules. Geometry generation is kept separate from the property
system: the scene object stores only parameter values, and meshes / mass
properties are derived on demand.

New types are registered with `ObjectRegistry.register()` from trusted code
(built-ins or application plugins in psim/plugins). Ordinary scene editing
only *uses* registered types; project files can never register types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..properties.property_system import PropertyDef, PropertyError
from .geometry import Mesh

MASS_MODELS = ("solid", "shell", "point", "none")

ALL_OVERLAYS = ("name", "id", "position", "orientation", "velocity", "acceleration", "angular_velocity",
                "mass", "force", "torque", "custom", "local_axes", "bbox", "com")

CATEGORIES = (
    "Basic Solids", "Composite Shapes", "Mechanical Components", "Rigid Bodies", "Reference Frames",
    "Vectors & Measurement", "User-Defined Objects", "Saved Templates",
)


class RegistryError(Exception):
    pass


@dataclass
class ObjectTypeDef:
    type_id: str
    name: str
    category: str
    description: str
    geometry: list[PropertyDef]
    build_mesh: Callable[[dict], Mesh]
    mass_model: str = "solid"
    default_color: tuple[float, float, float] | None = None
    default_material: str = "Matte Plastic"
    default_density: float = 1000.0
    default_mass: float = 1.0
    uses_vertex_colors: bool = False
    rigid_body: bool = False
    axes_scale: float = 0.75  # local axes length relative to the bounding radius
    overlays: tuple[str, ...] = ALL_OVERLAYS
    validate: Callable[[dict], list[str]] | None = None
    derived: Callable[[dict], dict[str, tuple[str, float, str | None]]] | None = None
    tags: tuple[str, ...] = ()
    user_creatable: bool = True  # False for types created through dedicated workflows (e.g. imports)
    thumbnail_view: tuple[float, float] = (35.0, 25.0)  # azimuth, elevation in degrees

    @property
    def physical(self) -> bool:
        return self.mass_model != "none"

    def default_params(self) -> dict:
        return {p.key: p.default_value() for p in self.geometry}

    def coerce_params(self, params: dict) -> dict:
        """Validate parameter values; unknown keys are rejected."""
        out = self.default_params()
        known = {p.key: p for p in self.geometry}
        for k, v in params.items():
            if k not in known:
                raise PropertyError(f"Unknown geometry parameter '{k}' for {self.name}")
            out[k] = known[k].coerce(v)
        if self.validate:
            errors = self.validate(out)
            if errors:
                raise PropertyError("; ".join(errors))
        return out


@dataclass
class ObjectRegistry:
    _types: dict[str, ObjectTypeDef] = field(default_factory=dict)

    def register(self, typedef: ObjectTypeDef, replace: bool = False) -> None:
        if not typedef.type_id or "." not in typedef.type_id:
            raise RegistryError("type_id must be namespaced, e.g. 'basic.cube'")
        if typedef.type_id in self._types and not replace:
            raise RegistryError(f"Object type '{typedef.type_id}' is already registered")
        if typedef.mass_model not in MASS_MODELS:
            raise RegistryError(f"Invalid mass model '{typedef.mass_model}'")
        if typedef.category not in CATEGORIES:
            raise RegistryError(f"Unknown category '{typedef.category}'")
        keys = [p.key for p in typedef.geometry]
        if len(keys) != len(set(keys)):
            raise RegistryError("Duplicate geometry parameter keys")
        # Make sure the defaults produce a mesh.
        mesh = typedef.build_mesh(typedef.default_params())
        if mesh.triangle_count == 0:
            raise RegistryError(f"'{typedef.type_id}' produced an empty mesh")
        self._types[typedef.type_id] = typedef

    def get(self, type_id: str) -> ObjectTypeDef:
        try:
            return self._types[type_id]
        except KeyError:
            raise RegistryError(f"Unknown object type '{type_id}'") from None

    def has(self, type_id: str) -> bool:
        return type_id in self._types

    def types(self) -> list[ObjectTypeDef]:
        return list(self._types.values())

    def by_category(self, category: str) -> list[ObjectTypeDef]:
        return [t for t in self._types.values() if t.category == category]


_registry: ObjectRegistry | None = None


def registry() -> ObjectRegistry:
    """The application-wide registry, populated with built-in types and plugins."""
    global _registry
    if _registry is None:
        _registry = ObjectRegistry()
        from . import builtin_types
        builtin_types.register_all(_registry)
        from ..plugins import load_plugins
        load_plugins(_registry)
    return _registry
