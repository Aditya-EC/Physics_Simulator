"""Typed property definitions shared by built-in, geometry and custom properties.

A PropertyDef describes *what* a property is (type, physical quantity, limits,
who controls it); values live on scene objects. Values are always stored in
canonical SI units; `quantity` names the unit-registry quantity kind used for
display conversion.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np


class PropertyError(ValueError):
    pass


class ValueType(str, Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    VEC2 = "vec2"
    VEC3 = "vec3"
    COLOR = "color"  # RGB floats in [0, 1]
    ENUM = "enum"

    @property
    def is_vector(self) -> bool:
        return self in (ValueType.VEC2, ValueType.VEC3)

    @property
    def size(self) -> int:
        return {ValueType.VEC2: 2, ValueType.VEC3: 3, ValueType.COLOR: 3}.get(self, 1)


class Origin(str, Enum):
    """Who controls a property's value."""

    USER = "user"              # set by the user (initial condition / parameter)
    CALCULATED = "calculated"  # computed from geometry (e.g. volume, inertia)
    DERIVED = "derived"        # derived from other properties (e.g. momentum = m v)
    SIMULATED = "simulated"    # written by simulation processes while running

    @property
    def badge(self) -> str:
        return {"user": "U", "calculated": "C", "derived": "D", "simulated": "S"}[self.value]

    @property
    def description(self) -> str:
        return {
            "user": "User-controlled value",
            "calculated": "Calculated from geometry and material",
            "derived": "Derived from other properties",
            "simulated": "Updated by simulation processes",
        }[self.value]


KEY_RE = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class PropertyDef:
    key: str
    label: str
    vtype: ValueType
    default: Any = None
    quantity: str | None = None  # unit-registry quantity kind
    description: str = ""
    origin: Origin = Origin.USER
    read_only: bool = False
    precision: int = 4
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    pattern: str | None = None  # regex for strings

    def with_default(self, default: Any) -> "PropertyDef":
        return replace(self, default=default)

    def coerce(self, value: Any) -> Any:
        """Validate and convert `value` to this property's canonical type."""
        t = self.vtype
        try:
            if t == ValueType.FLOAT:
                v = float(value)
                if not math.isfinite(v):
                    raise PropertyError(f"{self.label}: value must be finite")
                self._check_range(v)
                return v
            if t == ValueType.INT:
                if isinstance(value, float) and not value.is_integer():
                    raise PropertyError(f"{self.label}: value must be an integer")
                v = int(value)
                self._check_range(v)
                return v
            if t == ValueType.BOOL:
                if isinstance(value, str):
                    if value.lower() in ("true", "1", "yes"):
                        return True
                    if value.lower() in ("false", "0", "no"):
                        return False
                    raise PropertyError(f"{self.label}: '{value}' is not a Boolean")
                return bool(value)
            if t == ValueType.STRING:
                v = str(value)
                if self.pattern and not re.fullmatch(self.pattern, v):
                    raise PropertyError(f"{self.label}: '{v}' does not match the required pattern")
                return v
            if t == ValueType.ENUM:
                v = str(value)
                if self.choices and v not in self.choices:
                    raise PropertyError(f"{self.label}: '{v}' is not one of {', '.join(self.choices)}")
                return v
            if t in (ValueType.VEC2, ValueType.VEC3, ValueType.COLOR):
                arr = np.asarray(value, dtype=float).reshape(-1)
                if arr.size != t.size:
                    raise PropertyError(f"{self.label}: expected {t.size} components, got {arr.size}")
                if not np.all(np.isfinite(arr)):
                    raise PropertyError(f"{self.label}: components must be finite")
                if t == ValueType.COLOR:
                    arr = np.clip(arr, 0.0, 1.0)
                else:
                    for c in arr:
                        self._check_range(float(c))
                return [float(c) for c in arr]
        except PropertyError:
            raise
        except (TypeError, ValueError) as exc:
            raise PropertyError(f"{self.label}: invalid {t.value} value ({exc})") from None
        raise PropertyError(f"{self.label}: unsupported type {t}")

    def _check_range(self, v: float) -> None:
        if self.minimum is not None and v < self.minimum:
            raise PropertyError(f"{self.label}: {v:g} is below the minimum {self.minimum:g}")
        if self.maximum is not None and v > self.maximum:
            raise PropertyError(f"{self.label}: {v:g} is above the maximum {self.maximum:g}")

    def default_value(self) -> Any:
        if self.default is not None:
            return self.coerce(self.default)
        return {
            ValueType.FLOAT: 0.0, ValueType.INT: 0, ValueType.BOOL: False, ValueType.STRING: "",
            ValueType.VEC2: [0.0, 0.0], ValueType.VEC3: [0.0, 0.0, 0.0], ValueType.COLOR: [0.8, 0.8, 0.8],
            ValueType.ENUM: self.choices[0] if self.choices else "",
        }[self.vtype]


@dataclass
class PropertyGroup:
    """An ordered, titled list of property definitions (an inspector section)."""

    key: str
    title: str
    properties: list[PropertyDef] = field(default_factory=list)

    def get(self, key: str) -> PropertyDef | None:
        return next((p for p in self.properties if p.key == key), None)
