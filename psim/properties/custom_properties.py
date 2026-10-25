"""User-defined object properties.

A custom property has a stable internal key (used by future processes,
plots, labels and exports), a display name, a typed value, an optional
physical quantity/unit and optional validation limits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .property_system import KEY_RE, Origin, PropertyDef, PropertyError, ValueType

CUSTOM_TYPES = (ValueType.FLOAT, ValueType.INT, ValueType.BOOL, ValueType.STRING, ValueType.VEC2, ValueType.VEC3)


def make_key(name: str) -> str:
    """Suggest a stable key from a display name: 'Drag coeff.' -> 'drag_coeff'."""
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not key or key[0].isdigit():
        key = f"p_{key}" if key else "property"
    return key[:64]


@dataclass
class CustomProperty:
    definition: PropertyDef
    value: Any

    @property
    def key(self) -> str:
        return self.definition.key

    @staticmethod
    def create(*, key: str, name: str, vtype: ValueType | str, value: Any = None, default: Any = None,
               quantity: str | None = None, unit: str | None = None, description: str = "",
               read_only: bool = False, precision: int = 4, minimum: float | None = None,
               maximum: float | None = None) -> "CustomProperty":
        vtype = ValueType(vtype)
        if vtype not in CUSTOM_TYPES:
            raise PropertyError(f"Unsupported custom property type '{vtype.value}'")
        if not KEY_RE.match(key):
            raise PropertyError("Key must start with a letter or underscore and contain only a-z, 0-9 and _")
        if not name.strip():
            raise PropertyError("Name must not be empty")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise PropertyError("Minimum must not exceed maximum")
        if quantity is None and unit:
            from ..units.unit_registry import default_registry
            quantity = default_registry().quantity_for_unit(unit)
        numeric = vtype in (ValueType.FLOAT, ValueType.INT, ValueType.VEC2, ValueType.VEC3)
        d = PropertyDef(
            key=key, label=name.strip(), vtype=vtype, quantity=quantity if numeric else None,
            description=description, origin=Origin.USER, read_only=read_only,
            precision=max(0, min(12, int(precision))),
            minimum=minimum if numeric else None, maximum=maximum if numeric else None,
        )
        d = d.with_default(d.coerce(default) if default is not None else None)
        v = d.coerce(value) if value is not None else d.default_value()
        return CustomProperty(d, v)

    def set_value(self, value: Any) -> None:
        self.value = self.definition.coerce(value)

    def to_dict(self) -> dict:
        d = self.definition
        return {
            "key": d.key, "name": d.label, "type": d.vtype.value, "value": self.value, "default": d.default,
            "quantity": d.quantity, "description": d.description, "read_only": d.read_only,
            "precision": d.precision, "minimum": d.minimum, "maximum": d.maximum,
        }

    @staticmethod
    def from_dict(data: dict) -> "CustomProperty":
        return CustomProperty.create(
            key=str(data["key"]), name=str(data.get("name", data["key"])), vtype=data["type"],
            value=data.get("value"), default=data.get("default"), quantity=data.get("quantity"),
            description=str(data.get("description", "")), read_only=bool(data.get("read_only", False)),
            precision=int(data.get("precision", 4)), minimum=data.get("minimum"), maximum=data.get("maximum"),
        )
