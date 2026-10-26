"""Validation of project-wide global parameters (read by formulas as global.<name>)."""
from __future__ import annotations

import math
import re

from ..units.unit_registry import UnitError, default_registry
from .process_base import ProcessError

NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")


def validate_globals(params: list[dict]) -> list[dict]:
    out, names = [], set()
    reg = default_registry()
    for g in params:
        try:
            name = str(g["name"])
            kind = str(g.get("type", "float"))
            quantity = g.get("quantity") or None
            value = g["value"]
        except (KeyError, TypeError):
            raise ProcessError(f"Global parameter {g!r} needs a name and a value") from None
        if not NAME_RE.match(name):
            raise ProcessError(f"Global parameter name '{name}' must be lowercase letters, digits and _")
        if name in names:
            raise ProcessError(f"Duplicate global parameter '{name}'")
        names.add(name)
        if kind not in ("float", "vec3"):
            raise ProcessError(f"Global parameter '{name}': type must be float or vec3")
        if quantity:
            try:
                reg.quantity(quantity)
            except UnitError as exc:
                raise ProcessError(f"Global parameter '{name}': {exc}") from None
        try:
            if kind == "float":
                value = float(value)
                ok = math.isfinite(value)
            else:
                value = [float(v) for v in value]
                ok = len(value) == 3 and all(math.isfinite(v) for v in value)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ProcessError(f"Global parameter '{name}': invalid {kind} value")
        out.append({"name": name, "type": kind, "quantity": quantity, "value": value,
                    "description": str(g.get("description", ""))})
    return out
