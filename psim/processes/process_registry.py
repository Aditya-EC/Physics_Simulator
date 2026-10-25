"""Registry of approved process definitions.

The registry holds *definitions* only (application code). User projects hold
assignments that reference definitions by id; an assignment can never add
or change a definition.
"""
from __future__ import annotations

from .process_base import EXCLUSIVE, SUMMED, ProcessDef, ProcessError


class ProcessRegistry:
    def __init__(self) -> None:
        self._defs: dict[str, ProcessDef] = {}

    def register(self, pdef: ProcessDef) -> None:
        if pdef.process_id in self._defs:
            raise ProcessError(f"Process '{pdef.process_id}' is already registered")
        if pdef.kind not in ("contribution", "prescribed"):
            raise ProcessError(f"{pdef.process_id}: invalid kind '{pdef.kind}'")
        if pdef.mode not in ("per_object", "global"):
            raise ProcessError(f"{pdef.process_id}: invalid mode '{pdef.mode}'")
        allowed = SUMMED if pdef.kind == "contribution" else EXCLUSIVE
        if not set(pdef.writes) <= allowed:
            raise ProcessError(f"{pdef.process_id}: a {pdef.kind} process may only write {sorted(allowed)}")
        keys = [p.key for p in pdef.parameters]
        if len(keys) != len(set(keys)):
            raise ProcessError(f"{pdef.process_id}: duplicate parameter keys")
        self._defs[pdef.process_id] = pdef

    def get(self, process_id: str) -> ProcessDef:
        try:
            return self._defs[process_id]
        except KeyError:
            raise ProcessError(f"Unknown process '{process_id}'") from None

    def has(self, process_id: str) -> bool:
        return process_id in self._defs

    def definitions(self) -> list[ProcessDef]:
        return sorted(self._defs.values(), key=lambda d: (d.category, d.priority, d.name))

    def categories(self) -> list[str]:
        return sorted({d.category for d in self._defs.values()})


_registry: ProcessRegistry | None = None


def process_registry() -> ProcessRegistry:
    global _registry
    if _registry is None:
        _registry = ProcessRegistry()
        from . import built_in_processes
        built_in_processes.register_all(_registry)
    return _registry
