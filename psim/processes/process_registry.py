"""Registry of approved process definitions.

The registry holds *definitions* (application code or signed formula
packages). Projects hold assignments that reference definitions by id; an
assignment can never add or change a definition.

Start-up order: built-in processes, the application's formula library
(psim/processes/library/*.json, shipped with the application), then signed
packages from the user data directory whose signatures verify against a
trusted key. After that the registry is *sealed*: any further registration,
publication or withdrawal requires a valid developer token.
"""
from __future__ import annotations

import glob
import json
import logging
import os

from .formula_process import FormulaProcessSpec, build_process_def, compile_spec, run_tests, version_tuple
from .permissions import AccessController, DeveloperToken, PermissionDenied, access_controller
from .process_base import EXCLUSIVE, SUMMED, UPDATE_WRITES, ProcessDef, ProcessError

log = logging.getLogger("psim.processes")
LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "library")


class ProcessRegistry:
    def __init__(self, access: AccessController | None = None) -> None:
        self._defs: dict[str, ProcessDef] = {}
        self.sources: dict[str, str] = {}      # process id -> "built-in" | "library" | package path
        self.load_errors: list[str] = []
        self._sealed = False
        self.access = access

    # ---- registration ---------------------------------------------------------------------------------------
    def register(self, pdef: ProcessDef, token: DeveloperToken | None = None, source: str = "built-in",
                 replace: bool = False) -> None:
        if self._sealed:
            self.access.require(token, "Registering a process")
        if pdef.process_id in self._defs and not replace:
            raise ProcessError(f"Process '{pdef.process_id}' is already registered")
        if pdef.kind not in ("contribution", "prescribed", "update"):
            raise ProcessError(f"{pdef.process_id}: invalid kind '{pdef.kind}'")
        if pdef.mode not in ("per_object", "global"):
            raise ProcessError(f"{pdef.process_id}: invalid mode '{pdef.mode}'")
        allowed = {"contribution": SUMMED, "prescribed": EXCLUSIVE, "update": UPDATE_WRITES}[pdef.kind]
        if not all(w in allowed or (pdef.kind == "update" and w.startswith("custom.")) for w in pdef.writes):
            raise ProcessError(f"{pdef.process_id}: a {pdef.kind} process may only write {sorted(allowed)}")
        keys = [p.key for p in pdef.parameters]
        if len(keys) != len(set(keys)):
            raise ProcessError(f"{pdef.process_id}: duplicate parameter keys")
        self._defs[pdef.process_id] = pdef
        self.sources[pdef.process_id] = source

    def seal(self, access: AccessController) -> None:
        self.access = access
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    # ---- queries -----------------------------------------------------------------------------------------------
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

    # ---- formula processes ------------------------------------------------------------------------------------------
    def register_spec(self, spec: FormulaProcessSpec, token: DeveloperToken | None = None, source: str = "library",
                      replace: bool = False) -> ProcessDef:
        cp, issues = compile_spec(spec)
        if cp is None:
            raise ProcessError(f"{spec.process_id}: " + "; ".join(str(i) for i in issues))
        pdef = build_process_def(cp)
        self.register(pdef, token, source=source, replace=replace)
        return pdef

    def publish(self, spec: FormulaProcessSpec, token: DeveloperToken, directory: str) -> tuple[ProcessDef, str]:
        """Validate, test, sign and register a formula process (developer only)."""
        self.access.require(token, "Publishing a process")
        cp, issues = compile_spec(spec)
        if cp is None:
            raise ProcessError("The process is not valid:\n" + "\n".join(str(i) for i in issues))
        failed = [r for r in run_tests(cp) if not r.passed]
        if failed:
            raise ProcessError("Test cases failed:\n" + "\n".join(f"{r.name}: {r.message}" for r in failed))
        existing = self._defs.get(spec.process_id)
        if existing is not None:
            if not existing.is_formula or self.sources.get(spec.process_id) in ("built-in", "library"):
                raise ProcessError(f"'{spec.process_id}' is an application process and cannot be replaced")
            if version_tuple(spec.version) <= version_tuple(existing.version):
                raise ProcessError(f"Version {spec.version} must be greater than the published {existing.version}")
        from .packages import EXTENSION, write_package
        path = os.path.join(directory, f"{spec.process_id}-{spec.version}{EXTENSION}")
        write_package(path, spec, self.access, token)
        old = self.sources.get(spec.process_id)
        pdef = build_process_def(cp)
        self.register(pdef, token, source=path, replace=True)
        if old and old not in ("built-in", "library") and old != path and os.path.exists(old):
            os.replace(old, old + ".superseded")
        return pdef, path

    def withdraw(self, process_id: str, token: DeveloperToken) -> None:
        self.access.require(token, "Withdrawing a process")
        src = self.sources.get(process_id)
        if src in (None, "built-in", "library"):
            raise ProcessError(f"'{process_id}' is not a published package and cannot be withdrawn")
        del self._defs[process_id]
        del self.sources[process_id]
        if os.path.exists(src):
            os.replace(src, src + ".withdrawn")

    def load_library(self) -> None:
        for path in sorted(glob.glob(os.path.join(LIBRARY_DIR, "*.json"))):
            try:
                with open(path, encoding="utf-8") as fh:
                    spec = FormulaProcessSpec.from_dict(json.load(fh))
                cp, issues = compile_spec(spec)
                if cp is None:
                    raise ProcessError("; ".join(str(i) for i in issues))
                failed = [r for r in run_tests(cp) if not r.passed]
                if failed:
                    raise ProcessError("tests failed: " + "; ".join(f"{r.name}: {r.message}" for r in failed))
                self.register(build_process_def(cp), source="library")
            except (OSError, ValueError, ProcessError) as exc:
                self.load_errors.append(f"Library process {os.path.basename(path)}: {exc}")

    def load_packages(self, directory: str, access: AccessController) -> None:
        from .packages import EXTENSION, read_package
        for path in sorted(glob.glob(os.path.join(directory, f"*{EXTENSION}"))):
            try:
                spec, ok, msg = read_package(path, access)
                if not ok:
                    raise ProcessError(f"rejected — {msg}")
                existing = self._defs.get(spec.process_id)
                if existing is not None:
                    if self.sources.get(spec.process_id) in ("built-in", "library"):
                        raise ProcessError("conflicts with an application process id")
                    if version_tuple(spec.version) <= version_tuple(existing.version):
                        continue  # an equal or newer version is already loaded
                self.register_spec(spec, source=path, replace=existing is not None)
            except (ProcessError, PermissionDenied) as exc:
                self.load_errors.append(f"Package {os.path.basename(path)}: {exc}")


_registry: ProcessRegistry | None = None


def process_registry() -> ProcessRegistry:
    global _registry
    if _registry is None:
        _registry = ProcessRegistry()
        from . import built_in_processes
        built_in_processes.register_all(_registry)
        _registry.load_library()
        access = access_controller()
        from .packages import packages_dir
        _registry.load_packages(packages_dir(access.base), access)
        _registry.seal(access)
        for msg in _registry.load_errors:
            log.warning(msg)
    return _registry
