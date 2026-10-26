"""Conditional stopping events: Boolean formulas checked after every simulation step.

A condition is written in the safe formula language (no ``eval``), e.g.

    object("OBJ-000002").position.z < 0[m]
    norm(object("OBJ-000003").velocity) > 20[m/s] && t > 1[s]

It is type- and dimension-checked when the run is planned. The action (pause or stop) fires
when the expression *becomes* true — on the step where it changes from false to true — so a
resumed run is not paused again by a condition that is still true.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..processes.dimensional_analysis import Checker, CheckEnv, FormulaTypeError
from ..processes.formula_parser import FormulaSyntaxError, parse
from ..processes.formula_runtime import EvalContext, FormulaRuntimeError, compile_expr

ACTIONS = ("pause", "stop")
CONDITION_KEYS = {"name", "expression", "action", "enabled"}
_NAME_RE = re.compile(r".*\S.*")


class ConditionError(ValueError):
    pass


def validate_conditions(data) -> list[dict]:
    """Normalised condition dicts; raises ConditionError for malformed entries (expressions are checked later)."""
    if not isinstance(data, list):
        raise ConditionError("Stop conditions must be a list")
    out, names = [], set()
    for d in data:
        if not isinstance(d, dict):
            raise ConditionError("A stop condition must be an object")
        unknown = set(d) - CONDITION_KEYS
        if unknown:
            raise ConditionError(f"Stop condition has unknown field(s) {', '.join(sorted(unknown))}")
        name = str(d.get("name", "")).strip()
        expr = str(d.get("expression", "")).strip()
        action = str(d.get("action", "pause"))
        if not _NAME_RE.match(name):
            raise ConditionError("A stop condition needs a name")
        if name.lower() in names:
            raise ConditionError(f"Duplicate stop condition name '{name}'")
        if not expr:
            raise ConditionError(f"Stop condition '{name}' has no expression")
        if action not in ACTIONS:
            raise ConditionError(f"Stop condition '{name}': action must be pause or stop")
        names.add(name.lower())
        out.append({"name": name, "expression": expr, "action": action, "enabled": bool(d.get("enabled", True))})
    return out


@dataclass
class CompiledCondition:
    name: str
    expression: str
    action: str
    fn: object
    object_ids: set[str] = field(default_factory=set)
    frame_ids: set[str] = field(default_factory=set)
    was_true: bool = False


def compile_condition(cond: dict, globals_types: dict) -> CompiledCondition:
    """Parse, type-check (must be Boolean) and compile one condition. Raises ConditionError with the position."""
    src = cond["expression"]
    env = CheckEnv(mode="global", kind="update", globals_=dict(globals_types), source=src)
    try:
        node = parse(src)
        checker = Checker(env)
        ftype = checker.check(node)
    except (FormulaSyntaxError, FormulaTypeError) as exc:
        raise ConditionError(f"Stop condition '{cond['name']}': {exc}") from None
    if ftype.kind != "bool":
        raise ConditionError(f"Stop condition '{cond['name']}' must be a Boolean (a comparison), "
                             f"but it is a {ftype.describe()}")
    return CompiledCondition(cond["name"], src, cond["action"], compile_expr(node, checker.types, src),
                             set(checker.object_ids), set(checker.frame_ids))


def evaluate(conditions: list[CompiledCondition], world, t: float, dt: float) -> list[CompiledCondition]:
    """Conditions that became true at this step (false → true), in definition order."""
    fired = []
    ctx = EvalContext(world, None, (), t, dt, {}, world.globals)
    for c in conditions:
        try:
            now = bool(c.fn(ctx))
        except FormulaRuntimeError as exc:
            raise FormulaRuntimeError(f"Stop condition '{c.name}': {exc}") from None
        if now and not c.was_true:
            fired.append(c)
        c.was_true = now
    return fired


def attach_conditions(plan, scene, settings) -> None:
    """Compile the enabled stop conditions of `settings` into `plan.conditions`; problems become plan errors."""
    from ..processes.formula_process import _ftype
    from ..processes.process_scheduler import Diagnostic
    globals_types = {}
    for g in scene.global_parameters:
        try:
            globals_types[g["name"]] = _ftype(g.get("type", "float"), g.get("quantity"), g.get("unit"))
        except Exception:   # malformed globals are reported by the process plan
            continue
    plan.conditions = []
    try:
        conditions = validate_conditions(settings.stop_conditions)
    except ConditionError as exc:
        plan.diagnostics.append(Diagnostic("error", str(exc)))
        return
    for cond in conditions:
        if not cond.get("enabled", True):
            continue
        try:
            cc = compile_condition(cond, globals_types)
        except ConditionError as exc:
            plan.diagnostics.append(Diagnostic("error", str(exc)))
            continue
        missing = sorted(i for i in cc.object_ids if i not in scene.objects)
        missing_frames = sorted(i for i in cc.frame_ids if scene.frame(i) is None)
        if missing or missing_frames:
            plan.diagnostics.append(Diagnostic("error", f"Stop condition '{cc.name}' refers to the missing "
                                                        f"{'object' if missing else 'frame'} "
                                                        f"'{(missing or missing_frames)[0]}'"))
            continue
        plan.conditions.append(cc)
