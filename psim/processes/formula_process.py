"""Formula-defined processes: specification, script language, compilation and tests.

A formula process is described by a ``FormulaProcessSpec`` (pure data, JSON
serializable). Its behaviour is a short script, one statement per line::

    # comments start with '#'
    let speed = norm(self.velocity)                     # local definition
    require self.mass > 0[kg] : "mass must be positive" # precondition
    force = -0.5 * param.rho * param.cd * param.area * speed * self.velocity

Output targets depend on the process kind:

* contribution: force, torque, acceleration, angular_acceleration (summed)
* prescribed:   position [, velocity], orientation [, angular_velocity]
* update:       position, velocity, orientation, angular_velocity, custom.<key>

Per-object processes write the current object (``force`` or ``self.force``);
global processes write assignment targets (``targets[0].force``).
Update processes read the state at the start of the step; ``updated(q)``
reads a value written earlier in the same step by another update process
(ordering is derived from these references and the declared ``after`` list).
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .. import numerics
from ..properties.property_system import PropertyDef, PropertyError, ValueType
from ..units.unit_registry import UnitError, default_registry
from .dimensional_analysis import (OUTPUT_TARGETS, TARGETS_BY_KIND, CheckEnv, Checker, FormulaTypeError, FType, T)
from .formula_parser import Attr, FormulaSyntaxError, Node, parse
from .formula_runtime import EvalContext, FormulaRuntimeError, QuatValue, compile_expr, finite
from .process_base import ProcessDef, ProcessError

PROCESS_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")
_VALUE_TYPES = {"float": ValueType.FLOAT, "int": ValueType.INT, "bool": ValueType.BOOL, "vec2": ValueType.VEC2,
                "vec3": ValueType.VEC3}
_FKIND = {"float": "num", "int": "num", "bool": "bool", "vec2": "vec2", "vec3": "vec3"}


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


@dataclass
class FormulaProcessSpec:
    process_id: str
    name: str
    version: str = "1.0.0"
    category: str = "Custom"
    description: str = ""
    kind: str = "contribution"            # contribution | prescribed | update
    mode: str = "per_object"              # per_object | global
    parameters: list[dict] = field(default_factory=list)       # key, label, type, quantity|unit, default, minimum, maximum, description
    globals: list[dict] = field(default_factory=list)          # name, type, quantity
    custom: list[dict] = field(default_factory=list)           # key, type, quantity (required custom properties)
    script: str = ""
    requires_mass: bool = False
    eligible_types: list[str] = field(default_factory=list)    # empty = any type
    min_targets: int = 1
    max_targets: int | None = None
    after: list[str] = field(default_factory=list)             # update-stage dependencies (process ids)
    error_policy: str = "stop"            # stop | skip
    documentation: str = ""
    tests: list[dict] = field(default_factory=list)
    author: str = ""

    def to_dict(self) -> dict:
        return copy.deepcopy(asdict(self))

    @staticmethod
    def from_dict(d: dict) -> "FormulaProcessSpec":
        if not isinstance(d, dict):
            raise ProcessError("Process specification must be an object")
        known = set(FormulaProcessSpec.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ProcessError(f"Unknown specification fields: {', '.join(sorted(unknown))}")
        try:
            return FormulaProcessSpec(**copy.deepcopy(d))
        except TypeError as exc:
            raise ProcessError(f"Invalid process specification: {exc}") from None


@dataclass
class Statement:
    line: int
    kind: str                  # let | require | output
    source: str                # the expression text
    name: str = ""             # let name or output quantity
    target_index: int | None = None
    message: str = ""
    node: Node | None = None


@dataclass
class ValidationIssue:
    line: int
    column: int
    message: str

    def __str__(self) -> str:
        loc = f"line {self.line}" + (f", column {self.column}" if self.column > 0 else "")
        return f"{loc}: {self.message}"


_LET_RE = re.compile(r"^let\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")
_REQ_RE = re.compile(r'^require\s+(.+?)(?:\s*:\s*"([^"]*)")?$')
_OUT_RE = re.compile(r"^((?:self\.|targets\[\s*(\d+)\s*\]\.)?(?:custom\.)?[A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)\s*(.+)$")
RESERVED = {"t", "dt", "self", "targets", "others", "it", "param", "global", "true", "false", "pi", "e", "g0", "G",
            "let", "require"}


def split_script(script: str) -> tuple[list[Statement], list[ValidationIssue]]:
    stmts, issues = [], []
    for ln, raw in enumerate(script.splitlines(), 1):
        line = raw.split("#", 1)[0].strip() if '"' not in raw else _strip_comment(raw)
        if not line:
            continue
        if line.startswith("let "):
            m = _LET_RE.match(line)
            if not m:
                issues.append(ValidationIssue(ln, 0, "Expected: let <name> = <expression>"))
                continue
            stmts.append(Statement(ln, "let", m.group(2), name=m.group(1)))
        elif line.startswith("require "):
            m = _REQ_RE.match(line)
            if not m:
                issues.append(ValidationIssue(ln, 0, 'Expected: require <condition> : "message"'))
                continue
            stmts.append(Statement(ln, "require", m.group(1), message=m.group(2) or f"precondition on line {ln}"))
        else:
            m = _OUT_RE.match(line)
            if not m:
                issues.append(ValidationIssue(ln, 0, "Expected a statement: let …, require …, or <output> = …"))
                continue
            target = m.group(1)
            idx = int(m.group(2)) if m.group(2) is not None else None
            name = re.sub(r"^(self\.|targets\[\s*\d+\s*\]\.)", "", target)
            stmts.append(Statement(ln, "output", m.group(3), name=name, target_index=idx))
    return stmts, issues


def _strip_comment(raw: str) -> str:
    out, in_str = [], False
    for ch in raw:
        if ch == '"':
            in_str = not in_str
        if ch == "#" and not in_str:
            break
        out.append(ch)
    return "".join(out).strip()


def _ftype(type_name: str, quantity: str | None, unit: str | None = None) -> FType:
    if type_name not in _FKIND:
        raise ProcessError(f"Unsupported type '{type_name}'")
    reg = default_registry()
    dim = None
    if quantity:
        dim = reg.quantity(quantity).dimension
    elif unit:
        dim = reg.unit(unit).dimension
    return T(_FKIND[type_name], dim) if dim is not None else T(_FKIND[type_name])


@dataclass
class CompiledProcess:
    spec: FormulaProcessSpec
    lets: list[tuple[str, Any]]
    requires: list[tuple[Any, str, int]]
    outputs: list[tuple[int | None, str, Any, int]]       # (target index, quantity, fn, line)
    param_defs: list[PropertyDef]
    object_ids: set[str]
    updated_refs: set[str]
    reads: set[str]

    @property
    def writes(self) -> set[str]:
        return {q for _, q, _, _ in self.outputs}


def _param_defs(spec: FormulaProcessSpec) -> list[PropertyDef]:
    defs = []
    reg = default_registry()
    for p in spec.parameters:
        try:
            key = str(p["key"])
            t = str(p.get("type", "float"))
            quantity = p.get("quantity") or (reg.quantity_for_unit(p["unit"]) if p.get("unit") else None)
            d = PropertyDef(key, str(p.get("label", key)), _VALUE_TYPES[t], None, quantity=quantity,
                            description=str(p.get("description", "")), minimum=p.get("minimum"),
                            maximum=p.get("maximum"))
            default = p.get("default")
            defs.append(d.with_default(d.coerce(default) if default is not None else None))
        except (KeyError, PropertyError, UnitError) as exc:
            raise ProcessError(f"Parameter definition {p!r}: {exc}") from None
    return defs


def compile_spec(spec: FormulaProcessSpec) -> tuple[CompiledProcess | None, list[ValidationIssue]]:
    """Validate and compile a specification. Returns (compiled or None, issues)."""
    issues: list[ValidationIssue] = []

    def meta(msg):
        issues.append(ValidationIssue(0, 0, msg))

    if not PROCESS_ID_RE.match(spec.process_id or ""):
        meta("Process id must be namespaced lowercase, e.g. 'lab.quadratic_drag'")
    if spec.process_id.split(".")[0] in ("kin", "dyn", "rot") and not spec.process_id.startswith("lib."):
        meta("The kin./dyn./rot. namespaces are reserved for built-in processes")
    if not spec.name.strip():
        meta("Name is required")
    if not VERSION_RE.match(spec.version or ""):
        meta("Version must look like 1.0 or 1.0.0")
    if spec.kind not in TARGETS_BY_KIND:
        meta(f"Kind must be one of {', '.join(TARGETS_BY_KIND)}")
    if spec.mode not in ("per_object", "global"):
        meta("Mode must be per_object or global")
    if spec.error_policy not in ("stop", "skip"):
        meta("Error policy must be 'stop' or 'skip'")
    if spec.min_targets < 1 or (spec.max_targets is not None and spec.max_targets < spec.min_targets):
        meta("Invalid target count limits")
    try:
        param_defs = _param_defs(spec)
        params = {d.key: _ftype({v: k for k, v in _VALUE_TYPES.items()}[d.vtype], d.quantity) for d in param_defs}
        globals_ = {str(g["name"]): _ftype(str(g.get("type", "float")), g.get("quantity"), g.get("unit"))
                    for g in spec.globals}
        custom = {str(c["key"]): _ftype(str(c.get("type", "float")), c.get("quantity"), c.get("unit"))
                  for c in spec.custom}
    except (ProcessError, KeyError, UnitError) as exc:
        meta(str(exc))
        return None, issues
    if issues:
        return None, issues

    stmts, script_issues = split_script(spec.script)
    issues += script_issues
    env = CheckEnv(mode=spec.mode, kind=spec.kind, params=params, globals_=globals_, custom=custom)
    object_ids: set[str] = set()
    updated: set[str] = set()
    reads: set[str] = set()
    lets, requires, outputs = [], [], []
    allowed = TARGETS_BY_KIND.get(spec.kind, set())
    seen_outputs: set[tuple] = set()
    for st in stmts:
        env.source = st.source
        try:
            node = parse(st.source)
            checker = Checker(env)
            ftype = checker.check(node)
        except (FormulaSyntaxError, FormulaTypeError) as exc:
            issues.append(ValidationIssue(st.line, exc.pos + 1 if exc.pos >= 0 else 0, exc.message))
            continue
        object_ids |= checker.object_ids
        updated |= checker.updated_refs
        reads |= _collect_reads(node)
        fn = compile_expr(node, checker.types, st.source)
        if st.kind == "let":
            if st.name in RESERVED or st.name in env.definitions:
                issues.append(ValidationIssue(st.line, 0, f"'{st.name}' is reserved or already defined"))
                continue
            env.definitions[st.name] = ftype
            lets.append((st.name, fn))
        elif st.kind == "require":
            if ftype.kind != "bool":
                issues.append(ValidationIssue(st.line, 0, f"A precondition must be a Boolean, got a {ftype.describe()}"))
                continue
            requires.append((fn, st.message, st.line))
        else:
            q = st.name
            base = "custom" if q.startswith("custom.") else q
            if base not in allowed:
                issues.append(ValidationIssue(st.line, 0, f"'{q}' cannot be written by a {spec.kind} process "
                                                          f"(allowed: {', '.join(sorted(allowed))})"))
                continue
            if spec.mode == "global" and st.target_index is None:
                issues.append(ValidationIssue(st.line, 0, "Global processes write targets[i].<quantity>"))
                continue
            if spec.mode == "per_object" and st.target_index is not None:
                issues.append(ValidationIssue(st.line, 0, "Per-object processes write their own object (e.g. force)"))
                continue
            if st.target_index is not None and spec.max_targets is not None and st.target_index >= spec.max_targets:
                issues.append(ValidationIssue(st.line, 0, f"targets[{st.target_index}] exceeds max_targets"))
                continue
            expected = custom.get(q[7:]) if base == "custom" else OUTPUT_TARGETS[q]
            if expected is None:
                issues.append(ValidationIssue(st.line, 0, f"Custom property '{q[7:]}' is not declared"))
                continue
            if ftype.kind != expected.kind or (not ftype.poly and ftype.dim != expected.dim):
                issues.append(ValidationIssue(st.line, 0, f"'{q}' needs a {expected.describe()}, "
                                                          f"but the expression is a {ftype.describe()}"))
                continue
            if (st.target_index, q) in seen_outputs:
                issues.append(ValidationIssue(st.line, 0, f"'{q}' is assigned more than once"))
                continue
            seen_outputs.add((st.target_index, q))
            outputs.append((st.target_index, q, fn, st.line))
    if not outputs and not script_issues:
        issues.append(ValidationIssue(0, 0, "The script has no output statement"))
    names = {q for _, q, _, _ in outputs}
    if spec.kind == "prescribed":
        if "velocity" in names and "position" not in names:
            issues.append(ValidationIssue(0, 0, "A prescribed velocity requires a prescribed position"))
        if "angular_velocity" in names and "orientation" not in names:
            issues.append(ValidationIssue(0, 0, "A prescribed angular velocity requires a prescribed orientation"))
    if spec.after and spec.kind != "update":
        issues.append(ValidationIssue(0, 0, "'after' dependencies apply to update processes only"))
    if issues:
        return None, issues
    return CompiledProcess(spec, lets, requires, outputs, param_defs, object_ids, updated, reads), []


def _collect_reads(node: Node) -> set[str]:
    out: set[str] = set()

    def walk(n):
        if isinstance(n, Attr):
            out.add(n.name)
        for v in vars(n).values():
            if isinstance(v, Node):
                walk(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, Node):
                        walk(x)
    walk(node)
    return out


# ---- evaluation -------------------------------------------------------------------------------------------

class PreconditionFailed(FormulaRuntimeError):
    pass


def evaluate_outputs(cp: CompiledProcess, world, self_ref, targets, t, dt, params, globals_) -> list[tuple]:
    """Evaluate the script once. Returns [(target_ref, quantity, value)]."""
    ctx = EvalContext(world, self_ref, targets, t, dt, params, globals_)
    for name, fn in cp.lets:
        ctx.defs[name] = fn(ctx)
    for fn, msg, line in cp.requires:
        if not fn(ctx):
            raise PreconditionFailed(f"Precondition failed (line {line}): {msg}")
    out = []
    for idx, q, fn, line in cp.outputs:
        v = fn(ctx)
        if not finite(v):
            raise FormulaRuntimeError(f"'{q}' (line {line}) is not finite")
        if idx is None:
            ref = self_ref
        else:
            if idx >= len(targets):
                raise FormulaRuntimeError(f"targets[{idx}] does not exist ({len(targets)} targets)")
            ref = targets[idx]
        out.append((ref, q, v))
    return out


def build_process_def(cp: CompiledProcess) -> ProcessDef:
    """Wrap a compiled formula process as a ProcessDef usable by the scheduler and engine."""
    spec = cp.spec
    if spec.kind == "contribution":   # accelerations are force contributions (summed)
        mapping = {"force": "force", "acceleration": "force", "torque": "torque", "angular_acceleration": "torque"}
    elif spec.kind == "prescribed":   # prescribed velocities belong to the prescribed position / orientation
        mapping = {"position": "position", "velocity": "position", "orientation": "orientation",
                   "angular_velocity": "orientation"}
    else:
        mapping = {}
    writes = tuple(dict.fromkeys(mapping.get(q, q) for q in sorted(cp.writes)))
    eligible_types = set(spec.eligible_types)
    required_custom = {c["key"]: c for c in spec.custom}

    def eligible(obj) -> bool:
        if eligible_types and obj.type_id not in eligible_types:
            return False
        for key, c in required_custom.items():
            prop = obj.custom.get(key)
            if prop is None or prop.definition.vtype.value != c.get("type", "float"):
                return False
        return True

    def run(world, idx_refs_pairs, t, dt, params) -> list[tuple]:
        """Evaluate for each target (per-object) or once (global); applies the error policy."""
        targets = tuple(world.ref_of_body(i) for i in idx_refs_pairs)
        params = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in params.items()}
        out = []
        selves = targets if spec.mode == "per_object" else (None,)
        for s in selves:
            try:
                out += evaluate_outputs(cp, world, s, targets, t, dt, params, world.globals)
            except FormulaRuntimeError as exc:
                if spec.error_policy == "skip":
                    world.report_skip(spec.process_id, str(exc))
                    continue
                raise ProcessError(f"{spec.name} ({spec.process_id}): {exc.describe()}") from None
        return out

    if spec.kind == "contribution":
        def evaluate(ctx, idx, params):
            world = ctx.world
            for ref, q, v in run(world, idx, ctx.t, ctx.dt, params):
                i = np.array([world.body_of_ref(ref)])
                v = np.asarray(v, float)[None, :]
                if q == "force":
                    ctx.add_force(i, v)
                elif q == "acceleration":
                    ctx.add_acceleration(i, v)
                elif q == "torque":
                    ctx.add_torque(i, v)
                else:
                    ctx.add_angular_acceleration(i, v)
    elif spec.kind == "prescribed":
        def evaluate(t, idx, params, bodies):
            world = bodies.world
            vals = {}
            for ref, q, v in run(world, idx, t, world.dt, params):
                vals.setdefault(q, {})[world.body_of_ref(ref)] = v
            res: dict[str, np.ndarray] = {}
            order = list(idx)
            if "position" in vals:
                res["position"] = np.array([vals["position"][i] for i in order], float)
                if "velocity" in vals:
                    res["velocity"] = np.array([vals["velocity"][i] for i in order], float)
                else:
                    res["velocity"] = _numeric_rate(run, world, idx, t, params, "position", order)
            if "orientation" in vals:
                res["orientation"] = np.array([vals["orientation"][i].q for i in order], float)
                if "angular_velocity" in vals:
                    res["angular_velocity"] = np.array([vals["angular_velocity"][i] for i in order], float)
                else:
                    res["angular_velocity"] = _numeric_rate(run, world, idx, t, params, "orientation", order)
            return res
    else:
        def evaluate(world, idx, t, dt, params):
            return run(world, idx, t, dt, params)

    return ProcessDef(
        spec.process_id, spec.name, spec.version, spec.category, spec.description, spec.kind, spec.mode,
        cp.param_defs, tuple(sorted(cp.reads)), writes, spec.script.strip(), evaluate,
        requires_mass=spec.requires_mass, min_targets=spec.min_targets, max_targets=spec.max_targets,
        eligible=eligible, preconditions="; ".join(m for _, m, _ in cp.requires),
        error_handling=("Errors stop the simulation with a report." if spec.error_policy == "stop" else
                        "Errors skip the affected object for that step and are reported."),
        formula_spec=spec, compiled=cp)


def _numeric_rate(run, world, idx, t, params, quantity, order) -> np.ndarray:
    """Central-difference time derivative of a prescribed position (velocity) or orientation (angular velocity)."""
    h = 1e-6 * max(1.0, abs(t))
    plus = {world.body_of_ref(r): v for r, q, v in run(world, idx, t + h, world.dt, params) if q == quantity}
    minus = {world.body_of_ref(r): v for r, q, v in run(world, idx, t - h, world.dt, params) if q == quantity}
    rows = []
    for i in order:
        if quantity == "position":
            rows.append((np.asarray(plus[i]) - np.asarray(minus[i])) / (2 * h))
        else:
            dq = numerics.quat_multiply(plus[i].q, numerics.quat_conjugate(minus[i].q))
            dq = numerics.quat_normalize(dq)
            angle = 2 * math.atan2(float(np.linalg.norm(dq[1:])), float(dq[0]))
            axis = dq[1:] / np.linalg.norm(dq[1:]) if np.linalg.norm(dq[1:]) > 0 else np.zeros(3)
            rows.append(axis * angle / (2 * h))
    return np.array(rows, float)


# ---- test cases ---------------------------------------------------------------------------------------------

DEFAULT_OBJECT = {
    "position": [0.0, 0.0, 0.0], "com": None, "velocity": [0.0, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.0],
    "orientation": [1.0, 0.0, 0.0, 0.0], "mass": 1.0, "density": 1000.0, "volume": 0.001,
    "inertia_local": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "com_local": [0.0, 0.0, 0.0], "force": [0.0, 0.0, 0.0],
    "torque": [0.0, 0.0, 0.0], "linear_acceleration": [0.0, 0.0, 0.0], "angular_acceleration": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0], "custom": {}, "type": "basic.cube", "tags": [],
}


class MockWorld:
    """Minimal world for process test cases (objects given as plain attribute dictionaries)."""

    def __init__(self, objects: dict[str, dict], globals_: dict, dt: float) -> None:
        self.ids = list(objects)
        self.objs = []
        for oid in self.ids:
            o = dict(DEFAULT_OBJECT)
            o.update(objects[oid] or {})
            self.objs.append(o)
        self.globals = globals_
        self.dt = dt
        self.updated_values: dict = {}
        self.skips: list[str] = []

    def ref_of_body(self, i):
        return int(i)

    def body_of_ref(self, ref):
        return int(ref)

    def report_skip(self, pid, msg):
        self.skips.append(msg)

    def attr(self, ref, name):
        o = self.objs[ref]
        q = QuatValue(o["orientation"])
        r = np.asarray(numerics.quat_to_matrix(q.q), float)
        if name in ("velocity", "linear_velocity"):
            return np.asarray(o["velocity"], float)
        if name == "orientation" or name == "orientation0":
            return q
        if name == "euler":
            return np.asarray(numerics.quat_to_euler(q.q), float)
        if name in ("position", "position0"):
            return np.asarray(o["position"], float)
        if name == "com":
            return np.asarray(o["position"], float) + r @ np.asarray(o["com_local"], float)
        if name == "inertia":
            return r @ np.asarray(o["inertia_local"], float) @ r.T
        if name in ("x_axis", "y_axis", "z_axis"):
            return r[:, "xyz".index(name[0])].copy()
        if name == "parent":
            return None
        if name == "children":
            return ()
        v = o[name]
        return float(v) if isinstance(v, (int, float)) else np.asarray(v, float)

    def custom(self, ref, key):
        try:
            v = self.objs[ref]["custom"][key]
        except KeyError:
            raise FormulaRuntimeError(f"Object {self.ids[ref]} has no custom property '{key}'") from None
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else (
            v if isinstance(v, bool) else np.asarray(v, float))

    def updated(self, ref, q):
        return self.updated_values[(ref, q)]

    def resolve(self, oid):
        if oid not in self.ids:
            raise FormulaRuntimeError(f"Missing object reference '{oid}'")
        return self.ids.index(oid)

    def all_refs(self):
        return tuple(range(len(self.ids)))

    def tagged(self, tag):
        return tuple(i for i, o in enumerate(self.objs) if tag in o["tags"])

    def of_type(self, tid):
        return tuple(i for i, o in enumerate(self.objs) if o["type"] == tid)

    def children(self, ref):
        return ()


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str


def _to_value(v):
    if isinstance(v, QuatValue):
        return [float(c) for c in v.q]
    if isinstance(v, np.ndarray):
        return [float(c) for c in v.reshape(-1)]
    return v


def run_tests(cp: CompiledProcess) -> list[TestResult]:
    """Run the specification's test cases.

    A test case: {"name", "objects": {"self" | id: {attr: value}}, "targets": [ids], "params": {...},
    "globals": {...}, "updated": {"<id>.<quantity>": value}, "t", "dt", "expect": {"<quantity>" or
    "<id>.<quantity>": value}, "tolerance", "expect_error": "substring"}.
    Values are SI. Objects default to a 1 kg unit cube at rest at the origin.
    """
    results = []
    for k, tc in enumerate(cp.spec.tests):
        name = str(tc.get("name", f"test {k + 1}"))
        try:
            objects = tc.get("objects") or {"self": {}}
            world = MockWorld(objects, {g: v for g, v in (tc.get("globals") or {}).items()}, float(tc.get("dt", 0.01)))
            for key, val in (tc.get("updated") or {}).items():
                oid, q = key.split(".", 1)
                world.updated_values[(world.resolve(oid), q)] = QuatValue(val) if q == "orientation" else (
                    np.asarray(val, float) if isinstance(val, list) else float(val))
            target_ids = tc.get("targets") or [world.ids[0]]
            targets = tuple(world.resolve(t) for t in target_ids)
            params = {d.key: d.coerce(tc.get("params", {}).get(d.key, d.default_value())) for d in cp.param_defs}
            params = {k2: (np.asarray(v, float) if isinstance(v, list) else v) for k2, v in params.items()}
            self_ref = targets[0] if cp.spec.mode == "per_object" else None
            out = evaluate_outputs(cp, world, self_ref, targets, float(tc.get("t", 0.0)), world.dt, params,
                                   world.globals)
        except (FormulaRuntimeError, PropertyError, KeyError, ValueError, TypeError) as exc:
            want = tc.get("expect_error")
            msg = exc.describe() if isinstance(exc, FormulaRuntimeError) else str(exc)
            if want and want in msg:
                results.append(TestResult(name, True, f"raised as expected: {msg.splitlines()[0]}"))
            else:
                results.append(TestResult(name, False, f"error: {msg.splitlines()[0]}"))
            continue
        if tc.get("expect_error"):
            results.append(TestResult(name, False, f"expected an error containing '{tc['expect_error']}'"))
            continue
        tol = float(tc.get("tolerance", 1e-9))
        produced = {}
        for ref, q, v in out:
            produced[f"{world.ids[ref]}.{q}"] = _to_value(v)
            if ref == targets[0]:
                produced.setdefault(q, _to_value(v))
        failures = []
        for key, want in (tc.get("expect") or {}).items():
            got = produced.get(key)
            if got is None:
                failures.append(f"{key} was not produced")
                continue
            if not np.allclose(np.asarray(got, float), np.asarray(want, float), rtol=tol, atol=tol):
                failures.append(f"{key}: expected {want}, got {[round(x, 12) for x in np.atleast_1d(got)]}")
        results.append(TestResult(name, not failures, "; ".join(failures) or "passed"))
    return results
