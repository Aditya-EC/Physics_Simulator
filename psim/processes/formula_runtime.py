"""Compilation of checked formulas into Python closures, and their evaluation.

``compile_expr`` turns a type-annotated AST into a tree of small closures
(no ``eval``/``exec``). Operations are selected once, at compile time, from
the static types, so evaluation does not re-inspect types. Runtime errors
(division by zero, domain errors, non-finite results, missing objects)
raise ``FormulaRuntimeError`` with the source position.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Protocol

import numpy as np

from .. import numerics
from .dimensional_analysis import CONSTANTS, FType
from .formula_parser import Attr, Binary, Call, Cond, FormulaSyntaxError, Index, Name, Node, Num, Str, Unary


class FormulaRuntimeError(FormulaSyntaxError):
    pass


class QuatValue:
    """Quaternion value (w, x, y, z), distinct from 4-vectors."""

    __slots__ = ("q",)

    def __init__(self, q) -> None:
        self.q = np.asarray(numerics.quat_normalize(np.asarray(q, float)), float)

    def __repr__(self) -> str:
        return f"Quat{tuple(round(float(c), 6) for c in self.q)}"


class World(Protocol):
    """What formulas can see of the scene (implemented by the simulation engine)."""

    def attr(self, ref: int, name: str) -> Any: ...
    def custom(self, ref: int, key: str) -> Any: ...
    def updated(self, ref: int, quantity: str) -> Any: ...
    def resolve(self, object_id: str) -> int: ...
    def all_refs(self) -> tuple[int, ...]: ...
    def tagged(self, tag: str) -> tuple[int, ...]: ...
    def of_type(self, type_id: str) -> tuple[int, ...]: ...
    def children(self, ref: int) -> tuple[int, ...]: ...


class EvalContext:
    __slots__ = ("world", "self_ref", "targets", "t", "dt", "params", "globals", "defs", "it")

    def __init__(self, world, self_ref, targets, t, dt, params, globals_, defs=None) -> None:
        self.world, self.self_ref, self.targets = world, self_ref, targets
        self.t, self.dt, self.params, self.globals = t, dt, params, globals_
        self.defs = defs if defs is not None else {}
        self.it: list[int] = []


Fn = Callable[[EvalContext], Any]


class Compiler:
    def __init__(self, types: dict[int, FType], source: str) -> None:
        self.types = types
        self.source = source

    def err(self, msg: str, node: Node) -> FormulaRuntimeError:
        return FormulaRuntimeError(msg, node.pos, self.source)

    def type_of(self, node: Node) -> FType:
        return self.types[id(node)]

    def compile(self, node: Node) -> Fn:
        if isinstance(node, Num):
            v = float(node.value)
            return lambda ctx: v
        if isinstance(node, Str):
            s = node.value
            return lambda ctx: s
        if isinstance(node, Name):
            return self._name(node)
        if isinstance(node, Attr):
            return self._attr(node)
        if isinstance(node, Index):
            return self._index(node)
        if isinstance(node, Call):
            return self._call(node)
        if isinstance(node, Unary):
            f = self.compile(node.operand)
            if node.op == "!":
                return lambda ctx: not f(ctx)
            if self.type_of(node.operand).kind == "quat":
                return lambda ctx: QuatValue(-f(ctx).q)
            return lambda ctx: -f(ctx)
        if isinstance(node, Cond):
            c, a, b = self.compile(node.test), self.compile(node.then), self.compile(node.other)
            return lambda ctx: a(ctx) if c(ctx) else b(ctx)
        if isinstance(node, Binary):
            return self._binary(node)
        raise self.err("Unsupported expression", node)

    # ---- names and attributes ------------------------------------------------------------------------
    def _name(self, n: Name) -> Fn:
        i = n.id
        if i == "t":
            return lambda ctx: ctx.t
        if i == "dt":
            return lambda ctx: ctx.dt
        if i in ("true", "false"):
            v = i == "true"
            return lambda ctx: v
        if i == "self":
            return lambda ctx: ctx.self_ref
        if i == "targets":
            return lambda ctx: ctx.targets
        if i == "others":
            return lambda ctx: tuple(r for r in ctx.targets if r != ctx.self_ref)
        if i == "it":
            return lambda ctx: ctx.it[-1]
        if i in CONSTANTS:
            v = CONSTANTS[i][0]
            return lambda ctx: v
        return lambda ctx: ctx.defs[i]  # let-definition (checked statically)

    def _attr(self, n: Attr) -> Fn:
        if isinstance(n.obj, Name) and n.obj.id == "param":
            key = n.name
            return lambda ctx: ctx.params[key]
        if isinstance(n.obj, Name) and n.obj.id == "global":
            key = n.name
            return lambda ctx: ctx.globals[key]
        if isinstance(n.obj, Attr) and n.obj.name == "custom":
            obj = self.compile(n.obj.obj)
            key = n.name
            return lambda ctx: ctx.world.custom(obj(ctx), key)
        obj = self.compile(n.obj)
        kind = self.type_of(n.obj).kind
        name = n.name
        if kind == "obj":
            def get(ctx):
                ref = obj(ctx)
                if ref is None:
                    raise self.err(f"Object has no {name} (no parent?)", n)
                return ctx.world.attr(ref, name)
            return get
        if kind == "quat":
            k = "wxyz".index(name)
            return lambda ctx: float(obj(ctx).q[k])
        k = "xyz".index(name)
        return lambda ctx: float(obj(ctx)[k])

    def _index(self, n: Index) -> Fn:
        obj, idx = self.compile(n.obj), self.compile(n.index)
        kind = self.type_of(n.obj).kind

        def get(ctx):
            seq = obj(ctx)
            i = idx(ctx)
            if not float(i).is_integer():
                raise self.err(f"Index {i:g} is not an integer", n)
            i = int(i)
            if not -len(seq) <= i < len(seq):
                raise self.err(f"Index {i} is out of range (size {len(seq)})", n)
            return seq[i] if kind == "set" else float(seq[i])
        return get

    # ---- operators -----------------------------------------------------------------------------------------
    def _binary(self, n: Binary) -> Fn:
        a, b = self.compile(n.left), self.compile(n.right)
        ta = self.type_of(n.left)
        op = n.op
        if op == "&&":
            return lambda ctx: bool(a(ctx)) and bool(b(ctx))
        if op == "||":
            return lambda ctx: bool(a(ctx)) or bool(b(ctx))
        if op in ("==", "!=", "<", "<=", ">", ">="):
            cmp = {"==": lambda x, y: x == y, "!=": lambda x, y: x != y, "<": lambda x, y: x < y,
                   "<=": lambda x, y: x <= y, ">": lambda x, y: x > y, ">=": lambda x, y: x >= y}[op]
            return lambda ctx: bool(cmp(a(ctx), b(ctx)))
        if op == "+":
            return lambda ctx: a(ctx) + b(ctx)
        if op == "-":
            return lambda ctx: a(ctx) - b(ctx)
        if op == "*":
            if ta.kind == "quat":
                return lambda ctx: QuatValue(numerics.quat_multiply(a(ctx).q, b(ctx).q))
            if ta.kind == "mat3":
                return lambda ctx: a(ctx) @ b(ctx)
            return lambda ctx: a(ctx) * b(ctx)
        if op in ("/", "%"):
            def div(ctx):
                d = b(ctx)
                if d == 0:
                    raise self.err("Division by zero", n)
                return a(ctx) / d if op == "/" else a(ctx) % d
            return div
        if op == "^":
            def power(ctx):
                x, y = a(ctx), b(ctx)
                if x < 0 and not float(y).is_integer():
                    raise self.err(f"Negative base {x:g} raised to non-integer power {y:g}", n)
                if x == 0 and y < 0:
                    raise self.err("Zero raised to a negative power", n)
                try:
                    return float(x) ** float(y)
                except OverflowError:
                    raise self.err("Numerical overflow in '^'", n) from None
            return power
        raise self.err(f"Unknown operator '{op}'", n)

    # ---- calls -------------------------------------------------------------------------------------------
    def _call(self, n: Call) -> Fn:
        f = n.func
        if f == "object":
            oid = n.args[0].value
            return lambda ctx: ctx.world.resolve(oid)
        if f == "tagged":
            tag = n.args[0].value
            return lambda ctx: ctx.world.tagged(tag)
        if f == "of_type":
            tid = n.args[0].value
            return lambda ctx: ctx.world.of_type(tid)
        if f == "all":
            return lambda ctx: ctx.world.all_refs()
        if f == "children":
            o = self.compile(n.args[0])
            return lambda ctx: ctx.world.children(o(ctx))
        if f == "count":
            s = self.compile(n.args[0])
            return lambda ctx: float(len(s(ctx)))
        if f in ("sum_of", "mean_of", "min_of", "max_of"):
            return self._aggregate(n)
        if f == "updated":
            a = n.args[0]
            obj = self.compile(a.obj.obj if isinstance(a.obj, Attr) else a.obj)
            q = f"custom.{a.name}" if isinstance(a.obj, Attr) and a.obj.name == "custom" else \
                {"linear_velocity": "velocity"}.get(a.name, a.name)

            def upd(ctx):
                try:
                    return ctx.world.updated(obj(ctx), q)
                except KeyError:
                    raise self.err(f"updated({q}) was not written earlier in this step", n) from None
            return upd
        if f == "if":
            c, x, y = (self.compile(a) for a in n.args)
            return lambda ctx: x(ctx) if c(ctx) else y(ctx)
        if f in ("to_local", "to_global", "point_to_local", "point_to_global"):
            o, v = self.compile(n.args[0]), self.compile(n.args[1])
            to_local = f.endswith("local")
            point = f.startswith("point_")

            def frame(ctx):
                ref = o(ctx)
                r = np.asarray(numerics.quat_to_matrix(ctx.world.attr(ref, "orientation").q), float)
                vec = np.asarray(v(ctx), float)
                if to_local:
                    if point:
                        vec = vec - ctx.world.attr(ref, "position")
                    return r.T @ vec
                out = r @ vec
                return out + ctx.world.attr(ref, "position") if point else out
            return frame
        args = [self.compile(a) for a in n.args]
        impl = _IMPLS[f]
        node = n

        def call(ctx):
            try:
                return impl(*[a(ctx) for a in args])
            except _DomainError as exc:
                raise self.err(f"{f}(): {exc}", node) from None
        return call

    def _aggregate(self, n: Call) -> Fn:
        s, e = self.compile(n.args[0]), self.compile(n.args[1])
        kind = self.type_of(n.args[1]).kind
        f = n.func
        zero = 0.0 if kind == "num" else np.zeros({"vec2": 2, "vec3": 3, "mat3": (3, 3)}[kind])

        def agg(ctx):
            refs = s(ctx)
            vals = []
            for r in refs:
                ctx.it.append(r)
                try:
                    vals.append(e(ctx))
                finally:
                    ctx.it.pop()
            if f == "sum_of":
                return sum(vals, zero * 1.0)
            if not vals:
                raise self.err(f"{f}() over an empty object set", n)
            if f == "mean_of":
                return sum(vals, zero * 1.0) / len(vals)
            return float(min(vals) if f == "min_of" else max(vals))
        return agg


class _DomainError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise _DomainError(msg)


def _sqrt(x):
    _check(x >= 0, f"negative argument {x:g}")
    return math.sqrt(x)


def _log(x, base=None):
    _check(x > 0, f"non-positive argument {x:g}")
    return math.log(x) if base is None else math.log10(x)


def _asin(x):
    _check(-1 <= x <= 1, f"argument {x:g} outside [-1, 1]")
    return math.asin(x)


def _acos(x):
    _check(-1 <= x <= 1, f"argument {x:g} outside [-1, 1]")
    return math.acos(x)


def _exp(x):
    _check(x < 709, f"overflow for argument {x:g}")
    return math.exp(x)


def _normalize(v):
    n = float(np.linalg.norm(v))
    _check(n > 0, "zero-length vector")
    return v / n


def _project(a, b):
    bb = float(b @ b)
    _check(bb > 0, "projection onto a zero vector")
    return b * (float(a @ b) / bb)


def _angle(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    _check(na > 0 and nb > 0, "angle with a zero vector")
    return math.acos(max(-1.0, min(1.0, float(a @ b) / (na * nb))))


def _inverse(m):
    _check(abs(float(np.linalg.det(m))) > 1e-300, "singular matrix")
    return np.linalg.inv(m)


def _quat_axis_angle(axis, angle):
    _check(float(np.linalg.norm(axis)) > 0, "zero rotation axis")
    return QuatValue(numerics.quat_from_axis_angle(axis, angle))


_IMPLS: dict[str, Callable] = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": _asin, "acos": _acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "exp": _exp, "log": _log,
    "log10": lambda x: _log(x, 10), "atan2": math.atan2, "sqrt": _sqrt, "abs": lambda x: abs(x) if isinstance(x, float)
    else np.abs(x), "sign": lambda x: float(np.sign(x)), "floor": lambda x: float(math.floor(x)),
    "ceil": lambda x: float(math.ceil(x)), "round": lambda x: float(round(x)),
    "min": lambda *a: min(a), "max": lambda *a: max(a), "clamp": lambda x, lo, hi: min(max(x, lo), hi),
    "hypot": math.hypot, "lerp": lambda a, b, s: a + (b - a) * s, "step": lambda x: 1.0 if x >= 0 else 0.0,
    "ramp": lambda x: x if x > 0 else 0.0 * x,
    "vec2": lambda x, y: np.array([x, y], float), "vec3": lambda x, y, z: np.array([x, y, z], float),
    "norm": lambda v: float(np.linalg.norm(v)), "mag": lambda v: float(np.linalg.norm(v)),
    "length": lambda v: float(np.linalg.norm(v)), "normalize": _normalize, "dot": lambda a, b: float(a @ b),
    "cross": lambda a, b: np.cross(a, b), "distance": lambda a, b: float(np.linalg.norm(a - b)),
    "project": _project, "angle_between": _angle,
    "transpose": lambda m: m.T, "inverse": _inverse, "det": lambda m: float(np.linalg.det(m)),
    "identity": lambda: np.eye(3), "diag": lambda v: np.diag(v), "outer": lambda a, b: np.outer(a, b),
    "rotate": lambda q, v: np.asarray(numerics.quat_rotate(q.q, v), float), "conj": lambda q: QuatValue(
        numerics.quat_conjugate(q.q)),
    "quat_axis_angle": _quat_axis_angle, "quat_from_euler": lambda e: QuatValue(numerics.quat_from_euler(e)),
    "to_euler": lambda q: np.asarray(numerics.quat_to_euler(q.q), float),
}


def finite(value) -> bool:
    if isinstance(value, QuatValue):
        return bool(np.all(np.isfinite(value.q)))
    if isinstance(value, (bool, int, float)):
        return math.isfinite(value) if not isinstance(value, bool) else True
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    return True


def compile_expr(node: Node, types: dict[int, FType], source: str) -> Fn:
    return Compiler(types, source).compile(node)

