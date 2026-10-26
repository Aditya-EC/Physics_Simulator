"""Static type checking and dimensional analysis for formulas.

Every AST node is annotated with an ``FType``: a value kind (number, 2D/3D
vector, 3×3 matrix, quaternion, Boolean, object, object set, string) and, for
numeric kinds, a physical dimension. Invalid operations are reported before
any simulation runs:

* incompatible kinds (e.g. vector + scalar, vector × vector, wrong arity);
* incompatible dimensions (e.g. length + time), transcendental functions of
  dimensioned arguments, outputs whose dimension does not match the target;
* unknown names, attributes, functions or undeclared custom properties.

Angles are treated as dimensionless inside formulas (SI convention), so
expressions such as ``cross(self.angular_velocity, r)`` have the dimension of
velocity. The literal ``0`` (without a unit) is dimension-polymorphic, so
``vec3(0, 0, 0)`` can initialise any vector quantity; any other number without
a unit annotation is dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..units import dimensions as D
from ..units.dimensions import Dimension
from .formula_parser import Attr, Binary, Call, Cond, FormulaSyntaxError, Index, Name, Node, Num, Str, Unary

NUMERIC = ("num", "vec2", "vec3", "mat3")
VECTORS = ("vec2", "vec3")


class FormulaTypeError(FormulaSyntaxError):
    pass


def formula_dim(d: Dimension) -> Dimension:
    """Dimension used in formulas: the angle pseudo-dimension is dropped (radians are dimensionless)."""
    e = list(d.exponents)
    e[7] = 0
    return Dimension(tuple(e))


DIMLESS = Dimension()


@dataclass(frozen=True)
class FType:
    kind: str
    dim: Dimension = DIMLESS
    const: float | None = None      # compile-time numeric value (literals)
    poly: bool = False              # literal zero: compatible with any dimension

    def describe(self) -> str:
        names = {"num": "scalar", "vec2": "2D vector", "vec3": "3D vector", "mat3": "3×3 matrix", "quat": "quaternion",
                 "bool": "Boolean", "obj": "object", "set": "object set", "str": "string"}
        base = names.get(self.kind, self.kind)
        if self.kind in NUMERIC and not self.poly:
            base += f" [{self.dim}]"
        return base


def T(kind: str, dim: Dimension = DIMLESS) -> FType:
    return FType(kind, formula_dim(dim))


NUM, VEC3, BOOL, OBJ, SET, QUAT, MAT3 = T("num"), T("vec3"), FType("bool"), FType("obj"), FType("set"), FType("quat"), \
    T("mat3")

# Object attributes visible to formulas (global frame unless stated).
OBJECT_ATTRS: dict[str, FType] = {
    "position": T("vec3", D.LENGTH),            # object origin
    "com": T("vec3", D.LENGTH),                 # centre of mass
    "velocity": T("vec3", D.VELOCITY),          # COM velocity
    "linear_velocity": T("vec3", D.VELOCITY),
    "angular_velocity": T("vec3", D.FREQUENCY),
    "orientation": QUAT,
    "euler": T("vec3"),
    "mass": T("num", D.MASS),
    "density": T("num", D.DENSITY),
    "volume": T("num", D.VOLUME),
    "inertia": T("mat3", D.INERTIA),            # about the COM, global axes
    "inertia_local": T("mat3", D.INERTIA),
    "com_local": T("vec3", D.LENGTH),
    "force": T("vec3", D.FORCE),                # applied force property (input)
    "torque": T("vec3", D.FORCE * D.LENGTH),    # applied torque property (input)
    "linear_acceleration": T("vec3", D.ACCELERATION),        # constant-acceleration input property
    "angular_acceleration": T("vec3", D.FREQUENCY / D.TIME),
    "position0": T("vec3", D.LENGTH),           # at the start of the run
    "orientation0": QUAT,
    "scale": T("vec3"),
    "x_axis": T("vec3"), "y_axis": T("vec3"), "z_axis": T("vec3"),
    "parent": OBJ,
    "children": SET,
}

# Output targets of formula processes.
OUTPUT_TARGETS: dict[str, FType] = {
    "force": T("vec3", D.FORCE),
    "torque": T("vec3", D.FORCE * D.LENGTH),
    "acceleration": T("vec3", D.ACCELERATION),
    "angular_acceleration": T("vec3", D.FREQUENCY / D.TIME),
    "position": T("vec3", D.LENGTH),
    "velocity": T("vec3", D.VELOCITY),
    "orientation": QUAT,
    "angular_velocity": T("vec3", D.FREQUENCY),
}
TARGETS_BY_KIND = {
    "contribution": {"force", "torque", "acceleration", "angular_acceleration"},
    "prescribed": {"position", "velocity", "orientation", "angular_velocity"},
    "update": {"position", "velocity", "orientation", "angular_velocity", "custom"},
}
UPDATABLE_ATTRS = {"position": "position", "velocity": "velocity", "linear_velocity": "velocity",
                   "orientation": "orientation", "angular_velocity": "angular_velocity"}

CONSTANTS: dict[str, tuple[float, FType]] = {
    "pi": (3.141592653589793, NUM),
    "e": (2.718281828459045, NUM),
    "g0": (9.80665, T("num", D.ACCELERATION)),
    "G": (6.6743e-11, T("num", D.LENGTH ** 3 / D.MASS / D.TIME ** 2)),
}


@dataclass
class CheckEnv:
    mode: str = "per_object"               # per_object | global
    kind: str = "contribution"             # contribution | prescribed | update
    params: dict[str, FType] = field(default_factory=dict)
    globals_: dict[str, FType] = field(default_factory=dict)
    custom: dict[str, FType] = field(default_factory=dict)
    definitions: dict[str, FType] = field(default_factory=dict)
    source: str = ""


class Checker:
    def __init__(self, env: CheckEnv) -> None:
        self.env = env
        self.types: dict[int, FType] = {}
        self.object_ids: set[str] = set()      # literal object("...") references, validated at plan time
        self.updated_refs: set[str] = set()    # quantities read through updated(...)
        self._it_depth = 0

    def error(self, msg: str, node: Node) -> FormulaTypeError:
        return FormulaTypeError(msg, node.pos, self.env.source)

    def check(self, node: Node) -> FType:
        t = self._check(node)
        self.types[id(node)] = t
        return t

    # ---- helpers -----------------------------------------------------------------------------------
    def _same_dim(self, a: FType, b: FType, node: Node, what: str) -> Dimension:
        if a.poly:
            return b.dim
        if b.poly or a.dim == b.dim:
            return a.dim
        raise self.error(f"Incompatible dimensions in {what}: [{a.dim}] and [{b.dim}]", node)

    def _need(self, t: FType, kinds: tuple[str, ...] | str, node: Node, what: str) -> None:
        kinds = (kinds,) if isinstance(kinds, str) else kinds
        if t.kind not in kinds:
            names = " or ".join(FType(k).describe().split(" [")[0] for k in kinds)
            raise self.error(f"{what} expects a {names}, got a {t.describe()}", node)

    def _dimless(self, t: FType, node: Node, what: str) -> None:
        if not t.poly and not t.dim.is_dimensionless:
            raise self.error(f"{what} requires a dimensionless argument, got [{t.dim}]", node)

    # ---- nodes ----------------------------------------------------------------------------------------
    def _check(self, n: Node) -> FType:
        if isinstance(n, Num):
            return FType("num", formula_dim(n.dim), n.value, poly=n.value == 0 and not n.unit_text)
        if isinstance(n, Str):
            return FType("str")
        if isinstance(n, Name):
            return self._name(n)
        if isinstance(n, Attr):
            return self._attr(n)
        if isinstance(n, Index):
            return self._index(n)
        if isinstance(n, Call):
            return self._call(n)
        if isinstance(n, Unary):
            t = self.check(n.operand)
            if n.op == "!":
                self._need(t, "bool", n, "'not'")
                return BOOL
            self._need(t, NUMERIC + ("quat",), n, "Negation")
            return replace(t, const=-t.const if t.const is not None else None)
        if isinstance(n, Cond):
            c = self.check(n.test)
            self._need(c, "bool", n.test, "The condition")
            a, b = self.check(n.then), self.check(n.other)
            return self._merge(a, b, n, "the two branches of a conditional")
        if isinstance(n, Binary):
            return self._binary(n)
        raise self.error("Unsupported expression", n)

    def _merge(self, a: FType, b: FType, n: Node, what: str) -> FType:
        if a.kind != b.kind:
            raise self.error(f"Incompatible types in {what}: {a.describe()} and {b.describe()}", n)
        if a.kind in NUMERIC:
            return FType(a.kind, self._same_dim(a, b, n, what), poly=a.poly and b.poly)
        return a

    def _name(self, n: Name) -> FType:
        env = self.env
        if n.id in env.definitions:
            return env.definitions[n.id]
        if n.id in ("t", "dt"):
            return T("num", D.TIME)
        if n.id in ("true", "false"):
            return BOOL
        if n.id in CONSTANTS:
            return CONSTANTS[n.id][1]
        if n.id == "self":
            if env.mode != "per_object":
                raise self.error("'self' is only available in per-object processes; use targets[i]", n)
            return OBJ
        if n.id == "targets":
            return SET
        if n.id == "others":
            if env.mode != "per_object":
                raise self.error("'others' is only available in per-object processes", n)
            return SET
        if n.id == "it":
            if not self._it_depth:
                raise self.error("'it' is only defined inside sum_of / mean_of / min_of / max_of", n)
            return OBJ
        if n.id in ("param", "global"):
            raise self.error(f"Use {n.id}.<name> to read a {'parameter' if n.id == 'param' else 'global parameter'}", n)
        raise self.error(f"Unknown name '{n.id}'", n)

    def _attr(self, n: Attr) -> FType:
        if isinstance(n.obj, Name) and n.obj.id == "param":
            if n.name not in self.env.params:
                raise self.error(f"Unknown parameter '{n.name}'", n)
            return self.env.params[n.name]
        if isinstance(n.obj, Name) and n.obj.id == "global":
            if n.name not in self.env.globals_:
                raise self.error(f"Global parameter '{n.name}' is not declared by this process", n)
            return self.env.globals_[n.name]
        if isinstance(n.obj, Attr) and n.obj.name == "custom":
            base = self.check(n.obj.obj)
            self._need(base, "obj", n.obj, "'.custom'")
            if n.name not in self.env.custom:
                raise self.error(f"Custom property '{n.name}' is not declared by this process", n)
            return self.env.custom[n.name]
        t = self.check(n.obj)
        if t.kind == "obj":
            if n.name == "custom":
                raise self.error("Use .custom.<key> to read a custom property", n)
            if n.name not in OBJECT_ATTRS:
                raise self.error(f"Objects have no attribute '{n.name}'", n)
            return OBJECT_ATTRS[n.name]
        if t.kind in VECTORS and n.name in ("x", "y", "z")[: 2 if t.kind == "vec2" else 3]:
            return FType("num", t.dim, poly=t.poly)
        if t.kind == "quat" and n.name in ("w", "x", "y", "z"):
            return NUM
        raise self.error(f"A {t.describe()} has no attribute '{n.name}'", n)

    def _index(self, n: Index) -> FType:
        t = self.check(n.obj)
        i = self.check(n.index)
        self._need(i, "num", n.index, "An index")
        self._dimless(i, n.index, "An index")
        if t.kind == "set":
            return OBJ
        if t.kind in VECTORS:
            return FType("num", t.dim, poly=t.poly)
        raise self.error(f"A {t.describe()} cannot be indexed", n)

    def _binary(self, n: Binary) -> FType:
        a, b = self.check(n.left), self.check(n.right)
        op = n.op
        if op in ("&&", "||"):
            self._need(a, "bool", n.left, f"'{op}'")
            self._need(b, "bool", n.right, f"'{op}'")
            return BOOL
        if op in ("==", "!=", "<", "<=", ">", ">="):
            if op in ("==", "!=") and a.kind == b.kind == "obj":
                return BOOL
            self._need(a, "num", n.left, "A comparison")
            self._need(b, "num", n.right, "A comparison")
            self._same_dim(a, b, n, "a comparison")
            return BOOL
        if op in ("+", "-"):
            if a.kind != b.kind or a.kind not in NUMERIC:
                raise self.error(f"Cannot apply '{op}' to a {a.describe()} and a {b.describe()}", n)
            const = None
            if a.const is not None and b.const is not None:
                const = a.const + b.const if op == "+" else a.const - b.const
            return FType(a.kind, self._same_dim(a, b, n, f"'{op}'"), const, poly=a.poly and b.poly)
        if op == "*":
            if a.kind == "quat" and b.kind == "quat":
                return QUAT
            if a.kind == "num" and b.kind in NUMERIC or b.kind == "num" and a.kind in NUMERIC:
                kind = b.kind if a.kind == "num" else a.kind
                const = a.const * b.const if a.const is not None and b.const is not None else None
                return FType(kind, a.dim * b.dim, const, poly=a.poly or b.poly)
            if a.kind == "mat3" and b.kind in ("vec3", "mat3"):
                return FType(b.kind, a.dim * b.dim)
            raise self.error(f"Cannot multiply a {a.describe()} by a {b.describe()}"
                             + (" (use dot() or cross())" if a.kind in VECTORS and b.kind in VECTORS else ""), n)
        if op in ("/", "%"):
            if b.kind != "num" or a.kind not in NUMERIC:
                raise self.error(f"Cannot divide a {a.describe()} by a {b.describe()}", n)
            if op == "%":
                return FType(a.kind, self._same_dim(a, b, n, "'%'"))
            if b.const == 0:
                raise self.error("Division by the constant zero", n)
            const = a.const / b.const if a.const is not None and b.const is not None else None
            return FType(a.kind, a.dim / b.dim, const, poly=a.poly)
        if op == "^":
            self._need(a, "num", n.left, "The base of '^'")
            self._need(b, "num", n.right, "The exponent of '^'")
            self._dimless(b, n.right, "An exponent")
            if a.dim.is_dimensionless or a.poly:
                return FType("num", DIMLESS)
            if b.const is None:
                raise self.error("A dimensioned value can only be raised to a constant power", n)
            exps = [x * b.const for x in a.dim.exponents]
            if any(abs(x - round(x)) > 1e-9 for x in exps):
                raise self.error(f"[{a.dim}]^{b.const:g} does not have integer dimension exponents", n)
            return FType("num", Dimension(tuple(int(round(x)) for x in exps)))
        raise self.error(f"Unknown operator '{op}'", n)

    # ---- functions ---------------------------------------------------------------------------------------
    def _call(self, n: Call) -> FType:
        f = n.func
        if f in AGGREGATES:
            return self._aggregate(n)
        if f == "updated":
            return self._updated(n)
        if f in ("object", "tagged", "of_type"):
            if len(n.args) != 1 or not isinstance(n.args[0], Str):
                raise self.error(f"{f}() takes one string literal argument", n)
            self.types[id(n.args[0])] = FType("str")
            if f == "object":
                self.object_ids.add(n.args[0].value)
                return OBJ
            return SET
        args = [self.check(a) for a in n.args]
        if f not in FUNCTIONS:
            raise self.error(f"Unknown function '{f}'", n)
        arity, rule = FUNCTIONS[f]
        if arity is not None and len(args) not in (arity if isinstance(arity, tuple) else (arity,)):
            raise self.error(f"{f}() takes {arity} argument(s), got {len(args)}", n)
        return rule(self, n, args)

    def _aggregate(self, n: Call) -> FType:
        f = n.func
        if f == "count":
            if len(n.args) != 1:
                raise self.error("count() takes one object set", n)
            self._need(self.check(n.args[0]), "set", n.args[0], "count()")
            return NUM
        if len(n.args) != 2:
            raise self.error(f"{f}(set, expression) takes two arguments", n)
        self._need(self.check(n.args[0]), "set", n.args[0], f"{f}()")
        self._it_depth += 1
        try:
            t = self.check(n.args[1])
        finally:
            self._it_depth -= 1
        allowed = ("num",) if f in ("min_of", "max_of") else ("num", "vec2", "vec3", "mat3")
        self._need(t, allowed, n.args[1], f"{f}()")
        return FType(t.kind, t.dim, poly=t.poly)

    def _updated(self, n: Call) -> FType:
        if self.env.kind != "update":
            raise self.error("updated() is only available in update processes", n)
        if len(n.args) != 1 or not isinstance(n.args[0], Attr):
            raise self.error("updated() takes an object quantity, e.g. updated(self.velocity)", n)
        a = n.args[0]
        t = self.check(a)
        if isinstance(a.obj, Attr) and a.obj.name == "custom":
            self.updated_refs.add(f"custom.{a.name}")
        elif a.name in UPDATABLE_ATTRS:
            self.updated_refs.add(UPDATABLE_ATTRS[a.name])
        else:
            raise self.error(f"'{a.name}' is not a quantity that update processes write", a)
        return t


def _num_same(name):
    def rule(c: Checker, n, args):
        for a in args:
            c._need(a, "num", n, f"{name}()")
        d = args[0]
        for a in args[1:]:
            d = FType("num", c._same_dim(d, a, n, f"{name}()"), poly=d.poly and a.poly)
        return FType("num", d.dim, poly=d.poly)
    return rule


def _transcendental(name):
    def rule(c: Checker, n, args):
        c._need(args[0], "num", n, f"{name}()")
        c._dimless(args[0], n, f"{name}()")
        return NUM
    return rule


def _atan2(c, n, args):
    _num_same("atan2")(c, n, args)  # y and x must share a dimension
    return NUM


def _sign(c, n, args):
    c._need(args[0], "num", n, "sign()")
    return NUM


def _sqrt(c, n, args):
    c._need(args[0], "num", n, "sqrt()")
    if any(e % 2 for e in args[0].dim.exponents):
        raise c.error(f"sqrt() of [{args[0].dim}] has no integer-exponent dimension", n)
    return FType("num", Dimension(tuple(e // 2 for e in args[0].dim.exponents)))


def _abs(c, n, args):
    c._need(args[0], ("num", "vec2", "vec3"), n, "abs()")
    return FType(args[0].kind, args[0].dim, poly=args[0].poly)


def _vec_ctor(kind):
    def rule(c, n, args):
        for a in args:
            c._need(a, "num", n, f"{kind}()")
        d = args[0]
        for a in args[1:]:
            d = FType("num", c._same_dim(d, a, n, f"{kind}()"), poly=d.poly and a.poly)
        return FType(kind, d.dim, poly=d.poly)
    return rule


def _norm(c, n, args):
    c._need(args[0], VECTORS, n, "norm()")
    return FType("num", args[0].dim)


def _normalize(c, n, args):
    c._need(args[0], VECTORS, n, "normalize()")
    return FType(args[0].kind)


def _dot(c, n, args):
    a, b = args
    c._need(a, VECTORS, n, "dot()")
    if a.kind != b.kind:
        raise c.error("dot() needs two vectors of the same size", n)
    return FType("num", a.dim * b.dim)


def _cross(c, n, args):
    for a in args:
        c._need(a, "vec3", n, "cross()")
    return FType("vec3", args[0].dim * args[1].dim)


def _distance(c, n, args):
    for a in args:
        c._need(a, VECTORS, n, "distance()")
    return FType("num", c._same_dim(args[0], args[1], n, "distance()"))


def _project(c, n, args):
    for a in args:
        c._need(a, VECTORS, n, "project()")
    return FType(args[0].kind, args[0].dim)


def _angle(c, n, args):
    for a in args:
        c._need(a, VECTORS, n, "angle_between()")
    return NUM


def _clamp(c, n, args):
    return _num_same("clamp")(c, n, args)


def _lerp(c, n, args):
    a, b, s = args
    t = c._merge(a, b, n, "lerp()")
    c._need(s, "num", n, "lerp()")
    c._dimless(s, n, "The lerp() fraction")
    return t


def _if(c, n, args):
    c._need(args[0], "bool", n, "if()")
    return c._merge(args[1], args[2], n, "the branches of if()")


def _step(c, n, args):
    c._need(args[0], "num", n, "step()")
    return NUM


def _ramp(c, n, args):
    c._need(args[0], "num", n, "ramp()")
    return FType("num", args[0].dim)


def _mat_unary(name):
    def rule(c, n, args):
        c._need(args[0], "mat3", n, f"{name}()")
        d = args[0].dim
        if name == "inverse":
            d = DIMLESS / d
        elif name == "det":
            return FType("num", d ** 3)
        return FType("mat3", d)
    return rule


def _identity(c, n, args):
    return MAT3


def _diag(c, n, args):
    c._need(args[0], "vec3", n, "diag()")
    return FType("mat3", args[0].dim)


def _outer(c, n, args):
    for a in args:
        c._need(a, "vec3", n, "outer()")
    return FType("mat3", args[0].dim * args[1].dim)


def _rotate(c, n, args):
    c._need(args[0], "quat", n, "rotate()")
    c._need(args[1], "vec3", n, "rotate()")
    return FType("vec3", args[1].dim, poly=args[1].poly)


def _conj(c, n, args):
    c._need(args[0], "quat", n, "conj()")
    return QUAT


def _quat_axis_angle(c, n, args):
    c._need(args[0], "vec3", n, "quat_axis_angle()")
    c._need(args[1], "num", n, "quat_axis_angle()")
    c._dimless(args[1], n, "The angle of quat_axis_angle()")
    return QUAT


def _quat_from_euler(c, n, args):
    c._need(args[0], "vec3", n, "quat_from_euler()")
    c._dimless(args[0], n, "quat_from_euler()")
    return QUAT


def _to_euler(c, n, args):
    c._need(args[0], "quat", n, "to_euler()")
    return T("vec3")


def _frame(name):
    def rule(c, n, args):
        c._need(args[0], "obj", n, f"{name}()")
        c._need(args[1], "vec3", n, f"{name}()")
        if name.startswith("point_"):
            c._same_dim(args[1], T("vec3", D.LENGTH), n, f"{name}() (expects a position)")
        return FType("vec3", args[1].dim, poly=args[1].poly)
    return rule


def _set_of(name):
    def rule(c, n, args):
        if name == "all":
            return SET
        c._need(args[0], "obj", n, f"{name}()")
        return SET
    return rule


FUNCTIONS = {
    "sin": (1, _transcendental("sin")), "cos": (1, _transcendental("cos")), "tan": (1, _transcendental("tan")),
    "asin": (1, _transcendental("asin")), "acos": (1, _transcendental("acos")), "atan": (1, _transcendental("atan")),
    "sinh": (1, _transcendental("sinh")), "cosh": (1, _transcendental("cosh")), "tanh": (1, _transcendental("tanh")),
    "exp": (1, _transcendental("exp")), "log": (1, _transcendental("log")), "log10": (1, _transcendental("log10")),
    "atan2": (2, _atan2),
    "sqrt": (1, _sqrt), "abs": (1, _abs), "sign": (1, _sign),
    "floor": (1, _num_same("floor")), "ceil": (1, _num_same("ceil")), "round": (1, _num_same("round")),
    "min": ((2, 3, 4, 5, 6), _num_same("min")), "max": ((2, 3, 4, 5, 6), _num_same("max")),
    "clamp": (3, _clamp), "hypot": (2, _num_same("hypot")), "lerp": (3, _lerp), "if": (3, _if),
    "step": (1, _step), "ramp": (1, _ramp),
    "vec2": (2, _vec_ctor("vec2")), "vec3": (3, _vec_ctor("vec3")),
    "norm": (1, _norm), "mag": (1, _norm), "length": (1, _norm), "normalize": (1, _normalize),
    "dot": (2, _dot), "cross": (2, _cross), "distance": (2, _distance), "project": (2, _project),
    "angle_between": (2, _angle),
    "transpose": (1, _mat_unary("transpose")), "inverse": (1, _mat_unary("inverse")), "det": (1, _mat_unary("det")),
    "identity": (0, _identity), "diag": (1, _diag), "outer": (2, _outer),
    "rotate": (2, _rotate), "conj": (1, _conj), "quat_axis_angle": (2, _quat_axis_angle),
    "quat_from_euler": (1, _quat_from_euler), "to_euler": (1, _to_euler),
    "to_local": (2, _frame("to_local")), "to_global": (2, _frame("to_global")),
    "point_to_local": (2, _frame("point_to_local")), "point_to_global": (2, _frame("point_to_global")),
    "all": (0, _set_of("all")), "children": (1, _set_of("children")),
}
AGGREGATES = {"sum_of", "mean_of", "min_of", "max_of", "count"}

FUNCTION_DOCS = {
    "Math": "sin cos tan asin acos atan atan2 sinh cosh tanh exp log log10 sqrt abs sign floor ceil round "
            "min max clamp hypot lerp step ramp if",
    "Vectors": "vec2 vec3 norm|mag|length normalize dot cross distance project angle_between, v.x v.y v.z, v[i]",
    "Matrices": "identity diag outer transpose inverse det, M * v, M * N",
    "Rotations": "rotate(q, v) conj quat_axis_angle quat_from_euler to_euler, q1 * q2",
    "Frames": "to_local(obj, v) to_global(obj, v) point_to_local(obj, p) point_to_global(obj, p)",
    "Objects": "self, object(\"OBJ-000001\"), targets[i], others, it, parent, children, all(), tagged(\"tag\"), "
               "of_type(\"basic.sphere\"), obj.custom.<key>",
    "Aggregates": "sum_of(set, expr) mean_of(set, expr) min_of(set, expr) max_of(set, expr) count(set)",
    "Inputs": "t, dt, param.<key>, global.<name>, updated(obj.quantity) (update processes)",
    "Constants": "pi, e, g0 = 9.80665 m/s², G = 6.6743e-11 m³/(kg·s²), true, false; units: 9.81[m/s^2], 30[deg]",
}
