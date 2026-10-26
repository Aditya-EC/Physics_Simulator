"""Formula language: parsing, type checking, dimensional analysis, evaluation and scripts."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim.processes.dimensional_analysis import CheckEnv, Checker, FormulaTypeError, T
from psim.processes.formula_parser import Binary, FormulaSyntaxError, Num, parse, parse_unit
from psim.processes.formula_process import FormulaProcessSpec, MockWorld, compile_spec, evaluate_outputs, run_tests
from psim.processes.formula_runtime import EvalContext, FormulaRuntimeError, QuatValue, compile_expr
from psim.units import dimensions as D


def check(src, **env):
    e = CheckEnv(source=src, **env)
    c = Checker(e)
    node = parse(src)
    return c.check(node), node, c


def evaluate(src, objects=None, params=None, t=0.0, dt=0.01, param_types=None, **env):
    ftype, node, checker = check(src, params=param_types or {}, **env)
    fn = compile_expr(node, checker.types, src)
    world = MockWorld(objects or {"self": {}}, {}, dt)
    return fn(EvalContext(world, 0, (0,), t, dt, params or {}, {}))


# ---- parser -----------------------------------------------------------------------------------------------

def test_precedence_and_associativity():
    assert evaluate("1 + 2 * 3 ^ 2") == 19
    assert evaluate("2 ^ 3 ^ 2") == 512          # right associative
    assert evaluate("-2 ^ 2") == -4              # exponentiation binds tighter than negation
    assert evaluate("10 - 4 - 3") == 3
    assert evaluate("1 < 2 and not (3 < 2) or false") is True
    assert evaluate("2 > 1 ? 10 : 20") == 10


def test_unit_literals():
    n = parse("9.81[m/s^2]")
    assert isinstance(n, Num) and n.dim == D.ACCELERATION and n.value == pytest.approx(9.81)
    assert parse("36[km/h]").value == pytest.approx(10.0)
    assert parse("90[deg]").value == pytest.approx(math.pi / 2)
    f, d = parse_unit("kg*m^2/s")
    assert f == 1.0 and d == D.ANGULAR_MOMENTUM / D.ANGLE
    assert parse_unit("N·m")[1] == D.TORQUE
    assert isinstance(parse("2[m] + 3[cm]"), Binary)


@pytest.mark.parametrize("src,msg", [
    ("1 +", "Unexpected end"), ("(1", "Expected ')'"), ("a b", "Unexpected 'b'"), ("3[furlongs]", "Unknown unit"),
    ("1 < 2 < 3", "Chained comparisons"), ("x.1", "Unexpected '.1'"), ("@", "Unexpected character"),
    ("f(1)(2)", "Only named functions"), ("(" * 2000 + "1" + ")" * 2000, "nested too deeply"),
])
def test_syntax_errors(src, msg):
    with pytest.raises(FormulaSyntaxError, match=msg.replace("(", r"\(").replace(")", r"\)")):
        parse(src)


def test_no_python_evaluation():
    """The language has no access to Python: attribute names like __class__ are unknown attributes."""
    for src in ("self.__class__", "__import__(\"os\")", "self.mass.__class__", "open(\"x\")"):
        with pytest.raises((FormulaTypeError, FormulaSyntaxError)):
            check(src)


# ---- types & dimensions ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("src,kind,dim", [
    ("self.mass * self.velocity", "vec3", D.MOMENTUM),
    ("0.5 * self.mass * dot(self.velocity, self.velocity)", "num", D.ENERGY),
    ("cross(self.angular_velocity, self.position)", "vec3", D.VELOCITY),     # angles are dimensionless
    ("self.inertia * self.angular_velocity", "vec3", D.MASS * D.AREA / D.TIME),
    ("sqrt(dot(self.velocity, self.velocity))", "num", D.VELOCITY),
    ("norm(self.position)^3", "num", D.VOLUME),
    ("G * self.mass / norm(self.position)^2", "num", D.ACCELERATION),
    ("vec3(0, 0, 0)", "vec3", D.DIMENSIONLESS),
    ("self.position + vec3(0, 0, 0)", "vec3", D.LENGTH),
    ("sin(2 * pi * 2[Hz] * t)", "num", D.DIMENSIONLESS),
    ("atan2(self.position.y, self.position.x)", "num", D.DIMENSIONLESS),
    ("to_local(self, self.velocity)", "vec3", D.VELOCITY),
    ("t > 1[s] ? self.force : vec3(0, 0, 0)", "vec3", D.FORCE),
    ("count(others)", "num", D.DIMENSIONLESS),
    ("sum_of(others, it.mass)", "num", D.MASS),
    ("rotate(self.orientation, vec3(1[m], 0[m], 0[m]))", "vec3", D.LENGTH),
    ("det(self.inertia)", "num", (D.MASS * D.AREA) ** 3),
])
def test_dimension_inference(src, kind, dim):
    from psim.processes.dimensional_analysis import formula_dim
    t, _, _ = check(src)
    assert t.kind == kind
    if not t.poly:
        assert t.dim == formula_dim(dim)


@pytest.mark.parametrize("src,msg", [
    ("self.position + self.velocity", "Incompatible dimensions"),
    ("self.position + 1", "Cannot apply '\\+'"),
    ("self.velocity * self.velocity", "use dot\\(\\) or cross\\(\\)"),
    ("sin(t)", "dimensionless argument"),
    ("exp(self.mass)", "dimensionless argument"),
    ("sqrt(self.mass)", "integer-exponent"),
    ("norm(self.position)^1.5", "integer dimension"),
    ("self.mass ^ t", "dimensionless"),
    ("self.mass > 1[m]", "Incompatible dimensions in a comparison"),
    ("vec3(1[m], 2[s], 0)", "Incompatible dimensions"),
    ("cross(vec2(1, 2), vec2(3, 4))", "expects a 3D vector"),
    ("self.nonexistent", "no attribute"),
    ("param.k", "Unknown parameter"),
    ("global.rho", "not declared"),
    ("self.custom.ke", "not declared"),
    ("unknown_name", "Unknown name"),
    ("it.mass", "only defined inside"),
    ("updated(self.velocity)", "only available in update"),
    ("1 / 0", "Division by the constant zero"),
    ("t ? 1 : 2", "condition"),
    ("dot(self.velocity)", "takes 2 argument"),
])
def test_type_and_dimension_errors(src, msg):
    with pytest.raises(FormulaTypeError, match=msg):
        check(src)


def test_self_only_in_per_object_mode():
    with pytest.raises(FormulaTypeError, match="per-object"):
        check("self.mass", mode="global")
    t, _, _ = check("targets[0].mass + targets[1].mass", mode="global")
    assert t.dim == D.MASS


# ---- runtime -------------------------------------------------------------------------------------------------

def test_vector_matrix_quaternion_evaluation():
    objs = {"self": {"velocity": [3, 4, 0], "mass": 2, "orientation": [math.cos(math.pi / 4), 0, 0,
                                                                         math.sin(math.pi / 4)]}}
    assert evaluate("norm(self.velocity)", objs) == 5.0
    np.testing.assert_allclose(evaluate("normalize(self.velocity)", objs), [0.6, 0.8, 0])
    np.testing.assert_allclose(evaluate("rotate(self.orientation, vec3(1, 0, 0))", objs), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(evaluate("self.x_axis", objs), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(evaluate("to_local(self, vec3(0, 1, 0))", objs), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(evaluate("inverse(diag(vec3(2, 4, 8))) * vec3(2, 4, 8)", objs), [1, 1, 1])
    np.testing.assert_allclose(evaluate("to_euler(quat_axis_angle(vec3(0, 0, 1), pi / 2))"), [0, 0, math.pi / 2])
    q = evaluate("conj(self.orientation) * self.orientation", objs)
    assert isinstance(q, QuatValue)
    np.testing.assert_allclose(q.q, [1, 0, 0, 0], atol=1e-12)
    assert evaluate("project(vec3(1, 1, 0), vec3(1, 0, 0)).x") == 1.0
    assert evaluate("angle_between(vec3(1, 0, 0), vec3(0, 1, 0))") == pytest.approx(math.pi / 2)
    assert evaluate("clamp(5, 0, 3) + min(1, 2, -4) + max(2, 7)") == 6.0


def test_aggregates_and_object_sets():
    objs = {"A": {"mass": 1.0, "tags": ["x"]}, "B": {"mass": 2.0, "tags": ["x"]}, "C": {"mass": 4.0}}
    world = MockWorld(objs, {}, 0.01)
    for src, want in (("sum_of(all(), it.mass)", 7.0), ("mean_of(tagged(\"x\"), it.mass)", 1.5),
                      ("max_of(others, it.mass)", 4.0), ("count(of_type(\"basic.cube\"))", 3.0),
                      ("object(\"C\").mass", 4.0)):
        t, node, c = check(src)
        fn = compile_expr(node, c.types, src)
        assert fn(EvalContext(world, 0, (0, 1, 2), 0.0, 0.01, {}, {})) == want, src


@pytest.mark.parametrize("src,msg", [
    ("self.mass / (self.mass - 1[kg])", "Division by zero"),
    ("sqrt(-1)", "negative argument"),
    ("log(0)", "non-positive"),
    ("asin(2)", "outside"),
    ("normalize(vec3(0, 0, 0))", "zero-length"),
    ("object(\"nope\").mass", "Missing object reference"),
    ("targets[3].mass", "out of range"),
    ("min_of(tagged(\"none\"), it.mass)", "empty object set"),
])
def test_runtime_errors(src, msg):
    with pytest.raises(FormulaRuntimeError, match=msg):
        evaluate(src)


# ---- scripts --------------------------------------------------------------------------------------------------

def spec(**kw) -> FormulaProcessSpec:
    base = dict(process_id="test.proc", name="Test", script="force = vec3(0, 0, 0)")
    base.update(kw)
    return FormulaProcessSpec(**base)


def test_script_compiles_and_runs():
    s = spec(parameters=[{"key": "c", "type": "float", "quantity": "damping", "default": 2.0}],
             script="# viscous\nlet v = self.velocity\nrequire self.mass > 0[kg] : \"mass\"\nforce = -param.c * v\n",
             tests=[{"name": "t1", "objects": {"self": {"velocity": [1, 2, 3]}}, "expect": {"force": [-2, -4, -6]}}])
    cp, issues = compile_spec(s)
    assert cp is not None, issues
    assert cp.writes == {"force"} and "velocity" in cp.reads
    assert all(r.passed for r in run_tests(cp))


@pytest.mark.parametrize("kw,msg", [
    (dict(script="force = self.velocity"), "needs a 3D vector"),
    (dict(script="position = self.position"), "cannot be written by a contribution"),
    (dict(script="let x = 1\nlet x = 2\nforce = vec3(0,0,0)"), "already defined"),
    (dict(script="let t = 1\nforce = vec3(0,0,0)"), "reserved"),
    (dict(script="force = vec3(0,0,0)\nforce = vec3(0,0,0)"), "more than once"),
    (dict(script="require 1 : \"x\"\nforce = vec3(0,0,0)"), "must be a Boolean"),
    (dict(script="frobnicate"), "Expected a statement"),
    (dict(script="# only a comment"), "no output"),
    (dict(process_id="Bad Id"), "namespaced"),
    (dict(process_id="dyn.mine"), "reserved"),
    (dict(version="v1"), "Version"),
    (dict(kind="prescribed", script="velocity = vec3(0[m/s],0[m/s],0[m/s])"), "requires a prescribed position"),
    (dict(mode="global", script="force = vec3(0,0,0)"), "targets\\[i\\]"),
    (dict(kind="contribution", after=["x.y"]), "update processes only"),
    (dict(kind="update", script="custom.e = 1"), "not declared"),
])
def test_script_validation_errors(kw, msg):
    cp, issues = compile_spec(spec(**kw))
    assert cp is None
    assert any(__import__("re").search(msg, str(i)) for i in issues), [str(i) for i in issues]


def test_validation_reports_line_and_column():
    cp, issues = compile_spec(spec(script="let a = 1\nforce = self.velocity * 2[kg/s] + self.mass"))
    assert cp is None
    assert issues[0].line == 2 and issues[0].column > 1


def test_test_case_failures_are_reported():
    cp, _ = compile_spec(spec(tests=[{"name": "wrong", "objects": {"self": {}}, "expect": {"force": [1, 0, 0]}},
                                     {"name": "missing", "objects": {"self": {}}, "expect": {"torque": [0, 0, 0]}}]))
    res = run_tests(cp)
    assert not res[0].passed and "expected" in res[0].message
    assert not res[1].passed and "not produced" in res[1].message


def test_update_process_with_updated_values():
    cp, issues = compile_spec(spec(kind="update", script="position = self.position + updated(self.velocity) * dt"))
    assert cp is not None, issues
    assert cp.updated_refs == {"velocity"}
    world = MockWorld({"self": {"position": [0, 0, 1]}}, {}, 0.5)
    world.updated_values[(0, "velocity")] = np.array([2.0, 0, 0])
    out = evaluate_outputs(cp, world, 0, (0,), 0.0, 0.5, {}, {})
    np.testing.assert_allclose(out[0][2], [1, 0, 1])


def test_global_parameters_in_scripts():
    s = spec(globals=[{"name": "rho", "type": "float", "quantity": "density"}],
             script="force = -global.rho * 1[m^3/s] * self.velocity")
    cp, issues = compile_spec(s)
    assert cp is not None, issues
    world = MockWorld({"self": {"velocity": [1, 0, 0]}}, {"rho": 2.0}, 0.01)
    out = evaluate_outputs(cp, world, 0, (0,), 0.0, 0.01, {}, {"rho": 2.0})
    np.testing.assert_allclose(out[0][2], [-2, 0, 0])
    assert compile_spec(spec(script="force = -global.rho * self.velocity"))[0] is None  # undeclared global


def test_parameter_types_and_vectors():
    s = spec(parameters=[{"key": "f", "type": "vec3", "quantity": "force", "default": [1, 2, 3]},
                         {"key": "on", "type": "bool", "default": True}],
             script="force = param.on ? param.f : vec3(0, 0, 0)",
             tests=[{"name": "default", "objects": {"self": {}}, "expect": {"force": [1, 2, 3]}},
                    {"name": "off", "objects": {"self": {}}, "params": {"on": False}, "expect": {"force": [0, 0, 0]}}])
    cp, issues = compile_spec(s)
    assert cp is not None, issues
    assert all(r.passed for r in run_tests(cp))
    T_force = T("vec3", D.FORCE)
    assert T_force.kind == "vec3"
