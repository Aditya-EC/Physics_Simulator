"""Formula processes in the scheduler and engine, the process library, permissions and signed packages."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from psim.processes.formula_process import FormulaProcessSpec, build_process_def, compile_spec
from psim.processes.packages import canonical_bytes, read_package
from psim.processes.permissions import AccessController, PermissionDenied
from psim.processes.process_base import ProcessAssignment, ProcessError, TargetFilter
from psim.processes.process_registry import ProcessRegistry, process_registry
from psim.processes.process_scheduler import build_plan
from psim.properties.custom_properties import CustomProperty
from psim.simulation.settings import SimulationSettings
from psim.simulation.simulation_engine import SimulationEngine, SimulationError
from psim.simulation.state import capture_bodies
from psim.simulation.world import EngineWorld, capture_static, global_values


def make_engine(scene, **settings):
    s, _ = scene
    plan = build_plan(s, s.assignments)
    assert plan.ok, [d.message for d in plan.errors]
    bodies, state = capture_bodies(s, plan.body_ids)
    world = EngineWorld(capture_static(s), plan.body_ids, global_values(s), settings.get("delta_time", 0.01))
    return SimulationEngine(SimulationSettings(**settings), plan, bodies, state, world)


def run(scene, **settings):
    eng = make_engine(scene, **settings)
    while eng.step():
        pass
    return eng


_injected: list[str] = []


@pytest.fixture(autouse=True)
def _remove_injected():
    yield
    reg = process_registry()
    while _injected:
        pid = _injected.pop()
        reg._defs.pop(pid, None)
        reg.sources.pop(pid, None)


def register_temp(spec_dict) -> str:
    """Register a formula process in the (sealed) application registry for one test, bypassing signing."""
    reg = process_registry()
    spec = FormulaProcessSpec.from_dict(spec_dict)
    cp, issues = compile_spec(spec)
    assert cp is not None, [str(i) for i in issues]
    reg._defs[spec.process_id] = build_process_def(cp)  # test-only injection
    reg.sources[spec.process_id] = "test"
    _injected.append(spec.process_id)
    return spec.process_id


def test_library_loaded_and_registry_sealed():
    reg = process_registry()
    assert reg.load_errors == []
    assert reg.sealed
    for pid in ("lib.quadratic_drag", "lib.nbody_gravity", "lib.periodic_force", "lib.tether", "lib.lissajous",
                "lib.euler_velocity_update", "lib.euler_position_update", "lib.kinetic_energy_monitor"):
        assert reg.get(pid).is_formula
        assert reg.sources[pid] == "library"


def test_discrete_update_rules_match_semi_implicit_euler(scene):
    s, e = scene
    a = e.add_new("body.point_mass", position=[0, 0, 10])
    a.linear_velocity = [2, 0, 3]
    a.linear_acceleration = [0, 0, -9.81]
    e.assign_process("lib.euler_position_update", TargetFilter(ids=[a.id]))  # assigned first: ordering is derived
    e.assign_process("lib.euler_velocity_update", TargetFilter(ids=[a.id]))
    plan = build_plan(s, s.assignments)
    assert [p.pdef.process_id for p in plan.updates] == ["lib.euler_velocity_update", "lib.euler_position_update"]
    discrete = run(scene, delta_time=0.01, duration=1.0).snapshot()
    s.assignments = []
    e.assign_process("kin.constant_acceleration", TargetFilter(ids=[a.id]))
    builtin = run(scene, delta_time=0.01, duration=1.0, integrator="semi_implicit_euler").snapshot()
    assert np.array_equal(discrete.origin, builtin.origin) and np.array_equal(discrete.v, builtin.v)


def test_discrete_rotation_updates(scene):
    s, e = scene
    c = e.add_new("basic.cube")
    c.angular_acceleration = [0, 0, 1.0]
    e.assign_process("lib.euler_angular_velocity_update", TargetFilter(ids=[c.id]))
    e.assign_process("lib.euler_orientation_update", TargetFilter(ids=[c.id]))
    sn = run(scene, delta_time=0.01, duration=1.0).snapshot()
    assert sn.w[0][2] == pytest.approx(1.0)
    angle = sum(0.01 * 0.01 * k for k in range(1, 101))  # Σ ω_{n+1} Δt
    assert 2 * math.atan2(sn.q[0][3], sn.q[0][0]) == pytest.approx(angle)


def test_kinetic_energy_monitor_writes_custom_property(scene):
    s, e = scene
    b = e.add_new("body.point_mass")
    b.mass_input = 2.0
    b.linear_velocity = [3, 0, 4]
    b.add_custom(CustomProperty.create(key="kinetic_energy", name="KE", vtype="float", quantity="energy"))
    e.assign_process("kin.free_motion", TargetFilter(ids=[b.id]))
    e.assign_process("lib.kinetic_energy_monitor", TargetFilter(ids=[b.id]))
    eng = make_engine(scene, delta_time=0.01, duration=0.1)
    eng.step()
    snap = eng.snapshot()
    assert snap.custom[b.id]["kinetic_energy"] == pytest.approx(25.0)
    other = e.add_new("body.point_mass")  # lacks the required custom property -> ineligible
    plan = build_plan(s, [ProcessAssignment("PRC-0100", "lib.kinetic_energy_monitor", TargetFilter(ids=[other.id]))])
    assert not plan.ok and "not eligible" in plan.errors[0].message


def test_nbody_gravity_circular_orbit(scene):
    s, e = scene
    m_sun, r = 1e12, 10.0
    v = math.sqrt(6.6743e-11 * m_sun / r)
    sun = e.add_new("body.point_mass")
    sun.mass_input = m_sun
    p = e.add_new("body.point_mass", position=[r, 0, 0])
    p.linear_velocity = [0, v, 0]
    e.assign_process("lib.nbody_gravity", TargetFilter(ids=[sun.id, p.id]))
    period = 2 * math.pi * r / v
    eng = run(scene, delta_time=period / 2000, duration=period, integrator="rk4")
    np.testing.assert_allclose(eng.snapshot().origin[1], [r, 0, 0], atol=1e-6)


def test_quadratic_drag_terminal_velocity(scene):
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 1000])
    ball.mass_input = 1.0
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    e.assign_process("lib.quadratic_drag", TargetFilter(ids=[ball.id]), {"rho": 1.2, "cd": 0.5, "area": 0.1})
    sn = run(scene, delta_time=0.01, duration=40.0, integrator="rk4").snapshot()
    vt = math.sqrt(2 * 9.80665 / (1.2 * 0.5 * 0.1))
    assert -sn.v[0][2] == pytest.approx(vt, rel=1e-6)


def test_global_tether_and_prescribed_lissajous(scene):
    s, e = scene
    anchor = e.add_new("body.point_mass", position=[0, 0, 1])
    bob = e.add_new("body.point_mass", position=[0.5, 0, 1])
    e.assign_process("lib.lissajous", TargetFilter(ids=[anchor.id]), {"center": [0, 0, 1], "ax": 0.3, "ay": 0.2})
    e.assign_process("lib.tether", TargetFilter(ids=[anchor.id, bob.id]), {"length": 1.0, "stiffness": 500})
    plan = build_plan(s, s.assignments)
    assert plan.ok and any("driven anchor" in d.message for d in plan.diagnostics)
    eng = run(scene, delta_time=0.002, duration=1.0, integrator="rk4")
    sn = eng.snapshot()
    i = eng.bodies.index[anchor.id]
    np.testing.assert_allclose(sn.origin[i], [0.3 * math.sin(1.0), 0.2 * math.sin(2.0), 1.0], atol=1e-12)
    np.testing.assert_allclose(sn.v[i], [0.3 * math.cos(1.0), 0.4 * math.cos(2.0), 0], atol=1e-5)  # numerical rate


def test_required_globals_are_checked(scene):
    s, e = scene
    pid = register_temp({"process_id": "test.global_wind", "name": "Wind", "kind": "contribution",
                         "requires_mass": True, "globals": [{"name": "wind", "type": "vec3", "quantity": "force"}],
                         "script": "force = global.wind"})
    o = e.add_new("body.point_mass")
    e.assign_process(pid, TargetFilter(ids=[o.id]))
    plan = build_plan(s, s.assignments)
    assert not plan.ok and "requires the global parameter 'wind'" in plan.errors[0].message
    e.set_global_parameters([{"name": "wind", "type": "vec3", "quantity": "velocity", "value": [1, 0, 0]}])
    assert "wrong quantity" in build_plan(s, s.assignments).errors[0].message
    e.set_global_parameters([{"name": "wind", "type": "vec3", "quantity": "force", "value": [2, 0, 0]}])
    sn = run(scene, delta_time=0.01, duration=0.01).snapshot()
    np.testing.assert_allclose(sn.acc[0], [2, 0, 0])
    e.stack.undo()
    assert s.global_parameter("wind")["quantity"] == "velocity"


def test_update_dependency_cycle_and_missing_writer(scene):
    s, e = scene
    a = register_temp({"process_id": "test.cyc_a", "name": "A", "kind": "update",
                       "script": "velocity = updated(self.angular_velocity) * 1[m]"})
    b = register_temp({"process_id": "test.cyc_b", "name": "B", "kind": "update",
                       "script": "angular_velocity = updated(self.velocity) / 1[m]"})
    o = e.add_new("body.point_mass")
    e.assign_process(a, TargetFilter(ids=[o.id]))
    plan = build_plan(s, s.assignments)
    assert any("no active update process writes it" in d.message for d in plan.errors)
    e.assign_process(b, TargetFilter(ids=[o.id]))
    plan = build_plan(s, s.assignments)
    assert any("dependency cycle" in d.message for d in plan.errors)


def test_update_conflicts_with_integration(scene):
    s, e = scene
    o = e.add_new("body.point_mass")
    e.assign_process("lib.euler_velocity_update", TargetFilter(ids=[o.id]))
    e.assign_process("dyn.gravity", TargetFilter(ids=[o.id]))
    plan = build_plan(s, s.assignments)
    assert not plan.ok and "Conflicting writes" in plan.errors[0].message
    s.assignments = s.assignments[:1]
    assert any("will not move" in d.message for d in build_plan(s, s.assignments).diagnostics)


def test_missing_object_reference_in_formula(scene):
    s, e = scene
    pid = register_temp({"process_id": "test.follow", "name": "Follow", "kind": "contribution",
                         "requires_mass": True,
                         "script": "force = (object(\"OBJ-000777\").position - self.position) * 1[N/m]"})
    o = e.add_new("body.point_mass")
    e.assign_process(pid, TargetFilter(ids=[o.id]))
    plan = build_plan(s, s.assignments)
    assert any("missing object reference 'OBJ-000777'" in d.message for d in plan.errors)


def test_runtime_error_policies(scene):
    s, e = scene
    stop = register_temp({"process_id": "test.inv_stop", "name": "Inverse", "kind": "contribution",
                          "requires_mass": True, "script": "force = vec3(1[N*m], 0[N*m], 0[N*m]) / self.position.x"})
    skip = register_temp({"process_id": "test.inv_skip", "name": "Inverse (skip)", "kind": "contribution",
                          "requires_mass": True, "error_policy": "skip",
                          "script": "require self.position.x > 0[m] : \"x must be positive\"\nforce = vec3(1[N], 0[N], 0[N])"})
    o = e.add_new("body.point_mass")
    e.assign_process(stop, TargetFilter(ids=[o.id]))
    with pytest.raises(Exception, match="Division by zero"):
        run(scene, delta_time=0.01, duration=0.05)
    s.assignments = []
    e.assign_process(skip, TargetFilter(ids=[o.id]))
    eng = make_engine(scene, delta_time=0.01, duration=0.05)
    eng.step()
    snap = eng.snapshot()
    assert snap.warnings and "x must be positive" in snap.warnings[0]
    np.testing.assert_allclose(snap.acc[0], [0, 0, 0])


def test_engine_requires_world_for_formulas(scene):
    s, e = scene
    o = e.add_new("body.point_mass")
    e.assign_process("lib.quadratic_drag", TargetFilter(ids=[o.id]))
    plan = build_plan(s, s.assignments)
    bodies, state = capture_bodies(s, plan.body_ids)
    with pytest.raises(SimulationError, match="world"):
        SimulationEngine(SimulationSettings(), plan, bodies, state)


# ---- permissions & packages ---------------------------------------------------------------------------------------

@pytest.fixture
def dev(tmp_path):
    access = AccessController(str(tmp_path))
    access.create_credentials("correct horse battery", "Test Dev")
    reg = ProcessRegistry()
    from psim.processes import built_in_processes
    built_in_processes.register_all(reg)
    reg.load_library()
    reg.seal(access)
    return access, reg, str(tmp_path / "processes")


GOOD = {"process_id": "lab.spring_anchor", "name": "Anchor Spring", "version": "1.0.0", "kind": "contribution",
        "requires_mass": True,
        "parameters": [{"key": "k", "type": "float", "quantity": "stiffness", "default": 10.0}],
        "script": "force = -param.k * self.position",
        "tests": [{"name": "restoring", "objects": {"self": {"position": [0.5, 0, 0]}}, "expect": {"force": [-5, 0, 0]}}]}


def test_user_role_cannot_register_or_publish(dev):
    access, reg, d = dev
    spec = FormulaProcessSpec.from_dict(GOOD)
    cp, _ = compile_spec(spec)
    with pytest.raises(PermissionDenied):
        reg.register(build_process_def(cp))
    with pytest.raises(PermissionDenied):
        reg.publish(spec, None, d)
    forged = type(access.unlock("correct horse battery"))("x", "y", b"0" * 16)
    access.lock()
    with pytest.raises(PermissionDenied):
        reg.publish(spec, forged, d)
    with pytest.raises(PermissionDenied, match="Incorrect"):
        access.unlock("wrong passphrase")
    assert access.role.value == "user"


def test_publish_sign_load_and_tamper_detection(dev):
    access, reg, d = dev
    token = access.unlock("correct horse battery")
    assert access.role.value == "developer"
    os.makedirs(d, exist_ok=True)
    pdef, path = reg.publish(FormulaProcessSpec.from_dict(GOOD), token, d)
    assert reg.get("lab.spring_anchor") is pdef and os.path.exists(path)
    spec, ok, msg = read_package(path, access)
    assert ok, msg
    # versions must increase
    with pytest.raises(ProcessError, match="must be greater"):
        reg.publish(FormulaProcessSpec.from_dict(GOOD), token, d)
    # application processes cannot be replaced
    with pytest.raises(ProcessError, match="application process"):
        reg.publish(FormulaProcessSpec.from_dict({**GOOD, "process_id": "lib.tether", "version": "9.0.0"}), token, d)
    # failing tests block publication
    bad = {**GOOD, "version": "1.1.0", "tests": [{"name": "x", "objects": {"self": {}}, "expect": {"force": [1, 0, 0]}}]}
    with pytest.raises(ProcessError, match="Test cases failed"):
        reg.publish(FormulaProcessSpec.from_dict(bad), token, d)
    # a fresh installation loads the signed package
    reg2 = ProcessRegistry()
    reg2.load_packages(d, access)
    assert reg2.has("lab.spring_anchor") and reg2.load_errors == []
    # tampering with the formula invalidates the signature
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["process"]["script"] = "force = param.k * self.position"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    reg3 = ProcessRegistry()
    reg3.load_packages(d, access)
    assert not reg3.has("lab.spring_anchor")
    assert "invalid signature" in reg3.load_errors[0]
    # withdrawal
    reg.withdraw("lab.spring_anchor", token)
    assert not reg.has("lab.spring_anchor")


def test_untrusted_signer_is_rejected(dev, tmp_path):
    access, reg, d = dev
    other = AccessController(str(tmp_path / "other"))
    other.create_credentials("another passphrase!", "Stranger")
    tok = other.unlock("another passphrase!")
    from psim.processes.packages import write_package
    os.makedirs(d, exist_ok=True)
    write_package(os.path.join(d, "x.psimproc"), FormulaProcessSpec.from_dict(GOOD), other, tok)
    reg2 = ProcessRegistry()
    reg2.load_packages(d, access)
    assert not reg2.has("lab.spring_anchor") and "untrusted key" in reg2.load_errors[0]
    access.trust_public_key(other.public_key_text(), "stranger")
    reg3 = ProcessRegistry()
    reg3.load_packages(d, access)
    assert reg3.has("lab.spring_anchor")
    assert canonical_bytes({"a": 1}) == canonical_bytes({"a": 1})


def test_devtools_cli(tmp_path, monkeypatch):
    from psim import devtools
    monkeypatch.setenv("PSIM_TEST_PASS", "a long enough passphrase")
    monkeypatch.setattr(devtools, "_controller", lambda: AccessController(str(tmp_path)))
    assert devtools.main(["init-developer", "--author", "CI", "--passphrase-env", "PSIM_TEST_PASS"]) == 0
    assert devtools.main(["init-developer", "--passphrase-env", "PSIM_TEST_PASS"]) == 1  # refuses to overwrite
    assert AccessController(str(tmp_path)).unlock("a long enough passphrase").author == "CI"


def test_project_files_cannot_define_formulas(scene, tmp_path):
    """Projects reference process ids only; an embedded definition is rejected as an unknown field."""
    from psim.project import serialization as ser
    from psim.scene.scene_manager import Scene
    from psim.units.unit_registry import UnitRegistry
    s, e = scene
    o = e.add_new("basic.cube")
    e.assign_process("dyn.gravity", TargetFilter(ids=[o.id]))
    e.set_global_parameters([{"name": "rho", "type": "float", "quantity": "density", "value": 1.2}])
    d = json.loads(json.dumps(ser.project_to_dict(s, UnitRegistry(), ser.ProjectData())))
    s2 = Scene()
    ser.load_project_dict(d, s2, UnitRegistry())
    assert s2.global_parameters == s.global_parameters
    d["processes"][0]["script"] = "force = vec3(1e9, 0, 0)"
    with pytest.raises(ser.ProjectError):
        ser.load_project_dict(d, Scene(), UnitRegistry())
