"""Built-in processes against analytic solutions, plan validation and conflict diagnostics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim import numerics
from psim.processes.process_base import ProcessAssignment, ProcessError, TargetFilter
from psim.processes.process_registry import ProcessRegistry, process_registry
from psim.processes.process_scheduler import build_plan
from psim.properties.property_system import PropertyError
from psim.simulation.settings import SimulationSettings
from psim.simulation.simulation_engine import SimulationEngine
from psim.simulation.state import capture_bodies


def run(scene, duration=1.0, dt=0.001, integrator="rk4"):
    s, _ = scene
    plan = build_plan(s, s.assignments)
    assert plan.ok, [d.message for d in plan.errors]
    bodies, state = capture_bodies(s, plan.body_ids)
    eng = SimulationEngine(SimulationSettings(delta_time=dt, duration=duration, integrator=integrator), plan,
                           bodies, state)
    while eng.step():
        pass
    return eng


def test_registry_has_required_categories():
    reg = process_registry()
    ids = {d.process_id for d in reg.definitions()}
    assert {"kin.free_motion", "kin.constant_acceleration", "dyn.gravity", "dyn.applied_force", "dyn.damping",
            "dyn.spring", "kin.circular_motion", "kin.oscillation", "rot.fixed_axis", "rot.local_axis"} <= ids
    for d in reg.definitions():
        assert d.formula and d.description and d.version and d.category


def test_registry_rejects_invalid_definitions():
    reg = ProcessRegistry()
    good = process_registry().get("dyn.gravity")
    reg.register(good)
    with pytest.raises(ProcessError):
        reg.register(good)
    from dataclasses import replace
    with pytest.raises(ProcessError):
        reg.register(replace(good, process_id="x.bad", writes=("position",)))  # contribution writing position


def test_constant_velocity(scene):
    s, e = scene
    o = e.add_new("basic.cube", position=[1, 2, 3])
    o.linear_velocity = [0.5, -1.0, 2.0]
    e.assign_process("kin.free_motion", TargetFilter(ids=[o.id]))
    np.testing.assert_allclose(run(scene, 2.0, 0.01).snapshot().origin[0], [2, 0, 7], atol=1e-12)


def test_constant_acceleration_uses_object_property(scene):
    s, e = scene
    o = e.add_new("basic.cube")
    o.linear_acceleration = [1.0, 0.0, 0.0]
    e.assign_process("kin.constant_acceleration", TargetFilter(ids=[o.id]))
    snap = run(scene, 2.0)
    np.testing.assert_allclose(snap.snapshot().origin[0], [2.0, 0, 0], atol=1e-9)  # ½ a t²
    np.testing.assert_allclose(snap.snapshot().v[0], [2.0, 0, 0], atol=1e-9)


def test_newtons_second_law_and_torque(scene):
    s, e = scene
    o = e.add_new("basic.cuboid", params={"length": 1, "width": 2, "height": 3})
    o.mass_mode, o.mass_input = "mass", 4.0
    o.force = [8.0, 0.0, 0.0]
    o.torque = [0.0, 0.0, 1.0]
    e.assign_process("dyn.applied_force", TargetFilter(ids=[o.id]))
    snap = run(scene, 1.0).snapshot()
    np.testing.assert_allclose(snap.acc[0], [2.0, 0, 0], atol=1e-12)          # a = F / m
    izz = 4.0 * (1 + 4) / 12
    assert snap.w[0][2] == pytest.approx(1.0 / izz, rel=1e-9)                 # ω = (τ / I) t
    assert snap.alpha[0][2] == pytest.approx(1.0 / izz, rel=1e-9)


def test_viscous_damping_exponential_decay(scene):
    s, e = scene
    o = e.add_new("body.point_mass")
    o.mass_input = 2.0
    o.linear_velocity = [3.0, 0, 0]
    e.assign_process("dyn.damping", TargetFilter(ids=[o.id]), {"linear": 0.8})
    snap = run(scene, 2.0).snapshot()
    assert snap.v[0][0] == pytest.approx(3.0 * math.exp(-0.8 / 2.0 * 2.0), rel=1e-9)


def test_projectile_range(scene):
    s, e = scene
    o = e.add_new("body.point_mass", position=[0, 0, 0])
    v0, ang = 10.0, math.radians(30)
    o.linear_velocity = [v0 * math.cos(ang), 0, v0 * math.sin(ang)]
    e.assign_process("dyn.gravity", TargetFilter(ids=[o.id]))
    t_flight = 2 * v0 * math.sin(ang) / 9.80665
    snap = run(scene, t_flight).snapshot()
    np.testing.assert_allclose(snap.origin[0], [v0 ** 2 * math.sin(2 * ang) / 9.80665, 0, 0], atol=1e-9)


def test_circular_and_oscillation_are_exact(scene):
    s, e = scene
    a = e.add_new("basic.cube")
    b = e.add_new("basic.sphere", position=[1, 2, 3])
    e.assign_process("kin.circular_motion", TargetFilter(ids=[a.id]),
                     {"center": [0, 0, 1], "radius": 2.0, "angular_speed": math.pi / 2})
    e.assign_process("kin.oscillation", TargetFilter(ids=[b.id]),
                     {"direction": [0, 0, 1], "amplitude": 0.5, "frequency": 1.0, "phase": 0.0})
    eng = run(scene, 1.25, 0.01, "semi_implicit_euler")
    sn = eng.snapshot()
    ia, ib = eng.bodies.index[a.id], eng.bodies.index[b.id]
    th = math.pi / 2 * 1.25
    np.testing.assert_allclose(sn.origin[ia], [2 * math.cos(th), 2 * math.sin(th), 1], atol=1e-12)
    np.testing.assert_allclose(sn.origin[ib], [1, 2, 3 + 0.5 * math.sin(2 * math.pi * 1.25)], atol=1e-12)
    speed = np.linalg.norm(sn.v[ia])
    assert speed == pytest.approx(2 * math.pi / 2)


def test_rotation_processes(scene):
    s, e = scene
    fixed = e.add_new("basic.cone")
    local = e.add_new("basic.cylinder", position=[3, 0, 0])
    local.set_euler([math.pi / 2, 0, 0])  # local Z now points along global -Y
    e.assign_process("rot.fixed_axis", TargetFilter(ids=[fixed.id]), {"axis": [0, 0, 1], "angular_speed": math.pi})
    e.assign_process("rot.local_axis", TargetFilter(ids=[local.id]), {"axis": [0, 0, 1], "angular_speed": 1.0})
    eng = run(scene, 0.5, 0.01, "explicit_euler")
    sn = eng.snapshot()
    i, j = eng.bodies.index[fixed.id], eng.bodies.index[local.id]
    np.testing.assert_allclose(numerics.quat_to_euler(sn.q[i]), [0, 0, math.pi / 2], atol=1e-12)
    np.testing.assert_allclose(sn.origin[i], [0, 0, 0], atol=1e-12)   # rotates about its own origin despite COM offset
    np.testing.assert_allclose(sn.w[j], np.array([0, -1.0, 0]), atol=1e-12)  # body axis expressed globally


def test_spring_period(scene):
    s, e = scene
    a = e.add_new("body.point_mass", position=[0, 0, 1])
    b = e.add_new("body.point_mass", position=[1.5, 0, 1])
    e.assign_process("dyn.spring", TargetFilter(ids=[a.id, b.id]), {"stiffness": 40, "rest_length": 1.0})
    period = 2 * math.pi / math.sqrt(40 / 0.5)
    sn = run(scene, period).snapshot()
    assert sn.origin[1, 0] - sn.origin[0, 0] == pytest.approx(1.5, abs=1e-6)  # back to the initial stretch
    np.testing.assert_allclose(sn.origin.mean(axis=0), [0.75, 0, 1], atol=1e-9)  # momentum conserved


def test_com_offset_handled_for_dynamics(scene):
    """A cone (COM below its bounding-box centre) in free fall translates without spurious rotation."""
    s, e = scene
    c = e.add_new("basic.cone", position=[0, 0, 5])
    e.assign_process("dyn.gravity", TargetFilter(ids=[c.id]))
    sn = run(scene, 1.0).snapshot()
    np.testing.assert_allclose(sn.origin[0], [0, 0, 5 - 0.5 * 9.80665], atol=1e-9)
    np.testing.assert_allclose(sn.q[0], [1, 0, 0, 0], atol=1e-12)


# ---- plan validation & conflicts ------------------------------------------------------------------------

def _plan(scene, *assignments):
    s, _ = scene
    return build_plan(s, list(assignments))


def test_conflicting_prescribed_writes(scene):
    s, e = scene
    o = e.add_new("basic.cube")
    plan = _plan(scene,
                 ProcessAssignment("PRC-0001", "kin.circular_motion", TargetFilter(ids=[o.id])),
                 ProcessAssignment("PRC-0002", "kin.oscillation", TargetFilter(ids=[o.id])))
    assert not plan.ok and "Conflicting writes" in plan.errors[0].message
    # different quantities do not conflict
    plan = _plan(scene,
                 ProcessAssignment("PRC-0001", "kin.circular_motion", TargetFilter(ids=[o.id])),
                 ProcessAssignment("PRC-0002", "rot.fixed_axis", TargetFilter(ids=[o.id])))
    assert plan.ok


def test_prescribed_plus_force_is_conflict_but_driven_anchor_is_warning(scene):
    s, e = scene
    a = e.add_new("body.point_mass")
    b = e.add_new("body.point_mass", position=[1, 0, 0])
    plan = _plan(scene, ProcessAssignment("PRC-0001", "kin.oscillation", TargetFilter(ids=[a.id])),
                 ProcessAssignment("PRC-0002", "dyn.gravity", TargetFilter(ids=[a.id])))
    assert not plan.ok
    plan = _plan(scene, ProcessAssignment("PRC-0001", "kin.oscillation", TargetFilter(ids=[a.id])),
                 ProcessAssignment("PRC-0002", "dyn.spring", TargetFilter(ids=[a.id, b.id])))
    assert plan.ok and any("driven anchor" in d.message for d in plan.diagnostics)


def test_contributions_are_summed(scene):
    s, e = scene
    o = e.add_new("body.point_mass")
    e.assign_process("dyn.gravity", TargetFilter(ids=[o.id]), {"g": [0, 0, -10]})
    e.assign_process("dyn.gravity", TargetFilter(ids=[o.id]), {"g": [1, 0, 0]})
    sn = run(scene, 0.01, 0.01).snapshot()
    np.testing.assert_allclose(sn.acc[0], [1, 0, -10])


def test_reference_and_eligibility_errors(scene):
    s, e = scene
    frame = e.add_new("frame.reference")
    o = e.add_new("body.point_mass")
    plan = _plan(scene, ProcessAssignment("PRC-0001", "dyn.applied_force", TargetFilter(ids=[frame.id])))
    assert not plan.ok and "not eligible" in plan.errors[0].message
    plan = _plan(scene, ProcessAssignment("PRC-0001", "dyn.gravity", TargetFilter(ids=["OBJ-999999"])))
    assert any("missing object reference" in d.message for d in plan.errors)
    plan = _plan(scene, ProcessAssignment("PRC-0001", "dyn.spring", TargetFilter(ids=[o.id])))
    assert any("exactly 2" in d.message for d in plan.errors)
    plan = _plan(scene, ProcessAssignment("PRC-0001", "no.such.process", TargetFilter(ids=[o.id])))
    assert any("Unknown process" in d.message for d in plan.errors)
    plan = _plan(scene, ProcessAssignment("PRC-0001", "dyn.gravity", TargetFilter(ids=[o.id]), {"g": [1, 2]}))
    assert any("expected 3 components" in d.message for d in plan.errors)


def test_simulated_child_of_simulated_parent_is_rejected(scene):
    s, e = scene
    p = e.add_new("body.point_mass")
    c = e.add_new("body.point_mass", parent_id=p.id)
    plan = _plan(scene, ProcessAssignment("PRC-0001", "dyn.gravity", TargetFilter(ids=[p.id, c.id])))
    assert not plan.ok and "parent" in plan.errors[0].message


def test_targets_by_type_and_tag(scene):
    s, e = scene
    a = e.add_new("basic.sphere")
    b = e.add_new("basic.cube")
    b.tags = ["falling"]
    e.add_new("basic.cone")
    plan = _plan(scene, ProcessAssignment("PRC-0001", "dyn.gravity", TargetFilter(types=["basic.sphere"],
                                                                                 tags=["falling"])))
    assert set(plan.body_ids) == {a.id, b.id}


def test_assignment_editing_is_undoable(scene):
    s, e = scene
    o = e.add_new("basic.cube")
    a = e.assign_process("dyn.gravity", TargetFilter(ids=[o.id]))
    assert a.uid == "PRC-0001"
    e.update_assignment(a.uid, params={"g": [0, 0, -1.62]}, enabled=False)
    assert s.assignment(a.uid).params["g"] == [0, 0, -1.62] and not s.assignment(a.uid).enabled
    e.stack.undo()
    assert s.assignment(a.uid).params["g"] == [0, 0, -9.80665]
    e.remove_assignments([a.uid])
    assert not s.assignments
    e.stack.undo()
    assert s.assignment(a.uid) is not None
    with pytest.raises(PropertyError):
        e.update_assignment(a.uid, params={"g": "down"})
    b = e.assign_process("dyn.gravity", TargetFilter(ids=[o.id]))
    assert b.uid == "PRC-0002"
