"""Data recording: sampling every N steps from the engine, time consistency, overflow policies."""
from __future__ import annotations


import numpy as np
import pytest

from psim.processes.process_base import TargetFilter
from psim.processes.process_scheduler import build_plan
from psim.recording.recording import Recording
from psim.simulation.settings import SimulationSettings
from psim.simulation.simulation_engine import SimulationEngine
from psim.simulation.state import capture_bodies
from psim.simulation.world import EngineWorld, capture_frames, capture_static, global_values


def make_engine(scene, recording=None, process_outputs=True, **settings):
    s, _ = scene
    plan = build_plan(s, s.assignments)
    assert plan.ok, [d.message for d in plan.errors]
    bodies, state = capture_bodies(s, plan.body_ids)
    statics = capture_static(s)
    world = EngineWorld(statics, plan.body_ids, global_values(s), settings.get("delta_time", 0.01),
                        capture_frames(s, statics))
    eng = SimulationEngine(SimulationSettings(**settings), plan, bodies, state, world)
    if recording is not None:
        eng.attach_recording(recording, process_outputs)
    return eng


def run(eng):
    while eng.step():
        pass
    return eng


def test_samples_every_n_steps_and_final(scene):
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 10])
    ball.linear_velocity = [3, 0, 4]
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    rec = Recording(interval_steps=10)
    run(make_engine(scene, rec, delta_time=0.01, duration=1.005, integrator="rk4"))
    steps = rec.steps()
    assert list(steps[:-1]) == list(range(0, 101, 10)) and steps[-1] == 101  # plus the final (partial) step
    t = rec.times()
    np.testing.assert_allclose(t[:-1], steps[:-1] * 0.01, atol=1e-12)
    assert t[-1] == pytest.approx(1.005)
    g = 9.80665
    x = rec.column((ball.id, "origin"))
    np.testing.assert_allclose(x[:, 0], 3 * t, atol=1e-9)
    np.testing.assert_allclose(x[:, 2], 10 + 4 * t - 0.5 * g * t * t, atol=1e-9)
    np.testing.assert_allclose(rec.column((ball.id, "v"))[:, 2], 4 - g * t, atol=1e-9)
    np.testing.assert_allclose(rec.column((ball.id, "acc")), np.tile([0, 0, -g], (len(t), 1)), atol=1e-12)


def test_recording_does_not_change_the_simulation(scene):
    s, e = scene
    a = e.add_new("body.point_mass", position=[1, 0, 0])
    b = e.add_new("body.point_mass", position=[-1, 0, 0])
    a.linear_velocity = [0, 1, 0]
    e.assign_process("dyn.spring", TargetFilter(ids=[a.id, b.id]), {"stiffness": 20, "rest_length": 1.5, "damping": 0.3})
    plain = run(make_engine(scene, delta_time=0.005, duration=2.0)).snapshot()
    fine, coarse = Recording(interval_steps=1), Recording(interval_steps=7)
    rec1 = run(make_engine(scene, fine, delta_time=0.005, duration=2.0)).snapshot()
    run(make_engine(scene, coarse, delta_time=0.005, duration=2.0))
    assert np.array_equal(plain.origin, rec1.origin) and np.array_equal(plain.v, rec1.v)
    # the coarse recording is exactly a subset of the fine one (independent of Δt and render rate)
    idx = np.searchsorted(fine.steps(), coarse.steps())
    np.testing.assert_array_equal(fine.column((a.id, "origin"))[idx], coarse.column((a.id, "origin")))
    np.testing.assert_array_equal(fine.column((a.id, "acc"))[idx], coarse.column((a.id, "acc")))


def test_accelerations_are_consistent_with_sample_times(scene):
    """a(t_k) recorded with x(t_k), v(t_k): check against finite differences of the recorded velocity."""
    s, e = scene
    a = e.add_new("body.point_mass", position=[1, 0, 0])
    anchor = e.add_new("body.point_mass")
    e.assign_process("kin.circular_motion", TargetFilter(ids=[anchor.id]), {"center": [0, 0, 0], "radius": 0.5,
                                                                           "angular_speed": 2.0})
    e.assign_process("dyn.spring", TargetFilter(ids=[anchor.id, a.id]), {"stiffness": 40, "rest_length": 0.5})
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.001, duration=1.0, integrator="rk4"))
    t = rec.times()
    for oid in (a.id, anchor.id):
        v, acc = rec.column((oid, "v")), rec.column((oid, "acc"))
        fd = (v[2:] - v[:-2]) / (t[2:] - t[:-2])[:, None]
        np.testing.assert_allclose(acc[1:-1][:-1], fd[:-1], atol=2e-4)
    # the prescribed anchor's acceleration is the centripetal acceleration of its circle
    x = rec.column((anchor.id, "origin"))
    np.testing.assert_allclose(rec.column((anchor.id, "acc")), -4.0 * x, atol=1e-5)


def test_discrete_update_bodies_record_step_rates(scene):
    s, e = scene
    o = e.add_new("body.point_mass")
    o.linear_acceleration = [0, 0, -2.0]
    e.assign_process("lib.euler_velocity_update", TargetFilter(ids=[o.id]))
    e.assign_process("lib.euler_position_update", TargetFilter(ids=[o.id]))
    rec = Recording(interval_steps=5)
    run(make_engine(scene, rec, delta_time=0.01, duration=0.5))
    np.testing.assert_allclose(rec.column((o.id, "acc"))[:-1], np.tile([0, 0, -2.0], (len(rec.times()) - 1, 1)),
                               atol=1e-9)


def test_process_outputs_add_up_to_the_total(scene):
    s, e = scene
    o = e.add_new("body.point_mass")
    o.mass_input = 2.0
    o.force = [4, 0, 0]
    o.linear_velocity = [1, 0, 0]
    g = e.assign_process("dyn.gravity", TargetFilter(ids=[o.id]))
    f = e.assign_process("dyn.applied_force", TargetFilter(ids=[o.id]))
    d = e.assign_process("dyn.damping", TargetFilter(ids=[o.id]), {"linear": 0.5, "angular": 0.0})
    rec = Recording(interval_steps=3)
    run(make_engine(scene, rec, delta_time=0.01, duration=0.3))
    total = rec.column((o.id, "acc"))
    parts = [rec.column(("proc", a.uid, o.id, "acc")) for a in (g, f, d)]
    np.testing.assert_allclose(sum(parts), total, atol=1e-12)
    np.testing.assert_allclose(parts[1], np.tile([2, 0, 0], (len(total), 1)))   # F/m
    assert rec.meta["processes"][d.uid]["process_id"] == "dyn.damping"
    off = Recording()
    run(make_engine(scene, off, process_outputs=False, delta_time=0.01, duration=0.1))
    assert not off.has(("proc", g.uid, o.id, "acc"))


def test_children_of_simulated_assemblies_and_custom_outputs(scene):
    from psim.properties.custom_properties import CustomProperty
    s, e = scene
    asm = e.add_new("group.assembly")
    child = e.add_new("body.point_mass", position=[1, 0, 0])
    e.reparent([child.id], asm.id)
    e.assign_process("rot.fixed_axis", TargetFilter(ids=[asm.id]), {"axis": [0, 0, 1], "angular_speed": 2.0})
    ball = e.add_new("body.point_mass")
    ball.linear_velocity = [3, 0, 4]
    ball.add_custom(CustomProperty.create(key="kinetic_energy", name="KE", vtype="float", quantity="energy"))
    e.assign_process("kin.free_motion", TargetFilter(ids=[ball.id]))
    e.assign_process("lib.kinetic_energy_monitor", TargetFilter(ids=[ball.id]))
    rec = Recording(interval_steps=10)
    run(make_engine(scene, rec, delta_time=0.001, duration=0.5, integrator="rk4"))
    t = rec.times()
    x = rec.column((child.id, "origin"))
    np.testing.assert_allclose(x, np.c_[np.cos(2 * t), np.sin(2 * t), np.zeros_like(t)], atol=1e-9)
    np.testing.assert_allclose(rec.column((child.id, "acc")), -4 * x, atol=1e-6)
    assert rec.meta["objects"][child.id]["body"] is False
    ke = rec.column((ball.id, "custom.kinetic_energy"))[:, 0]
    np.testing.assert_allclose(ke[1:], 12.5)   # written by the update process from the first step on


def test_overflow_policies():
    def fill(rec, n):
        rec.ensure_columns([(("o", "x"), 1)])
        for k in range(n):
            if rec.wants(k):
                rec.append(k * 0.1, k, {("o", "x"): float(k)})
        return rec
    width_bytes = 8 * 3
    dec = fill(Recording(max_bytes=100 * width_bytes, overflow="decimate"), 1000)
    assert dec.n <= 100 and dec.stride >= 8 and dec.decimations >= 3
    steps = dec.steps()
    assert steps[0] == 0 and len(set(np.diff(steps))) == 1          # still evenly spaced from the start
    assert any(ev.kind == "recording" for ev in dec.events)
    win = fill(Recording(max_bytes=100 * width_bytes, overflow="window"), 1000)
    assert win.n <= 100 and win.steps()[-1] == 999 and win.dropped > 0
    assert np.all(np.diff(win.steps()) == 1)
    stop = fill(Recording(max_bytes=100 * width_bytes, overflow="stop"), 1000)
    assert stop.full_stop and stop.n == 100 and stop.steps()[-1] == 99
    # columns added later are NaN for earlier samples
    stop.ensure_columns([(("o", "y"), 2)])
    assert np.isnan(stop.column(("o", "y"))).all() and stop.column(("o", "y")).shape == (100, 2)
    assert "100 samples" in stop.status_text()


def test_quiet_reevaluation_does_not_duplicate_skip_warnings(scene):
    from psim.processes.formula_process import FormulaProcessSpec, build_process_def, compile_spec
    from psim.processes.process_registry import process_registry
    s, e = scene
    spec = FormulaProcessSpec.from_dict({"process_id": "t.skippy", "name": "Skippy", "kind": "contribution",
                                         "requires_mass": True, "error_policy": "skip",
                                         "script": 'require false : "nope"\nforce = vec3(1[N], 0[N], 0[N])'})
    cp, _ = compile_spec(spec)
    reg = process_registry()
    reg._defs["t.skippy"] = build_process_def(cp)
    try:
        o = e.add_new("body.point_mass")
        e.assign_process("t.skippy", TargetFilter(ids=[o.id]))
        eng = make_engine(scene, Recording(), delta_time=0.01, duration=0.03)
        eng.step()
        assert len(eng.snapshot().warnings) == 1
    finally:
        reg._defs.pop("t.skippy", None)
