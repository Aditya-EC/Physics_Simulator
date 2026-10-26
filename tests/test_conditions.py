"""Conditional stopping events (stop conditions), stage profiling and their settings."""
from __future__ import annotations

import pytest

from psim.processes.process_base import TargetFilter
from psim.processes.process_scheduler import build_plan
from psim.recording.recording import Recording
from psim.simulation.conditions import ConditionError, attach_conditions, validate_conditions
from psim.simulation.settings import SimulationSettings
from psim.simulation.simulation_engine import PROFILE_STAGES, SimulationEngine
from psim.simulation.state import capture_bodies
from psim.simulation.world import EngineWorld, capture_frames, capture_static, global_values

G = 9.80665


def _engine(scene, conditions, recording=None, **settings):
    s, _ = scene
    st = SimulationSettings(stop_conditions=conditions, **settings)
    plan = build_plan(s, s.assignments)
    attach_conditions(plan, s, st)
    if not plan.ok:
        return plan, None
    bodies, state = capture_bodies(s, plan.body_ids)
    statics = capture_static(s)
    world = EngineWorld(statics, plan.body_ids, global_values(s), st.delta_time, capture_frames(s, statics))
    eng = SimulationEngine(st, plan, bodies, state, world)
    if recording is not None:
        eng.attach_recording(recording)
    return plan, eng


def _drop(scene, height=10.0):
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, height])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    return ball


def test_stop_condition_ends_the_run_when_the_ball_reaches_the_ground(scene):
    ball = _drop(scene)
    rec = Recording()
    cond = {"name": "Ground", "expression": f'object("{ball.id}").position.z < 0[m]', "action": "stop"}
    plan, eng = _engine(scene, [cond], rec, delta_time=0.001, duration=10.0, integrator="rk4")
    while eng.step():
        pass
    t_hit = (2 * 10.0 / G) ** 0.5
    assert eng.finished and eng.clock.time == pytest.approx(t_hit, abs=0.0011)   # first step with z < 0
    assert eng.state.x[0, 2] < 0 and eng.state.x[0, 2] > -G * t_hit * 0.0011
    assert eng.take_fired() == [("Ground", "stop", eng.clock.time)]
    assert any(ev.kind == "condition" for ev in rec.event_list())
    assert rec.times()[-1] == pytest.approx(eng.clock.time)                         # final sample recorded


def test_pause_condition_fires_once_per_transition(scene):
    ball = _drop(scene)
    ball.linear_velocity = [0, 0, 5.0]
    expr = f'object("{ball.id}").velocity.z < 0[m/s] && t > 0.1[s]'
    plan, eng = _engine(scene, [{"name": "Apex", "expression": expr, "action": "pause"}], delta_time=0.01,
                        duration=3.0)
    fired = []
    while eng.step():
        fired += eng.take_fired()
    assert len(fired) == 1 and fired[0][0] == "Apex"
    assert fired[0][2] == pytest.approx(5.0 / G, abs=0.011)
    assert not eng.finished or eng.clock.time == pytest.approx(3.0)


def test_conditions_are_type_and_reference_checked(scene):
    s, e = scene
    ball = _drop(scene)
    plan, eng = _engine(scene, [{"name": "Bad", "expression": f'object("{ball.id}").position.z < 1[s]'}])
    assert eng is None and "Incompatible dimensions" in plan.errors[0].message
    plan, eng = _engine(scene, [{"name": "NotBool", "expression": f'object("{ball.id}").position.z'}])
    assert eng is None and "must be a Boolean" in plan.errors[0].message
    plan, eng = _engine(scene, [{"name": "Missing", "expression": 'object("OBJ-999999").position.z < 0[m]'}])
    assert eng is None and "missing object 'OBJ-999999'" in plan.errors[0].message
    plan, eng = _engine(scene, [{"name": "Off", "expression": "nonsense(", "enabled": False}])
    assert eng is not None                                  # disabled conditions are not compiled


def test_conditions_can_use_global_parameters(scene):
    s, e = scene
    ball = _drop(scene)
    e.set_global_parameters([{"name": "floor", "type": "float", "quantity": "length", "value": 5.0}])
    expr = f'object("{ball.id}").position.z < global.floor'
    plan, eng = _engine(scene, [{"name": "Floor", "expression": expr, "action": "stop"}], delta_time=0.001,
                        duration=5.0, integrator="rk4")
    while eng.step():
        pass
    assert eng.clock.time == pytest.approx((2 * 5.0 / G) ** 0.5, abs=0.0011)


def test_condition_settings_validation_and_round_trip():
    good = [{"name": "A", "expression": "t > 1[s]", "action": "stop"}]
    s = SimulationSettings.from_dict({"stop_conditions": good})
    assert s.stop_conditions == [{"name": "A", "expression": "t > 1[s]", "action": "stop", "enabled": True}]
    assert SimulationSettings.from_dict(s.to_dict()).stop_conditions == s.stop_conditions
    for bad in ([{"name": "", "expression": "t > 0"}], [{"name": "A", "expression": "t > 0", "action": "explode"}],
                [{"name": "A", "expression": "t>0"}, {"name": "a", "expression": "t>0"}],
                [{"name": "A", "expression": "t>0", "script": "x"}]):
        with pytest.raises(ConditionError):
            validate_conditions(bad)
        assert SimulationSettings(stop_conditions=bad).validate()


def test_profile_reports_every_stage(scene):
    _drop(scene)
    plan, eng = _engine(scene, [], delta_time=0.01, duration=0.5)
    while eng.step():
        pass
    rep = eng.profile_report()
    assert rep["steps"] == 50 and set(PROFILE_STAGES) <= set(rep)
    assert rep["total"] == pytest.approx(sum(rep[k] for k in PROFILE_STAGES))
    assert rep["processes_and_integration"] > 0
