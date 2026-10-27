"""Saving and restoring simulation states: exact continuation, validation and the session controller."""
from __future__ import annotations

import json

import numpy as np
import pytest

from psim.gui.simulation_controller import SimulationController
from psim.processes.process_base import TargetFilter
from psim.recording.recording import Recording
from psim.simulation.settings import PacingMode, SimulationSettings
from psim.simulation.simulation_engine import SimulationError

from test_recording import make_engine


def _scene_with_physics_and_formulas(scene):
    s, e = scene
    g = e.add_new("basic.plane", params={"length": 10, "width": 10})
    e.set_property([g.id], "physics.collider", "auto")
    box = e.add_new("body.rigid_box", position=[0, 0, 1.0], params={"length": 0.4, "width": 0.4, "height": 0.4})
    box.linear_velocity = [1.0, 0.0, 0.0]
    box.angular_velocity = [0.0, 3.0, 1.0]
    arm = e.add_new("body.rigid_box", position=[3.8, 0, 2.0], params={"length": 1.6, "width": 0.1, "height": 0.1})
    e.add_joint("revolute", arm.id, None, [3.0, 0, 2.0], [0, 1, 0])
    e.assign_process("dyn.gravity", TargetFilter(ids=[box.id, arm.id]))
    e.assign_process("lib.quadratic_drag", TargetFilter(ids=[box.id]), {"rho": 1.2, "cd": 1.0, "area": 0.16})
    return box, arm


@pytest.mark.parametrize("integrator", ["semi_implicit_euler", "rk4"])
def test_restored_run_continues_bit_for_bit(scene, integrator):
    box, arm = _scene_with_physics_and_formulas(scene)
    ref = make_engine(scene, delta_time=0.005, duration=2.0, integrator=integrator)
    while ref.clock.time < 0.75 - 1e-9:
        ref.step()
    saved = json.loads(json.dumps(ref.export_state()))          # through JSON, as on disk
    while ref.step():
        pass
    other = make_engine(scene, delta_time=0.005, duration=2.0, integrator=integrator)
    other.step()                                                  # some unrelated progress first
    other.import_state(saved)
    assert other.clock.time == pytest.approx(0.75) and other.clock.steps == 150
    while other.step():
        pass
    assert other.clock.steps == ref.clock.steps
    for a in ("x", "q", "v", "w"):
        np.testing.assert_array_equal(getattr(other.state, a), getattr(ref.state, a))


def test_restore_truncates_the_recording_and_logs_it(scene):
    _scene_with_physics_and_formulas(scene)
    rec = Recording(interval_steps=10)
    eng = make_engine(scene, rec, delta_time=0.01, duration=2.0)
    for _ in range(50):
        eng.step()
    saved = eng.export_state()
    for _ in range(100):
        eng.step()
    eng.import_state(saved)
    assert rec.times()[-1] == pytest.approx(0.4)          # t = 0.5 is recorded again by the next step
    assert rec.event_list()[-1].kind == "restore" and "discarded" in rec.event_list()[-1].message
    while eng.step():
        pass
    t = rec.times()
    assert np.all(np.diff(t) > 0) and t[-1] == pytest.approx(2.0)


def test_mismatched_or_invalid_states_are_rejected(scene):
    s, e = scene
    box, arm = _scene_with_physics_and_formulas(scene)
    eng = make_engine(scene, delta_time=0.01, duration=1.0)
    good = eng.export_state()
    with pytest.raises(SimulationError, match="Not a PSim simulation state"):
        eng.import_state({"bodies": {}})
    with pytest.raises(SimulationError, match="newer"):
        eng.import_state({**good, "version": 99})
    bad = json.loads(json.dumps(good))
    bad["bodies"][box.id]["x"] = [0, "nan", 0]
    with pytest.raises(SimulationError, match="Invalid simulation state"):
        eng.import_state(bad)
    other = json.loads(json.dumps(good))
    other["bodies"]["OBJ-999999"] = other["bodies"].pop(box.id)
    with pytest.raises(SimulationError, match="does not match"):
        eng.import_state(other)
    np.testing.assert_array_equal(eng.state.x, make_engine(scene, delta_time=0.01, duration=1.0).state.x)


def test_controller_saves_and_restores_through_a_file(scene, qapp, qtbot, tmp_path):
    s, e = scene
    ball = e.add_new("basic.sphere", position=[0, 0, 50])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    settings = SimulationSettings(delta_time=0.01, duration=2.0, pacing=PacingMode.AS_FAST_AS_POSSIBLE)
    ctl = SimulationController(s, e, lambda: settings)
    try:
        path = str(tmp_path / "mid.psimstate")
        with pytest.raises(SimulationError):
            ctl.save_state(path)
        assert ctl.start(paused=True)
        for _ in range(30):
            ctl.step()
        qtbot.waitUntil(lambda: ctl.last_stats is not None and ctl.last_stats.steps == 30, timeout=5000)
        assert ctl.save_state(path) == pytest.approx(0.3)
        assert json.load(open(path, encoding="utf-8"))["format"] == "psim-simulation-state"
        ctl.start()
        qtbot.waitUntil(lambda: ctl.state == "finished", timeout=10000)
        ctl.stop()
        assert ctl.restore_state(path) == pytest.approx(0.3)      # starts a paused session first
        qtbot.waitUntil(lambda: ctl.last_stats is not None and ctl.last_stats.time == pytest.approx(0.3),
                        timeout=5000)
        qtbot.waitUntil(lambda: abs(s.get(ball.id).linear_velocity[2] + 9.80665 * 0.3) < 1e-9, timeout=5000)
        ctl.start()
        qtbot.waitUntil(lambda: ctl.state == "finished", timeout=10000)
        assert ctl.runner.engine.clock.steps == 200
    finally:
        ctl.abandon()
