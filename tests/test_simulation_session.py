"""Simulation session behaviour (controller) and its GUI integration."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest

from psim.gui.simulation_controller import SimulationController
from psim.processes.process_base import TargetFilter
from psim.project import serialization as ser
from psim.scene.scene_manager import Scene
from psim.simulation.settings import PacingMode, SimulationSettings
from psim.units.unit_registry import UnitRegistry


@pytest.fixture
def session(scene, qapp):
    s, e = scene
    settings = SimulationSettings(delta_time=0.01, duration=0.5, pacing=PacingMode.AS_FAST_AS_POSSIBLE,
                                  render_fps=30)
    ctl = SimulationController(s, e, lambda: settings)
    messages = {"info": [], "warning": [], "error": []}
    for k in messages:
        getattr(ctl, k).connect(messages[k].append)
    yield s, e, ctl, settings, messages
    ctl.abandon()


def _falling_ball(e):
    ball = e.add_new("basic.sphere", position=[0, 0, 5])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    return ball


def test_run_stop_undo_reset(session, qtbot):
    s, e, ctl, settings, msgs = session
    ball = _falling_ball(e)
    n_cmds = e.stack.count()
    assert ctl.start()
    assert ctl.session_active and not e.recording
    qtbot.waitUntil(lambda: ctl.state == "finished", timeout=10000)
    qtbot.waitUntil(lambda: ctl.last_stats is not None and ctl.last_stats.steps == 50, timeout=5000)
    qtbot.wait(50)
    z = s.world_pose(ball.id)[0][2]
    assert z < 5 - 1.0  # fell (≈ 1.23 m in 0.5 s)
    assert s.get(ball.id).linear_velocity[2] == pytest.approx(-9.80665 * 0.5, rel=1e-9)
    assert e.stack.count() == n_cmds  # nothing recorded during the session
    ctl.stop()
    assert not ctl.session_active and e.recording
    assert e.stack.count() == n_cmds + 1 and e.stack.undoText().startswith("Simulation run")
    e.stack.undo()
    assert s.world_pose(ball.id)[0][2] == pytest.approx(5.0)
    e.stack.redo()
    ctl.reset()
    assert s.world_pose(ball.id)[0][2] == pytest.approx(5.0)
    assert s.get(ball.id).linear_velocity == [0.0, 0.0, 0.0]


def test_reset_during_session_restores_initial_state_exactly(session, qtbot):
    s, e, ctl, settings, _ = session
    ball = _falling_ball(e)
    before = s.snapshot([ball.id])
    settings.pacing = PacingMode.REAL_TIME
    settings.update_rate_hz = 50
    settings.duration = 100
    assert ctl.start()
    qtbot.waitUntil(lambda: ctl.last_stats is not None and ctl.last_stats.steps > 5, timeout=5000)
    ctl.reset()
    assert not ctl.session_active
    assert s.snapshot([ball.id]) == before


def test_live_edit_is_applied_at_next_step(session, qtbot):
    s, e, ctl, settings, msgs = session
    o = e.add_new("body.point_mass")
    e.assign_process("kin.free_motion", TargetFilter(ids=[o.id]))
    settings.pacing = PacingMode.REAL_TIME
    settings.update_rate_hz = 100
    settings.duration = 100
    assert ctl.start()
    qtbot.waitUntil(lambda: ctl.last_stats is not None and ctl.last_stats.steps > 2, timeout=5000)
    e.set_property([o.id], "linear_velocity", [2.0, 0.0, 0.0])  # user edit while running
    qtbot.wait(400)
    ctl.pause()
    qtbot.waitUntil(lambda: ctl.state == "paused", timeout=3000)
    qtbot.wait(50)
    assert s.get(o.id).linear_velocity == pytest.approx([2.0, 0.0, 0.0])
    assert s.world_pose(o.id)[0][0] > 0.3
    assert any("Applied edits" in m for m in msgs["info"])


def test_invalid_plan_prevents_start(session):
    s, e, ctl, settings, msgs = session
    o = e.add_new("basic.cube")
    e.assign_process("kin.circular_motion", TargetFilter(ids=[o.id]))
    e.assign_process("kin.oscillation", TargetFilter(ids=[o.id]))
    assert not ctl.start()
    assert not ctl.session_active
    assert any("Conflicting writes" in m for m in msgs["error"])


def test_nothing_to_simulate(session):
    s, e, ctl, settings, msgs = session
    e.add_new("basic.cube")
    assert not ctl.start()
    assert any("Nothing to simulate" in m for m in msgs["error"])


def test_single_step_from_idle(session, qtbot):
    s, e, ctl, settings, _ = session
    _falling_ball(e)
    ctl.step()
    qtbot.waitUntil(lambda: ctl.last_stats is not None and ctl.last_stats.steps == 1, timeout=5000)
    assert ctl.state == "paused"
    assert ctl.last_stats.time == pytest.approx(0.01)


# ---- project files -------------------------------------------------------------------------------------------

def test_assignments_round_trip(tmp_path, scene):
    s, e = scene
    o = e.add_new("basic.cube")
    a = e.assign_process("dyn.gravity", TargetFilter(ids=[o.id], tags=["x"]), {"g": [0, 0, -1.62]}, "Moon")
    e.update_assignment(a.uid, enabled=False)
    path = tmp_path / "p.psim"
    ser.save_project(str(path), s, UnitRegistry(), ser.ProjectData())
    s2 = Scene()
    ser.load_project(str(path), s2, UnitRegistry())
    assert s2.process_snapshot() == s.process_snapshot()
    assert s2.allocate_process_uid() == "PRC-0002"


def test_version_1_projects_still_load(scene):
    s, e = scene
    e.add_new("basic.cube")
    d = json.loads(json.dumps(ser.project_to_dict(s, UnitRegistry(), ser.ProjectData())))
    d["format_version"] = 1
    d.pop("processes")
    d.pop("process_counter")
    s2 = Scene()
    data = ser.load_project_dict(d, s2, UnitRegistry())
    assert len(s2) == 1 and s2.assignments == [] and data.warnings == []


def test_unknown_process_in_project_is_kept_disabled(scene):
    s, e = scene
    o = e.add_new("basic.cube")
    e.assign_process("dyn.gravity", TargetFilter(ids=[o.id]))
    d = json.loads(json.dumps(ser.project_to_dict(s, UnitRegistry(), ser.ProjectData())))
    d["processes"][0]["process"] = "vendor.future_model"
    s2 = Scene()
    data = ser.load_project_dict(d, s2, UnitRegistry())
    assert not s2.assignments[0].enabled and any("not available" in w for w in data.warnings)
    d["processes"][0]["uid"] = "bad"
    with pytest.raises(ser.ProjectError):
        ser.load_project_dict(d, Scene(), UnitRegistry())


# ---- GUI integration -------------------------------------------------------------------------------------------

@pytest.fixture
def window(qapp, requires_gl, tmp_path):
    from psim.gui import theme
    from psim.gui.main_window import MainWindow
    theme.apply(qapp, "dark")
    w = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), check_recovery=False)
    w.resize(1400, 900)
    w.show()
    QTest.qWaitForWindowExposed(w)
    yield w
    w.sim.abandon()
    w.undo.setClean()
    w._extra_dirty = False
    w.close()
    w.deleteLater()  # closed windows would otherwise accumulate (and be restyled by every theme change)
    QTest.qWait(0)


def test_gui_simulation_controls(window, qtbot):
    w = window
    ball = w.editor.add_new("basic.sphere", position=[0, 0, 5])
    w.editor.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    assert w.processes_panel.table.rowCount() == 1
    # toolbar fields edit the project settings (Δt and rate independently)
    w.sim_fields.dt.set_si(0.002)
    w.sim_fields.deltaTimeChanged.emit(0.002)
    w.sim_fields.updateRateChanged.emit(250.0)
    w.sim_fields.durationChanged.emit(0.4)
    s = w.project.simulation
    assert (s.delta_time, s.update_rate_hz, s.duration) == (0.002, 250.0, 0.4)
    w.actions_["sim.start"].trigger()
    assert w.sim.state == "running"
    assert not w.act_undo.isEnabled()
    n = len(w.scene)
    w.add_library_item("type:basic.cube")  # structural edits are blocked during a session
    assert len(w.scene) == n
    assert ball.id in w.inspector.simulated_ids or w.scene.selection == []
    qtbot.waitUntil(lambda: w.sim.state == "finished", timeout=15000)
    qtbot.waitUntil(lambda: w.sim.last_stats.steps == 200, timeout=5000)
    assert "finished" in w.sim_fields.status.text()
    w.actions_["sim.stop"].trigger()
    assert w.sim.state == "idle" and w.act_undo.isEnabled()
    assert w.undo.undoText().startswith("Simulation run")
    w.actions_["sim.reset"].trigger()
    np.testing.assert_allclose(w.scene.world_pose(ball.id)[0], [0, 0, 5])


def test_gui_inspector_marks_simulated_values(window, qtbot):
    from psim.gui.widgets.inputs import OriginBadge
    w = window
    ball = w.editor.add_new("basic.sphere", position=[0, 0, 5])
    w.editor.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    w.scene.set_selection([ball.id])
    qtbot.wait(50)
    assert "Assigned Processes" in [sec.header.text() for sec in w.inspector._sections]
    w.project.simulation.update_rate_hz = 20
    w.project.simulation.duration = 100
    w.project.simulation.pacing = PacingMode.REAL_TIME
    w._sim_start()
    qtbot.waitUntil(lambda: any(b.text() == "S" for b in w.inspector.findChildren(OriginBadge)), timeout=5000)
    w.sim.reset()
    qtbot.waitUntil(lambda: not any(b.text() == "S" for b in w.inspector.findChildren(OriginBadge)), timeout=5000)


def test_session_records_data_and_events(session, qtbot):
    s, e, ctl, settings, msgs = session
    ball = _falling_ball(e)
    settings.record_every_n_steps = 5
    settings.pacing = PacingMode.REAL_TIME
    settings.update_rate_hz = 200
    settings.duration = 5.0
    changed = []
    ctl.recordingChanged.connect(lambda: changed.append(1))
    assert ctl.start()
    rec = ctl.recording
    assert rec is not None and changed == [1] and rec.interval_steps == 5
    qtbot.waitUntil(lambda: rec.n >= 4, timeout=5000)
    ctl.pause()
    qtbot.waitUntil(lambda: ctl.state == "paused", timeout=5000)
    ctl.start()
    qtbot.waitUntil(lambda: rec.n >= 8, timeout=5000)
    ctl.stop()
    steps = rec.steps()
    assert np.all(np.diff(steps[:-1]) == 5) and steps[-1] >= steps[-2]   # final sample at the stop time
    t_stop = rec.times()[-1]
    z = rec.column((ball.id, "origin"))[:, 2]
    assert z[-1] == pytest.approx(s.world_pose(ball.id)[0][2], abs=1e-12)  # the end state is recorded
    n = steps[-1]   # semi-implicit Euler: z_n = z0 - g·Δt²·n(n+1)/2
    assert t_stop == pytest.approx(n * 0.01) and z[-1] == pytest.approx(5 - 9.80665 * 0.01 ** 2 * n * (n + 1) / 2)
    kinds = [ev.kind for ev in rec.event_list()]
    assert kinds[0] == "start" and "pause" in kinds and "resume" in kinds and kinds[-1] == "stop"
    assert ctl.recording is rec                      # kept after Stop for analysis
    ctl.clear_recording()
    assert ctl.recording is None and changed == [1, 1]
    settings.record_enabled = False
    settings.pacing = PacingMode.AS_FAST_AS_POSSIBLE
    settings.duration = 0.1
    assert ctl.start()
    assert ctl.recording is None


def test_broken_joint_is_reported_and_no_longer_drawn(session, qtbot):
    from psim.rendering import physics_overlay
    s, e, ctl, settings, msgs = session
    ball = e.add_new("body.point_mass", position=[0, 0, 1.0])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    m = ball.mass_properties()["mass"]
    jid = e.add_joint("ball", ball.id, None, [0, 0, 1.0], name="Weak Hook", break_force=0.5 * m * 9.80665)
    assert len(physics_overlay.joint_lines(s, 0.05)[0]) == 1
    assert ctl.start()
    qtbot.waitUntil(lambda: ctl.state == "finished", timeout=10000)
    qtbot.waitUntil(lambda: s.broken_joints == {jid}, timeout=5000)
    assert any("Weak Hook" in m and "broke" in m for m in msgs["info"])
    assert physics_overlay.joint_lines(s, 0.05)[0] == []                  # a broken joint is not drawn
    ctl.reset()
    assert s.broken_joints == set() and len(physics_overlay.joint_lines(s, 0.05)[0]) == 1
    assert [j.id for j in s.joints] == [jid]                               # the joint itself is unchanged
