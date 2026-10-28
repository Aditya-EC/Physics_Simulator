"""The 30 mandatory acceptance tests of the specification (§28), one end-to-end test each.

Each test exercises the feature the way a user meets it (the main window, the session controller, the
project file or the engine) and is deliberately self-contained. The detailed, component-level tests
that back each item are listed in docs/acceptance.md.
"""
# ruff: noqa: F811  (pytest fixtures imported from other test modules)
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QTimer
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from psim import numerics
from psim.gui.object_library import LIBRARY_MIME
from psim.processes.process_base import ProcessAssignment, TargetFilter
from psim.processes.process_scheduler import build_plan
from psim.project import serialization as ser
from psim.recording.plot_spec import PlotSpec, SeriesSpec, series_data
from psim.recording.recording import Recording
from psim.recording.variables import Evaluator, Variable
from psim.scene.scene_manager import Scene
from psim.simulation.settings import PacingMode
from psim.units.unit_registry import UnitRegistry

from test_gui import _render, _size_factor, window  # noqa: F401  (window fixture)
from test_recording import make_engine, run

G = 9.80665


def _wait(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    return False


def _pixels_changed(a, b, step=4):
    return sum(a.pixel(x, y) != b.pixel(x, y) for x in range(0, a.width(), step) for y in range(0, a.height(), step))


def _ball(editor, z=10.0, vel=(0.0, 0.0, 0.0)):
    b = editor.add_new("body.point_mass", position=[0, 0, z])
    editor.set_property([b.id], "linear_velocity", list(vel))
    editor.assign_process("dyn.gravity", TargetFilter(ids=[b.id]))
    return b


def _run_window(w, **settings):
    for k, v in settings.items():
        setattr(w.project.simulation, k, v)
    assert w.sim.start()
    assert _wait(lambda: w.sim.state == "finished", 30)
    w.sim.stop()


# ---- 1–7: application, objects and properties ---------------------------------------------------------------------

def test_at01_application_launches_as_a_desktop_application(window):
    assert window.isVisible() and window.windowTitle().endswith("PSim Studio")
    assert window.viewport.renderer.ready, window.viewport.renderer.error
    menus = [a.text().replace("&", "") for a in window.menuBar().actions() if a.isVisible()]
    assert len(menus) == 14 and menus[0] == "File" and menus[-1] == "Help"


def test_at02_multiple_genuine_3d_object_types_render(window):
    w = window
    w.project.view.axes.visible = False
    w.project.view.grids["XY"].visible = False
    empty = _render(w).copy()
    for k, t in enumerate(("basic.cube", "basic.sphere", "basic.cylinder", "basic.cone", "basic.capsule")):
        o = w.editor.add_new(t, position=[2.0 * k - 4, 0, 0.5])
        o.color = [0.9, 0.2 + 0.15 * k, 0.2]
    w.scene.set_selection([])
    w.viewport.frame_objects(None)
    img = _render(w)
    changed = _pixels_changed(empty, img)
    assert changed > 500 * _size_factor(w) ** 2, (changed, w.viewport.size())     # framed objects: an area
    for o in w.scene.objects.values():              # every object is visible where the camera projects it
        xy, ok = w.viewport.camera.project(np.array([w.scene.world_pose(o.id)[0]]), w.viewport.width(),
                                           w.viewport.height())
        c = img.pixelColor(int(xy[0][0]), int(xy[0][1]))
        assert ok[0] and c.red() > c.blue() + 30, (o.type_id, c.name())


def test_at03_objects_can_be_dragged_into_the_viewport(window):
    vp = window.viewport
    mime = QMimeData()
    mime.setData(LIBRARY_MIME, json.dumps({"key": "type:basic.sphere"}).encode())
    pos = QPointF(vp.width() * 0.5, vp.height() * 0.55)
    for E in (QDragEnterEvent, QDragMoveEvent):
        QApplication.sendEvent(vp, E(pos.toPoint(), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                                     Qt.KeyboardModifier.NoModifier))
    QApplication.sendEvent(vp, QDropEvent(pos, Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                                          Qt.KeyboardModifier.NoModifier))
    assert len(window.scene) == 1 and window.scene.primary.type_id == "basic.sphere"


def test_at04_each_object_receives_a_unique_persistent_id(scene, tmp_path):
    s, e = scene
    ids = [e.add_new("basic.cube").id for _ in range(3)]
    e.delete([ids[1]])
    ids.append(e.add_new("basic.cube").id)
    assert len(set(ids)) == 4 and ids[-1] == "OBJ-000004"       # a deleted ID is never reused
    path = str(tmp_path / "ids.psim")
    ser.save_project(path, s, UnitRegistry(), ser.ProjectData())
    s2 = Scene()
    ser.load_project(path, s2, UnitRegistry())
    assert sorted(s2.objects) == sorted(s.objects)
    from psim.scene.editing import SceneEditor
    assert SceneEditor(s2).add_new("basic.cube").id == "OBJ-000005"   # also after reopening


def test_at05_geometry_position_orientation_mass_and_color_are_editable(scene):
    s, e = scene
    o = e.add_new("basic.cuboid")
    m0, length0 = o.mass(), o.params["length"]
    e.set_property([o.id], "params.length", 2 * length0)
    assert o.mass() == pytest.approx(2 * m0)                          # mass follows the geometry (density mode)
    e.set_property([o.id], "position", [1, 2, 3])
    e.set_property([o.id], "orientation_euler", [0, 0, math.pi / 2])
    e.set_property([o.id], "color", [0.1, 0.2, 0.3])
    e.set_property([o.id], "mass_mode", "mass")
    e.set_property([o.id], "mass_input", 7.5)
    assert np.allclose(s.world_pose(o.id)[0], [1, 2, 3]) and o.mass() == 7.5 and o.color == [0.1, 0.2, 0.3]
    assert o.euler()[2] == pytest.approx(math.pi / 2)
    for _ in range(6):
        e.stack.undo()
    assert o.params["length"] == 2 * length0 and np.allclose(s.world_pose(o.id)[0], [0, 0, 0])


def test_at06_custom_scalar_and_vector_properties_can_be_added(scene):
    s, e = scene
    o = e.add_new("basic.sphere")
    e.add_custom_property([o.id], {"key": "drag_cd", "name": "Drag coefficient", "type": "float", "value": 0.47})
    e.add_custom_property([o.id], {"key": "wind", "name": "Wind", "type": "vec3", "value": [3, 0, 0],
                                   "quantity": "velocity"})
    assert o.get("custom.drag_cd") == 0.47 and o.get("custom.wind") == [3.0, 0.0, 0.0]
    e.set_property([o.id], "custom.wind", [1, 2, 3])
    assert o.get("custom.wind") == [1.0, 2.0, 3.0]
    with pytest.raises(Exception):
        e.set_property([o.id], "custom.wind", [1, 2])               # wrong vector dimension is rejected


def test_at07_a_formula_process_updates_velocity_and_position(scene):
    s, e = scene
    a = e.add_new("body.point_mass", position=[0, 0, 10])
    a.linear_velocity = [2, 0, 3]
    a.linear_acceleration = [0, 0, -9.81]
    e.assign_process("lib.euler_velocity_update", TargetFilter(ids=[a.id]))
    e.assign_process("lib.euler_position_update", TargetFilter(ids=[a.id]))
    eng = make_engine(scene, delta_time=0.01, duration=1.0)
    run(eng)
    v, x = eng.state.v[0], eng.state.x[0]
    assert v == pytest.approx([2, 0, 3 - 9.81]) and x[0] == pytest.approx(2.0)
    assert x[2] < 10 and x[2] == pytest.approx(10 + 3 - 9.81 * 0.01 * sum(range(1, 101)) * 0.01)


# ---- 8–16: simulation timing and controls -------------------------------------------------------------------------

def test_at08_delta_time_and_update_rate_are_independent(window):
    w = window
    w.sim_fields.dt.set_si(0.004)
    w.sim_fields.dt.committed.emit(0.004)
    w.sim_fields.rate.set_si(37.0)
    w.sim_fields.rate.committed.emit(37.0)
    assert w.project.simulation.delta_time == 0.004 and w.project.simulation.update_rate_hz == 37.0
    assert w.project.simulation.time_ratio == pytest.approx(0.148)


def test_at09_all_applicable_processes_execute_every_step(scene):
    s, e = scene
    b = e.add_new("body.point_mass", position=[0, 0, 0])
    b.linear_velocity = [1, 0, 0]
    for pid, params in (("dyn.gravity", {}), ("dyn.damping", {"linear": 0.5}),
                        ("dyn.force_field", {"force": [0, 2, 0], "use_region": False})):
        e.assign_process(pid, TargetFilter(ids=[b.id]), params)
    rec = Recording(interval_steps=1)
    run(make_engine(scene, rec, delta_time=0.01, duration=0.5))
    outputs = [k for k in rec.keys() if k[0] == "proc" and k[2] == b.id and k[3] == "acc"]
    assert len({k[1] for k in outputs}) == 3                           # one output column per process
    for k in outputs:
        assert np.all(np.isfinite(rec.column(k))) and rec.n == 51      # evaluated at every one of the 50 steps


def test_at10_requested_and_actual_update_rate_are_displayed(window):
    w = window
    _ball(w.editor, z=1e6)
    w.project.simulation.duration, w.project.simulation.update_rate_hz = 1000.0, 50.0
    assert w.sim.start()
    assert _wait(lambda: w.sim.last_stats is not None and w.sim.last_stats.steps > 30, 10)
    text = w.sim_fields.status.text()
    w.sim.stop()
    assert "/50 Hz" in text and float(text.split("·")[2].split("/")[0]) == pytest.approx(50, rel=0.3)


def test_at11_the_engine_reports_an_unachievable_rate(window):
    w = window
    _ball(w.editor, z=1e9)
    w.project.simulation.duration, w.project.simulation.update_rate_hz = 1e6, 100000.0
    w.project.simulation.delta_time = 1e-5
    assert w.sim.start()
    assert _wait(lambda: w.sim.last_stats is not None and w.sim.last_stats.overrun, 10)
    text = w.sim_fields.status.text()
    dt = w.sim.runner.engine.clock.delta_time
    w.sim.stop()
    assert "behind schedule" in text and dt == 1e-5                   # Δt is never enlarged


def test_at12_the_clock_advances_by_delta_time_per_completed_step(scene):
    _ball(scene[1])
    eng = make_engine(scene, delta_time=0.0137, duration=1.0)
    times = []
    while eng.step():
        times.append(eng.clock.time)
    steps = np.arange(1, len(times) + 1)
    np.testing.assert_allclose(times[:-1], steps[:-1] * 0.0137, rtol=0, atol=1e-12)
    assert times[-1] == pytest.approx(1.0)        # final partial step ends exactly at the duration


def test_at13_duration_is_measured_in_simulated_time(window):
    w = window
    _ball(w.editor)
    t0 = time.perf_counter()
    _run_window(w, duration=0.25, delta_time=0.001, update_rate_hz=1000.0, pacing=PacingMode.AS_FAST_AS_POSSIBLE)
    wall = time.perf_counter() - t0
    clock = w.sim.runner.engine.clock
    assert clock.time == pytest.approx(0.25) and clock.steps == 250 and wall < 10


def test_at14_rendering_fps_does_not_set_delta_time_or_process_frequency(window):
    w = window
    _ball(w.editor)
    results = []
    for fps in (5.0, 240.0):
        w.sim.reset()
        _run_window(w, duration=0.3, delta_time=0.002, render_fps=fps, pacing=PacingMode.AS_FAST_AS_POSSIBLE)
        eng = w.sim.runner.engine
        results.append((eng.clock.steps, eng.clock.delta_time, eng.state.x.copy()))
    assert results[0][0] == results[1][0] == 150 and results[0][1] == results[1][1] == 0.002
    assert np.array_equal(results[0][2], results[1][2])


def test_at15_pause_resume_stop_reset_and_single_step(window):
    w = window
    b = _ball(w.editor, z=100.0)
    w.project.simulation.duration, w.project.simulation.update_rate_hz = 100.0, 200.0
    w.sim.step()                                                       # single step from idle
    assert _wait(lambda: w.sim.last_stats is not None and w.sim.last_stats.steps == 1)
    w.sim.start()                                                      # resume
    assert _wait(lambda: w.sim.last_stats.steps > 20)
    w.sim.pause()
    assert _wait(lambda: w.sim.state == "paused")
    n = w.sim.last_stats.steps
    QTest.qWait(150)
    assert w.sim.last_stats.steps == n                                 # paused: no progress
    w.sim.step()
    assert _wait(lambda: w.sim.last_stats.steps == n + 1)
    w.sim.stop()
    assert not w.sim.session_active and w.scene.world_pose(b.id)[0][2] < 100
    w.sim.reset()
    assert w.scene.world_pose(b.id)[0][2] == pytest.approx(100.0)


def test_at16_as_fast_as_possible_mode_is_independent_of_real_time_pacing(window):
    w = window
    _ball(w.editor)
    t0 = time.perf_counter()
    _run_window(w, duration=20.0, delta_time=0.01, update_rate_hz=1.0, pacing=PacingMode.AS_FAST_AS_POSSIBLE)
    fast = w.sim.runner.engine.state.x.copy()
    assert time.perf_counter() - t0 < 10                               # 2000 steps at a "1 Hz" rate setting
    w.sim.reset()
    _run_window(w, duration=0.2, delta_time=0.01, update_rate_hz=1000.0, pacing=PacingMode.REAL_TIME)
    w.sim.reset()
    eng = make_engine((w.scene, w.editor), delta_time=0.01, duration=20.0)
    run(eng)
    assert np.array_equal(eng.state.x, fast)


# ---- 17–21: coordinates, trajectories and plots ------------------------------------------------------------------

def test_at17_global_and_local_coordinate_systems_render(window):
    w = window
    w.project.view.grids["XY"].visible = False
    w.project.view.axes.visible = False
    o = w.editor.add_new("basic.cube", position=[0, 0, 0.5])
    o.set_orientation(numerics.quat_from_axis_angle([0, 0, 1], 0.6))
    w.scene.set_selection([])
    w.viewport.frame_objects(None)
    plain = _render(w).copy()
    w.project.view.axes.visible = True
    with_global = _render(w).copy()
    w.editor.set_property([o.id], "overlay.local_axes", True)
    with_local = _render(w)
    assert _pixels_changed(plain, with_global) > 30 and _pixels_changed(with_global, with_local) > 30


def test_at18_relative_coordinate_transformations_are_numerically_tested():
    from psim.coordinates.kinematics import FrameState, point_in_frame, relative_motion
    w, a0, h = np.array([0, 0, 1.3]), np.array([0.5, 0, 0]), 1e-5

    def frame(t):        # a frame translating with constant acceleration and rotating about Z
        q = numerics.quat_from_axis_angle([0, 0, 1], 1.3 * t)
        return FrameState(0.5 * a0 * t * t, np.asarray(q, float), a0 * t, w, a0, np.zeros(3))

    def p(t):
        return np.array([math.cos(t), t, 0.2 * t * t])
    t = 0.7
    v, acc = np.array([-math.sin(t), 1, 0.4 * t]), np.array([-math.cos(t), 0, 0.4])
    rm = relative_motion(p(t), v, acc, frame(t))
    c = lambda tt: point_in_frame(p(tt), frame(tt))  # noqa: E731
    np.testing.assert_allclose(rm.position, c(t), atol=1e-12)
    np.testing.assert_allclose(rm.velocity_relative, (c(t + h) - c(t - h)) / (2 * h), atol=1e-8)
    np.testing.assert_allclose(rm.acceleration_relative, (c(t + h) - 2 * c(t) + c(t - h)) / h ** 2, atol=1e-4)


def test_at19_trajectories_display_using_the_configured_color(window):
    w = window
    w.project.view.axes.visible = False
    w.project.view.grids["XY"].visible = False
    ball = _ball(w.editor, z=0.5, vel=(2.0, 0, 3.0))
    w.editor.set_property([ball.id], "position", [-2, 0, 0.5])
    for path, value in (("trajectory.enabled", True), ("trajectory.color", [1.0, 0.0, 1.0]),
                        ("trajectory.width", 5.0)):
        w.editor.set_property([ball.id], path, value)
    w.viewport.frame_objects(None)
    _run_window(w, duration=0.6, delta_time=0.005, pacing=PacingMode.AS_FAST_AS_POSSIBLE)
    w.viewport.frame_objects(None)
    img = _render(w)
    magenta = sum(1 for x in range(0, img.width(), 2) for y in range(0, img.height(), 2)
                  if (lambda c: c.red() > 180 and c.blue() > 180 and c.green() < 90)(img.pixelColor(x, y)))
    assert magenta > 30 * _size_factor(w), (magenta, w.viewport.size())          # a line: scales with length


def test_at20_2d_plots_support_configurable_axes_and_multiple_objects(window):
    w = window
    a = _ball(w.editor, z=10, vel=(1, 0, 5))
    b = _ball(w.editor, z=5, vel=(2, 0, 0))
    _run_window(w, duration=0.5, delta_time=0.01, pacing=PacingMode.AS_FAST_AS_POSSIBLE)
    spec = PlotSpec("z vs x", "2d", [SeriesSpec(Variable(a.id, "position", "x"), Variable(a.id, "position", "z")),
                                     SeriesSpec(Variable(b.id, "position", "x"), Variable(b.id, "position", "z"))],
                    y_limits=(0.0, 12.0), x_label="Horizontal distance")
    w.show_plots()
    view = w.plot_workspace.add_plot(spec)
    QTest.qWait(100)
    assert len(view.curves) == 2
    xs, zs = view.curves[1].getData()
    rec = w.sim.recording
    np.testing.assert_allclose(zs, rec.column((b.id, "origin"))[:, 2])
    assert "Horizontal distance" in view.item.getAxis("bottom").labelText
    assert view.item.viewRange()[1] == pytest.approx([0.0, 12.0], abs=1e-6)


def test_at21_3d_plots_display_actual_trajectory_data(window):
    w = window
    a = _ball(w.editor, z=10, vel=(1, 2, 5))
    _run_window(w, duration=0.5, delta_time=0.01, pacing=PacingMode.AS_FAST_AS_POSSIBLE)
    w.show_plots()
    w.scene.set_selection([a.id])
    w.plot_workspace.add_quick("trajectory3d")
    QTest.qWait(100)
    spec, view = w.plot_workspace.plots[-1]
    data = series_data(Evaluator(w.sim.recording, w.scene), spec, w.units)[0]
    np.testing.assert_allclose(np.c_[data.x, data.y, data.z], w.sim.recording.column((a.id, "com")))
    line = [it for it in view._items if hasattr(it, "pos")][0]
    assert len(line.pos) == w.sim.recording.n


# ---- 22–30: units, projects, diagnostics, permissions, numerics, responsiveness, memory, reproducibility ---------

def test_at22_units_convert_and_incompatible_formulas_are_reported(scene):
    from psim.simulation.conditions import ConditionError, compile_condition
    u = UnitRegistry()
    assert u.convert(36.0, "km/h", "m/s") == pytest.approx(10.0)
    assert u.convert(1.0, "lbf", "N") == pytest.approx(4.4482216152605)
    with pytest.raises(Exception):
        u.convert(1.0, "m", "s")
    with pytest.raises(ConditionError, match="Incompatible dimensions"):
        compile_condition({"name": "x", "expression": "t > 1[m]", "action": "stop"}, {})


def test_at23_projects_save_and_reopen_with_scene_and_settings_intact(window, tmp_path):
    w = window
    a = _ball(w.editor)
    w.editor.add_frame("Lab", None, [1, 0, 0])
    w.editor.add_measurement("distance", [a.id, w.editor.add_new("basic.cube").id])
    w.project.simulation.delta_time, w.project.simulation.integrator = 0.0025, "velocity_verlet"
    w.project.view.camera.distance = 17.0
    path = str(tmp_path / "keep.psim")
    assert w._save_to(path)
    before = ser.project_to_dict(w.scene, w.units, w.project)
    assert w.open_project(path, confirm=False)
    after = ser.project_to_dict(w.scene, w.units, w.project)
    for d in (before, after):
        d.pop("metadata")
        d.pop("ui", None)
    assert after == before


def test_at24_invalid_formulas_and_conflicting_writes_give_useful_diagnostics(scene):
    from psim.processes.formula_parser import FormulaSyntaxError, parse
    s, e = scene
    o = e.add_new("basic.cube")
    plan = build_plan(s, [ProcessAssignment("PRC-0001", "kin.circular_motion", TargetFilter(ids=[o.id])),
                          ProcessAssignment("PRC-0002", "kin.oscillation", TargetFilter(ids=[o.id]))])
    assert not plan.ok and "Conflicting writes" in plan.errors[0].message and o.id in plan.errors[0].message
    with pytest.raises(FormulaSyntaxError) as info:
        parse("norm(self.velocity * 2")
    assert "column" in str(info.value) and "^" in str(info.value)


def test_at25_ordinary_users_cannot_access_process_authoring(window):
    from psim.processes.formula_process import FormulaProcessSpec, build_process_def, compile_spec
    from psim.processes.permissions import PermissionDenied
    from psim.processes.process_registry import process_registry
    w = window
    assert not w.developer_menu.menuAction().isVisible()
    assert not w.actions_["dev.new"].isEnabled() or not w.developer_menu.menuAction().isVisible()
    spec = FormulaProcessSpec.from_dict({"process_id": "user.hack", "name": "Hack", "version": "1.0.0",
                                         "category": "Test", "kind": "contribution", "mode": "per_object",
                                         "script": "force = self.velocity * 0[kg/s]"})
    cp, issues = compile_spec(spec)
    assert cp is not None, issues
    with pytest.raises(PermissionDenied):
        process_registry().register(build_process_def(cp))
    with pytest.raises(PermissionDenied):
        process_registry().publish(spec, None, "unused")


def test_at26_numerical_integrators_pass_regression_tests(scene):
    s, e = scene
    m = e.add_new("body.point_mass", position=[1, 0, 0])
    e.assign_process("dyn.anchor_spring", TargetFilter(ids=[m.id]), {"anchor": [0, 0, 0], "stiffness": 1.0,
                                                                       "rest_length": 0.0})
    errors = {}
    for integ in ("semi_implicit_euler", "midpoint", "rk4"):
        errs = []
        for dt in (0.02, 0.01):
            eng = make_engine(scene, delta_time=dt, duration=2.0, integrator=integ)
            run(eng)
            errs.append(abs(eng.state.x[0, 0] - math.cos(2.0)))
        errors[integ] = math.log2(errs[0] / errs[1])
    assert errors["semi_implicit_euler"] == pytest.approx(1, abs=0.2)
    assert errors["midpoint"] == pytest.approx(2, abs=0.2)
    assert errors["rk4"] == pytest.approx(4, abs=0.3)


def test_at27_the_gui_remains_responsive_while_simulating(window):
    w = window
    w.editor.add_new("basic.plane", params={"length": 10, "width": 10})
    for k in range(6):
        b = w.editor.add_new("body.rigid_box", position=[0.6 * k - 1.5, 0, 0.3 + 0.1 * k])
        w.editor.assign_process("dyn.gravity", TargetFilter(ids=[b.id]))
    w.editor.set_property([o.id for o in w.scene.objects.values() if o.type_id == "basic.plane"],
                          "physics.collider", "auto")
    w.project.simulation.duration, w.project.simulation.pacing = 1e6, PacingMode.AS_FAST_AS_POSSIBLE
    _render(w)       # the first frame of new objects (mesh upload, shadow map) is not part of the steady state
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
    assert w.sim.start()
    timer.start(10)
    QTest.qWait(1500)                     # a real event loop, as in the application
    timer.stop()
    steps = w.sim.last_stats.steps
    frame = w.viewport._paint_cost        # smoothed wall time of one viewport frame
    w.sim.stop()
    gaps = np.diff(ticks)
    # The event loop keeps being served while the engine thread works: no gap is longer than about one frame, and
    # frame pacing leaves at least as much idle time as a frame takes. (Under software OpenGL a frame of this
    # scene costs ~0.1 s, ~0.25 s at a 2x scale factor on a CI runner; the runner's backpressure and the
    # viewport's frame pacing keep the worker from starving the GUI thread of Python's interpreter lock.) The
    # bounds are those of a 0.1 s frame, scaled up for slower frames.
    slow = max(frame / 0.1, 1.0)
    assert steps >= 10, (steps, len(ticks), frame, gaps.max() if len(gaps) else None)   # the engine progresses
    assert len(ticks) > 10 / slow, (len(ticks), frame)
    # the 95th percentile only means something with enough gaps (with 4 gaps it is the maximum)
    p95 = np.percentile(gaps, 95) if len(gaps) >= 20 else 0.0
    assert p95 < 0.2 * slow and gaps.max() < 0.3 * slow, (p95, gaps.max(), frame, len(gaps))


def test_at28_long_simulations_record_without_excessive_memory(scene):
    _ball(scene[1], z=1e9)
    rec = Recording(max_bytes=64 * 1024, overflow="decimate", interval_steps=1)
    eng = make_engine(scene, rec, delta_time=0.001, duration=200.0)
    run(eng)
    assert eng.clock.steps == 200_000
    assert rec.bytes_used <= rec.max_bytes and rec.decimations >= 5
    t = rec.times()
    assert t[0] == 0 and t[-1] == pytest.approx(200.0) and np.all(np.diff(t[:-1]) > 0)


def test_at29_results_are_identical_when_only_the_render_rate_changes(scene):
    from psim.simulation.runner import SimulationRunner
    s, e = scene
    _ball(e, vel=(1, 2, 3))
    states = []
    for fps in (2.0, 120.0):
        runner = SimulationRunner()
        runner.start(make_engine(scene, delta_time=0.001, duration=0.5, render_fps=fps,
                                 pacing=PacingMode.AS_FAST_AS_POSSIBLE))
        assert runner.wait_until(lambda: runner.engine.finished, 10)
        runner.stop()
        states.append(runner.engine.state.x.copy())
    assert np.array_equal(states[0], states[1])


def test_at30_repeated_runs_are_reproducible(scene):
    s, e = scene
    g = e.add_new("basic.plane", params={"length": 10, "width": 10})
    e.set_property([g.id], "physics.collider", "auto")
    boxes = [e.add_new("body.rigid_box", position=[0.05 * k, 0, 0.21 + 0.45 * k]) for k in range(3)]
    for b in boxes:
        e.assign_process("dyn.gravity", TargetFilter(ids=[b.id]))
    e.assign_process("lib.quadratic_drag", TargetFilter(ids=[boxes[2].id]), {"rho": 1.2, "cd": 1.0, "area": 0.16})
    finals = []
    for _ in range(2):
        eng = make_engine(scene, delta_time=0.005, duration=1.0, integrator="rk4")
        run(eng)
        finals.append([getattr(eng.state, f).copy() for f in ("x", "q", "v", "w")])
    for a, b in zip(*finals):
        assert np.array_equal(a, b)
