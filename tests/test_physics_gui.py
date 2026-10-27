"""Phase 7 GUI: joints dialog, inspector collision section, solver settings, physics overlays."""
from __future__ import annotations

import math

import numpy as np
import pytest
from PySide6.QtTest import QTest

from psim import numerics
from psim.physics.colliders import collider_for
from psim.processes.process_base import TargetFilter
from psim.rendering import physics_overlay
from psim.units.unit_registry import UnitRegistry

from test_coordinates_gui import owned  # noqa: F401  (fixture)
from test_gui import window  # noqa: F401  (fixture)
from test_recording import make_engine


def test_joints_dialog_creates_edits_and_undoes(scene, owned):  # noqa: F811
    from psim.gui.dialogs.joints_dialog import JointsDialog
    s, e = scene
    box = e.add_new("body.rigid_box", position=[1, 0, 2])
    box.set_orientation(numerics.quat_from_axis_angle([0, 0, 1], math.pi / 2))
    other = e.add_new("basic.sphere", position=[3, 0, 2])
    s.set_selection([box.id])
    depth = e.stack.count()
    dlg = JointsDialog(s, e, UnitRegistry())
    owned(dlg)
    dlg._add()
    j = dlg.joints[0]
    assert j["body_a"] == box.id and j["body_b"] is None and j["type"] == "revolute"
    dlg.anchor.committed.emit([1.0, 0.0, 3.0])            # hinge 1 m above the box …
    dlg.axis.committed.emit([1.0, 0.0, 0.0])              # … about global X
    j = dlg.joints[0]
    np.testing.assert_allclose(j["anchor_b"], [1, 0, 3], atol=1e-12)          # world anchor (body B = world)
    np.testing.assert_allclose(j["anchor_a"], [0, 0, 1], atol=1e-12)          # in the box's (turned) frame
    np.testing.assert_allclose(j["axis_a"], [0, -1, 0], atol=1e-12)           # global X in the box frame
    dlg.limit.setChecked(True)
    dlg.lower.committed.emit(-0.5)
    dlg.upper.committed.emit(0.5)
    assert dlg.lower.isVisibleTo(dlg) and not dlg.length.isVisibleTo(dlg)
    assert dlg.apply()
    assert e.stack.count() == depth + 1
    stored = s.joints[0]
    assert stored.limit_enabled and stored.lower == -0.5 and stored.upper == 0.5
    # a rope between the two bodies, anchored at their centres: the length is the current distance
    dlg._add()
    dlg.type.setCurrentText("distance")
    dlg._store("type", "distance")
    assert dlg.length.isVisibleTo(dlg) and not dlg.lower.isVisibleTo(dlg)
    dlg.body_b.setCurrentIndex(dlg.body_b.findData(other.id))
    dlg.body_b.activated.emit(dlg.body_b.currentIndex())
    dlg.anchor_b.committed.emit([3.0, 0.0, 2.0])
    assert dlg.joints[1]["length"] == pytest.approx(2.0)
    dlg.rope.setChecked(True)
    assert dlg.apply()
    assert [j.type for j in s.joints] == ["revolute", "distance"] and s.joints[1].rope
    e.stack.undo()
    assert len(s.joints) == 1
    # same body on both ends is rejected without changing the joint
    dlg2 = JointsDialog(s, e, UnitRegistry())
    owned(dlg2)
    dlg2.body_b.setCurrentIndex(dlg2.body_b.findData(box.id))
    dlg2.body_b.activated.emit(dlg2.body_b.currentIndex())
    assert "different" in dlg2.error.text() and dlg2.joints[0]["body_b"] is None


def test_inspector_collision_section(scene, owned):  # noqa: F811
    from PySide6.QtWidgets import QComboBox, QLabel, QPushButton

    from psim.gui.properties_panel import PropertiesPanel
    s, e = scene
    torus = e.add_new("basic.torus", position=[0, 0, 1])
    panel = PropertiesPanel(s, e, UnitRegistry())
    owned(panel)
    s.set_selection([torus.id])
    panel._sync()
    texts = [lab.text() for lab in panel.findChildren(QLabel)]
    assert any("Does not collide" in t for t in texts)        # a plain shape needs an explicit collider
    combo = next(c for c in panel.findChildren(QComboBox) if c.findText("capsule") >= 0)
    combo.setCurrentText("auto")
    combo.activated.emit(combo.currentIndex())
    assert torus.physics_setting("collider") == "auto"
    panel._sync()
    texts = [lab.text() for lab in panel.findChildren(QLabel)]
    assert any(t.startswith("Collides as: ") and "convex parts" in t for t in texts)   # a torus is split into convex parts
    e.add_joint("ball", torus.id, None, [0, 0, 2])
    panel._sync()
    assert any("Ball 1" in lab.text() for lab in panel.findChildren(QLabel))
    got = []
    panel.jointsRequested.connect(got.append)
    next(b for b in panel.findChildren(QPushButton) if b.text() == "Joints…").click()
    assert got == [torus.id]


def test_settings_dialog_edits_solver_and_integrator(qapp, owned):  # noqa: F811
    from psim.gui.dialogs.simulation_settings_dialog import SimulationSettingsDialog
    from psim.simulation.settings import SimulationSettings
    dlg = SimulationSettingsDialog(SimulationSettings(), running=False)
    owned(dlg)
    dlg.integrator.setCurrentIndex(dlg.integrator.findData("dopri5"))
    dlg.tolerance.set_si(1e-10)
    dlg.gyroscopic.setChecked(False)
    dlg.physics.setChecked(False)
    dlg.iterations.setValue(40)
    dlg.pos_iterations.setValue(0)
    dlg.baumgarte.set_si(0.5)
    dlg.slop.set_si(0.001)
    dlg.rest_threshold.set_si(0.5)
    dlg._accept()
    r = dlg.result_settings
    assert r is not None
    assert (r.integrator, r.adaptive_tolerance, r.gyroscopic, r.physics_enabled) == ("dopri5", 1e-10, False, False)
    assert (r.solver_iterations, r.position_iterations, r.baumgarte, r.contact_slop, r.restitution_threshold) == \
        (40, 0, 0.5, 0.001, 0.5)


def test_overlay_geometry_for_shapes_joints_and_contacts(scene):
    s, e = scene
    ball = e.add_new("basic.sphere", position=[0, 0, 1], params={"radius": 0.5})
    cap = e.add_new("basic.capsule", position=[2, 0, 1])
    box = e.add_new("body.rigid_box", position=[-2, 0, 1])
    ground = e.add_new("basic.plane")
    for o in (ball, cap, ground):
        e.set_property([o.id], "physics.collider", "auto")
    e.set_property([ball.id], "physics.show_collider", True)
    pts, cols = physics_overlay.collider_lines(s, show_all=False)
    assert len(pts) == 1                                       # only the object with "Show collider"
    r = np.linalg.norm(pts[0] - [0, 0, 1], axis=1)
    np.testing.assert_allclose(r, 0.5, atol=1e-9)              # three great circles of the sphere
    pts, cols = physics_overlay.collider_lines(s, show_all=True)
    assert len(pts) == 4 and all(len(p) % 2 == 0 for p in pts)
    shape = collider_for(box)
    seg = physics_overlay.shape_lines(shape, [-2, 0, 1], np.eye(3))
    np.testing.assert_allclose(seg.min(axis=0), [-2, 0, 1] - shape.half_extents, atol=1e-9)
    e.add_joint("revolute", box.id, None, [-2, 0, 2], [0, 1, 0])
    jp, jc = physics_overlay.joint_lines(s, 0.05)
    assert len(jp) == 1 and np.any(np.all(np.isclose(jp[0], [-2, 0, 2]), axis=1))
    cp, cc = physics_overlay.contact_lines([(np.zeros(3), np.array([0, 0, 1.0]), 100.0)], 0.01)
    np.testing.assert_allclose(cp[0][-1], [0, 0, 2.0])         # normal scaled by the force (0.02 m/N)


def test_engine_snapshot_reports_contacts(scene):
    s, e = scene
    g = e.add_new("basic.plane", params={"length": 10, "width": 10})
    e.set_property([g.id], "physics.collider", "auto")
    box = e.add_new("body.rigid_box", position=[0, 0, 0.2], params={"length": 0.4, "width": 0.4, "height": 0.4})
    e.assign_process("dyn.gravity", TargetFilter(ids=[box.id]))
    eng = make_engine(scene, delta_time=0.005, duration=1.0)
    for _ in range(100):
        eng.step()
    snap = eng.snapshot()
    assert len(snap.contacts) == 4                              # the four bottom corners
    weight = box.mass() * 9.80665
    assert sum(f for _p, _n, f in snap.contacts) == pytest.approx(weight, rel=1e-3)
    for p, n, _f in snap.contacts:
        assert abs(p[2]) < 0.01 and abs(abs(n[2]) - 1) < 1e-9


def test_physics_menu_and_overlay_toggles(window):  # noqa: F811
    w = window
    menu = next(a.menu() for a in w.menuBar().actions() if a.text() == "&Physics")
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert "&Joints…" in texts and not any(t.startswith("Planned") for t in texts)
    assert all(a.isEnabled() for a in menu.actions() if not a.isSeparator() and a.text() != "&Mass Properties Report…")
    env = w.project.view.environment
    w.actions_["physics.show_colliders"].setChecked(True)
    assert env.show_colliders
    w.actions_["physics.show_contacts"].setChecked(False)
    assert not env.show_contacts
    g = w.editor.add_new("basic.plane", params={"length": 6, "width": 6})
    w.editor.set_property([g.id], "physics.collider", "auto")
    b = w.editor.add_new("body.rigid_box", position=[0, 0, 1])
    w.editor.add_joint("ball", b.id, None, [0, 0, 2])
    w.viewport.update()
    QTest.qWait(80)
    assert w.viewport.grabFramebuffer().width() > 0
