"""Phase 8 GUI: stop-conditions dialog, simulation-state actions, measurements, performance profile."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt

from test_coordinates_gui import owned  # noqa: F401  (fixture)
from test_gui import window  # noqa: F401  (fixture)


def test_stop_conditions_dialog_checks_and_returns_conditions(scene, owned):  # noqa: F811
    from psim.gui.dialogs.stop_conditions_dialog import StopConditionsDialog
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 3])
    dlg = StopConditionsDialog(s, [{"name": "Ground", "expression": f'object("{ball.id}").position.z < 0[m]',
                                    "action": "stop", "enabled": True}])
    owned(dlg)
    assert dlg.check() and "1 enabled condition checked" in dlg.status.text()
    dlg.add_row()
    dlg.table.item(1, 2).setText(f'object("{ball.id}").position.z < 2[s]')
    assert not dlg.check() and "Incompatible dimensions" in dlg.status.text()
    dlg.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)         # disabled: not compiled
    assert dlg.check()
    dlg.table.item(1, 1).setText("ground")                               # duplicate name (case-insensitive)
    assert not dlg.check() and "Duplicate" in dlg.status.text()
    dlg.table.selectRow(1)
    dlg._remove()
    dlg._accept()
    assert dlg.result_conditions == [{"name": "Ground", "expression": f'object("{ball.id}").position.z < 0[m]',
                                      "action": "stop", "enabled": True}]


def test_performance_panel_shows_the_stage_profile(qapp, owned):  # noqa: F811
    from psim.gui.bottom_panels import PerformancePanel
    from psim.simulation.runner import RunStats
    p = PerformancePanel()
    owned(p)
    prof = {"updates": 0.0, "processes_and_integration": 0.25, "prescribed": 0.0, "constraints": 1.5,
            "recording": 0.1, "conditions": 0.01, "total": 1.86, "steps": 10}
    p.update_sim_stats(RunStats("running", 0.1, 10, 0.01, 1.0, "afap", 100, 900, 9, 0.01, 0, False, 1.9, prof))
    text = p.labels["profile"].text()
    assert "constraints 1.500" in text and "total 1.860" in text and "updates" not in text


def test_objects_in_rect_selects_by_projected_centre(scene):
    from psim.rendering.camera import OrbitCamera
    from psim.rendering.picking import objects_in_rect
    s, e = scene
    a = e.add_new("basic.cube", position=[-2, 0, 0.5])
    b = e.add_new("basic.cube", position=[2, 0, 0.5])
    hidden = e.add_new("basic.sphere", position=[-2, 0, 0.5])
    e.set_property([hidden.id], "visible", False)
    cam = OrbitCamera()
    w, h = 800, 600
    (xa, ya), (xb, yb) = cam.project(np.array([[-2, 0, 0.5], [2, 0, 0.5]]), w, h)[0]
    assert objects_in_rect(s, cam, w, h, xa - 5, ya - 5, xa + 5, ya + 5) == [a.id]       # hidden ones excluded
    both = objects_in_rect(s, cam, w, h, xb + 5, yb + 5, xa - 5, ya - 5)                # any corner order
    assert set(both) == {a.id, b.id}
    assert objects_in_rect(s, cam, w, h, 0, 0, 3, 3) == []


def test_box_selection_in_window(window):  # noqa: F811
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from test_gui import _move, _render, _screen_pos
    w = window
    a = w.editor.add_new("basic.cube", position=[-1.5, 0, 0.5])
    b = w.editor.add_new("basic.cube", position=[1.5, 0, 0.5])
    w.viewport.frame_objects(None)
    _render(w)
    vp = w.viewport
    w.actions_["tool.select"].trigger()
    pa, pb = _screen_pos(w, [-1.5, 0, 0.5]), _screen_pos(w, [1.5, 0, 0.5])
    left = min(pa.x(), pb.x())
    start = QPoint(left - 60, max(pa.y(), pb.y()) + 60)
    if vp.pick_at(start) is not None:
        start = QPoint(3, vp.height() - 3)
    end = QPoint((pa.x() + pb.x()) // 2, min(pa.y(), pb.y()) - 60)
    cam_before = vp.camera.eye().copy()
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for t in np.linspace(0, 1, 6)[1:]:
        _move(vp, QPoint(int(start.x() + (end.x() - start.x()) * t), int(start.y() + (end.y() - start.y()) * t)))
    assert vp._box is not None
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    assert w.scene.selection == [a.id] and b.id not in w.scene.selection and vp._box is None
    assert np.allclose(vp.camera.eye(), cam_before)                  # no orbit in the Select tool
    w.actions_["tool.translate"].trigger()                          # other tools: the same drag orbits
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for t in np.linspace(0, 1, 6)[1:]:
        _move(vp, QPoint(int(start.x() + (end.x() - start.x()) * t), int(start.y() + (end.y() - start.y()) * t)))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    assert not np.allclose(vp.camera.eye(), cam_before) and w.scene.selection == [a.id]


def test_measuring_between_surface_points_in_the_viewport(window):  # noqa: F811
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from test_gui import _render, _screen_pos
    w = window
    a = w.editor.add_new("basic.cube", position=[-1.5, 0, 0.5])
    b = w.editor.add_new("basic.cube", position=[1.5, 0, 0.5])
    w.viewport.frame_objects(None)
    _render(w)
    vp = w.viewport

    def click(pos):
        QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
        QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)

    w.actions_["tools.measure_point_distance"].trigger()
    assert vp.picking_points and "click the first point" in w.statusBar().currentMessage()
    click(QPoint(2, 2))                                             # empty space: nothing picked
    assert vp.picking_points and not vp._point_pick["points"]
    # each cube is framed before its click, so the face centre lies far from its edges (beyond the snap radius)
    # whatever the window size
    vp.frame_objects([a.id])
    _render(w)
    click(_screen_pos(w, [-1.5, 0, 1.0]))                           # top face of A
    assert "second point" in w.statusBar().currentMessage()
    vp.grab_image()                                                 # paints the pending marker
    vp.frame_objects([b.id])
    _render(w)
    click(_screen_pos(w, [1.5, 0, 1.0]))                            # top face of B
    assert not vp.picking_points and len(w.scene.measurements) == 1
    m = w.scene.measurements[0]
    assert [p["object"] for p in m.points] == [a.id, b.id] and all(p["at"] == "point" for p in m.points)
    from psim.scene.measurements import evaluate
    val = evaluate(w.scene, m)
    np.testing.assert_allclose([p[2] for p in val.points], 1.0, atol=1e-9)    # on the top surfaces
    assert val.value == pytest.approx(3.0, abs=0.05)
    # angle via the panel, cancelled with Esc after one point; selection is never changed by picking
    w.scene.set_selection([a.id])
    panel = w.measurements_panel
    panel.at.setCurrentIndex(panel.at.findData("point"))
    assert panel.add_angle.isEnabled() and "click the points" in panel.hint.text()
    panel.add_angle.click()
    assert vp.picking_points and "point A" in w.statusBar().currentMessage()
    click(_screen_pos(w, [1.5, 0, 1.0]))
    QTest.keyClick(vp, Qt.Key.Key_Escape)
    assert not vp.picking_points and len(w.scene.measurements) == 1 and w.scene.selection == [a.id]
    w.editor.stack.undo()
    assert not w.scene.measurements
