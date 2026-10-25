"""End-to-end GUI tests (need an OpenGL 3.3 display; run with `xvfb-run -a python -m pytest` on headless Linux)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from psim.rendering.viewport import LIBRARY_MIME


@pytest.fixture
def window(qapp, requires_gl, tmp_path):
    from psim.gui import theme
    from psim.gui.main_window import MainWindow
    theme.apply(qapp, "dark")
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow(settings, check_recovery=False)
    w.resize(1500, 950)
    w.show()
    QTest.qWaitForWindowExposed(w)
    QTest.qWait(150)
    yield w
    w.undo.setClean()
    w._extra_dirty = False
    w.close()


def _render(w):
    w.viewport.update()
    QTest.qWait(80)
    return w.viewport.grabFramebuffer()


def _screen_pos(w, world):
    vp = w.viewport
    xy, ok = vp.camera.project(np.asarray(world, float)[None, :], vp.width(), vp.height())
    assert ok[0]
    return QPoint(int(round(xy[0][0])), int(round(xy[0][1])))


def _move(vp, pt, buttons=Qt.MouseButton.LeftButton):
    ev = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(pt), QPointF(vp.mapToGlobal(pt)), Qt.MouseButton.NoButton,
                     buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(vp, ev)


def test_application_launches_with_all_menus(window):
    titles = [a.text().replace("&", "") for a in window.menuBar().actions()]
    assert titles == ["File", "Edit", "View", "Scene", "Objects", "Physics", "Processes", "Coordinates", "Plots",
                      "Units", "Simulation", "Tools", "Window", "Help"]
    assert window.viewport.renderer.ready, window.viewport.renderer.error
    assert window.library_panel.findChildren(type(window.library_panel).__mro__[0]) is not None


def test_objects_render_in_3d(window):
    bg = _render(window).pixelColor(window.viewport.width() // 2, window.viewport.height() // 2)
    o = window.editor.add_new("basic.sphere", position=[0, 0, 0.5])
    o.color = [0.9, 0.1, 0.1]
    window.scene.set_selection([])  # no gizmo / highlight over the sampled pixels
    window.viewport.frame_objects([o.id])
    img = _render(window)
    c = img.pixelColor(window.viewport.width() // 2, window.viewport.height() // 2)
    assert c != bg
    assert c.red() > c.green() + 40 and c.red() > c.blue() + 40, f"sphere should be red, got {c.name()}"
    # shading varies across the sphere surface (genuine 3D lighting, not a flat icon)
    cam = window.viewport.camera
    centre = np.array([0, 0, 0.5])
    pts = [centre + 0.4 * d for d in (cam.right(), -cam.right(), cam.up(), -cam.up())]
    samples = [QColor(img.pixelColor(_screen_pos(window, p))).lightness() for p in pts]
    assert max(samples) - min(samples) > 15, samples


def test_drag_and_drop_from_library(window):
    vp = window.viewport
    window.project.view.snap.surface_snap = False
    mime = QMimeData()
    mime.setData(LIBRARY_MIME, json.dumps({"key": "type:basic.cylinder"}).encode())
    pos = QPointF(vp.width() * 0.5, vp.height() * 0.6)
    for E in (QDragEnterEvent, QDragMoveEvent):
        QApplication.sendEvent(vp, E(pos.toPoint(), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                                     Qt.KeyboardModifier.NoModifier))
    assert vp._ghost is not None
    QApplication.sendEvent(vp, QDropEvent(pos, Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                                          Qt.KeyboardModifier.NoModifier))
    assert len(window.scene) == 1
    obj = window.scene.primary
    assert obj.type_id == "basic.cylinder" and obj.id == "OBJ-000001"
    z = window.scene.world_pose(obj.id)[0][2]
    assert z == pytest.approx(0.5)  # resting on the XY plane (height 1.0)
    assert vp._ghost is None


def test_click_selection_and_gizmo_drag(window):
    w = window
    a = w.editor.add_new("basic.cube", position=[0, 0, 0.5])
    w.viewport.frame_objects(None)
    _render(w)
    vp = w.viewport
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(3, 3))
    assert w.scene.selection == []
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _screen_pos(w, [0, 0, 0.5]))
    assert w.scene.selection == [a.id]
    vp.set_tool("translate")
    _render(w)
    pivot, axes, size = vp._gizmo_frame()
    start = _screen_pos(w, pivot + axes[:, 1] * size * 0.9)
    end = _screen_pos(w, pivot + axes[:, 1] * size * 0.9 + [0, 2.0, 0])
    assert vp._pick_gizmo(start) == "t_y"
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for t in np.linspace(0, 1, 6)[1:]:
        _move(vp, QPoint(int(start.x() + (end.x() - start.x()) * t), int(start.y() + (end.y() - start.y()) * t)))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    np.testing.assert_allclose(w.scene.world_pose(a.id)[0], [0, 2.0, 0.5], atol=1e-9)
    assert w.undo.text(w.undo.index() - 1) == "Move"
    w.undo.undo()
    np.testing.assert_allclose(w.scene.world_pose(a.id)[0], [0, 0, 0.5], atol=1e-9)


def test_locked_objects_have_no_gizmo(window):
    a = window.editor.add_new("basic.cube")
    window.scene.set_selection([a.id])
    assert window.viewport._gizmo_frame() is not None
    window.editor.set_property([a.id], "locked", True)
    assert window.viewport._gizmo_frame() is None


def test_inspector_edits_properties(window):
    from psim.gui.widgets.inputs import ColorButton, NumberField, VectorField
    w = window
    o = w.editor.add_new("basic.sphere")
    w.scene.set_selection([o.id])
    QTest.qWait(50)
    insp = w.inspector
    assert insp.title.text() == o.name
    # geometry: radius field
    radius = next(f for f in insp.findChildren(NumberField) if f.text() == "0.5")
    radius.lineEdit().selectAll()
    QTest.keyClicks(radius.lineEdit(), "0.8")
    QTest.keyClick(radius.lineEdit(), Qt.Key.Key_Return)
    assert o.params["radius"] == pytest.approx(0.8) or w.scene.get(o.id).params["radius"] == pytest.approx(0.8)
    # transform: position vector (first VectorField after the frame selector)
    pos = insp.findChildren(VectorField)[0]
    pos.fields[2].lineEdit().selectAll()
    QTest.keyClicks(pos.fields[2].lineEdit(), "2.5")
    QTest.keyClick(pos.fields[2].lineEdit(), Qt.Key.Key_Return)
    assert w.scene.world_pose(o.id)[0][2] == pytest.approx(2.5)
    # colour
    btn = insp.findChildren(ColorButton)[0]
    btn.set_rgb([0.1, 0.9, 0.2])
    btn.colorChanged.emit(btn.rgb())
    assert w.scene.get(o.id).color == pytest.approx([0.1, 0.9, 0.2])
    # display units: show position in millimetres without changing the stored value
    w.units.set_display_unit("position", "length", "mm")
    insp._signature = None
    insp._sync()
    pos = insp.findChildren(VectorField)[0]
    assert pos.fields[2].text() == "2500"
    assert w.scene.world_pose(o.id)[0][2] == pytest.approx(2.5)
    w.units.set_display_unit("position", "length", None)


def test_custom_property_via_editor_appears_in_inspector(window):
    o = window.editor.add_new("basic.cube")
    window.editor.add_custom_property([o.id], {"key": "stiffness", "name": "Stiffness", "type": "vec3",
                                               "value": [1, 2, 3], "quantity": "force"})
    window.scene.set_selection([o.id])
    QTest.qWait(50)
    labels = [lab.text() for lab in window.inspector.findChildren(type(window.inspector.title))]
    assert "Stiffness" in labels


def test_save_and_reopen_through_window(window, tmp_path):
    w = window
    w.new_project(template="mechanical")
    n = len(w.scene)
    w.project.view.grids["XZ"].visible = True
    path = str(tmp_path / "demo.psim")
    assert w._save_to(path)
    w.new_project()
    assert len(w.scene) == 0
    assert w.open_project(path, confirm=False)
    assert len(w.scene) == n and w.project.view.grids["XZ"].visible
    assert w.actions_["grid.XZ"].isChecked()
    assert not w.is_dirty()


def test_library_contains_thumbnails_for_every_item(window):
    lib = window.library
    for it in lib.items():
        icon = window.library_panel.thumbnail(it.key)
        assert not icon.isNull(), it.key
    assert {it.category for it in lib.items()} >= {"Basic Solids", "Composite Shapes", "Mechanical Components",
                                                   "Rigid Bodies", "Reference Frames", "Vectors & Measurement"}


def test_composite_template_placement(window):
    objs = window.add_library_item("composite:dumbbell", position=np.array([1.0, 2.0, 0.5]))
    assert len(objs) == 4
    root = next(o for o in objs if o.type_id == "group.assembly")
    np.testing.assert_allclose(window.scene.world_pose(root.id)[0], [1, 2, 0.5])
    assert all(o.parent_id == root.id for o in objs if o is not root)


def test_light_theme_and_grid_toggle(window, qapp):
    from psim.gui import theme
    window.set_theme("light")
    assert theme.current().name == "light"
    window.actions_["grid.XY"].trigger()
    assert window.project.view.grids["XY"].visible is False
    window.set_theme("dark")


def test_render_rate_does_not_change_scene_state(window):
    """Rendering frequency is independent of model state (precondition for §9.6)."""
    o = window.editor.add_new("basic.cube", position=[1, 2, 3])
    before = window.scene.snapshot([o.id])
    window.viewport.set_continuous(True, 120)
    QTest.qWait(200)
    window.viewport.set_continuous(False)
    assert window.scene.snapshot([o.id]) == before


def test_color_of_rendered_ground_shadow(window):
    """Shadows are rendered: the shadow side of a cube on the ground is darker than open ground."""
    w = window
    w.editor.add_new("basic.cuboid", params={"length": 2, "width": 2, "height": 2}, position=[0, 0, 1.0])
    env = w.project.view.environment
    env.light_azimuth_deg, env.light_elevation_deg = 0.0, 45.0
    w.viewport.set_standard_view("top")
    w.viewport.camera.state.distance = 9
    w.viewport.camera.target = [0, 0, 0]
    img = _render(w)
    shadow = img.pixelColor(_screen_pos(w, [-1.6, 0, 0]))
    ground = img.pixelColor(_screen_pos(w, [1.6, 0.0, 0]))
    assert QColor(shadow).lightness() < QColor(ground).lightness() - 8


def test_camera_follow_modes_do_not_modify_objects(window):
    w = window
    o = w.editor.add_new("basic.cube", position=[1, 1, 0.5])
    w.scene.set_selection([o.id])
    vp = w.viewport
    vp.set_follow("follow", o.id)
    state_before = w.scene.snapshot([o.id])
    w.editor.set_property([o.id], "position", [4, -2, 0.5])
    _render(w)
    np.testing.assert_allclose(vp.camera.target, [4, -2, 0.5], atol=1e-9)
    eye = vp.camera.eye().copy()
    vp.set_follow("look_at", o.id)
    w.editor.set_property([o.id], "position", [0, 3, 0.5])
    _render(w)
    np.testing.assert_allclose(vp.camera.eye(), eye, atol=1e-6)  # camera stays put, but looks at the object
    np.testing.assert_allclose(vp.camera.target, [0, 3, 0.5], atol=1e-9)
    vp.set_follow("none", None)
    assert w.scene.get(o.id).position == [0.0, 3.0, 0.5]
    assert state_before[o.id]["name"] == w.scene.get(o.id).name
