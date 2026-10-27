"""Place Relative To: numeric placement against objects, named frames and the global frame."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim import numerics
from psim.scene.placement import PlacementError, current_offset, place

from test_coordinates_gui import owned  # noqa: F401  (fixture)


def _q(axis, ang):
    return np.asarray(numerics.quat_from_axis_angle(axis, ang), float)


def _same_rotation(q1, q2):
    return abs(abs(float(np.dot(q1, q2))) - 1.0) < 1e-12


def test_offset_along_reference_axes_and_global_axes(scene):
    s, e = scene
    ref = e.add_new("basic.cube", position=[1, 2, 3])
    ref.set_orientation(_q([0, 0, 1], math.pi / 2))
    obj = e.add_new("basic.sphere", position=[0, 0, 0])
    q0 = np.asarray(obj.orientation, float).copy()
    e.set_world_poses(place(s, [obj.id], obj.id, ref.id, [1, 0, 0]))
    p, q = s.world_pose(obj.id)
    np.testing.assert_allclose(p, [1, 3, 3], atol=1e-12)                 # reference x axis = global y
    assert _same_rotation(q, q0)                                          # orientation kept
    e.set_world_poses(place(s, [obj.id], obj.id, ref.id, [1, 0, 0], axes="global"))
    np.testing.assert_allclose(s.world_pose(obj.id)[0], [2, 2, 3], atol=1e-12)
    rel = _q([1, 0, 0], 0.3)
    e.set_world_poses(place(s, [obj.id], obj.id, ref.id, [0, 0, 2], rel))
    p, q = s.world_pose(obj.id)
    np.testing.assert_allclose(p, [1, 2, 5], atol=1e-12)
    assert _same_rotation(q, numerics.quat_multiply(ref.orientation, rel))
    d, r = current_offset(s, obj.id, ref.id)                              # round trip
    np.testing.assert_allclose(d, [0, 0, 2], atol=1e-12)
    assert _same_rotation(r, rel)
    d, _ = current_offset(s, obj.id, ref.id, "global")
    np.testing.assert_allclose(d, [0, 0, 2], atol=1e-12)
    e.set_world_poses(place(s, [obj.id], obj.id, None, [5, 0, 0]))       # global frame
    np.testing.assert_allclose(s.world_pose(obj.id)[0], [5, 0, 0], atol=1e-12)


def test_group_moves_rigidly_and_children_follow(scene):
    s, e = scene
    ref = e.add_new("basic.cube", position=[0, 0, 0])
    a = e.add_new("basic.sphere", position=[1, 0, 0])
    b = e.add_new("basic.sphere", position=[1, 1, 0])
    child = e.add_new("basic.sphere", position=[1, 0, 1])
    e.reparent([child.id], a.id)
    before = {i: s.world_pose(i) for i in (a.id, b.id, child.id)}
    poses = place(s, [a.id, b.id, child.id], a.id, ref.id, [0, 0, 5], _q([0, 0, 1], math.pi))
    assert set(poses) == {a.id, b.id}                                     # the child follows its parent
    e.set_world_poses(poses, "Place")
    np.testing.assert_allclose(s.world_pose(a.id)[0], [0, 0, 5], atol=1e-12)
    for i, j in ((a.id, b.id), (a.id, child.id)):                       # relative poses unchanged
        d0 = before[j][0] - before[i][0]
        d1 = s.world_pose(j)[0] - s.world_pose(i)[0]
        assert np.linalg.norm(d1) == pytest.approx(np.linalg.norm(d0))
    np.testing.assert_allclose(s.world_pose(b.id)[0], [0, -1, 5], atol=1e-12)   # turned 180° about z with A
    e.stack.undo()                                                        # one undo step
    for i in (a.id, b.id, child.id):
        np.testing.assert_allclose(s.world_pose(i)[0], before[i][0], atol=1e-12)


def test_invalid_references_and_inputs(scene):
    s, e = scene
    a = e.add_new("basic.cube", position=[0, 0, 0])
    child = e.add_new("basic.sphere", position=[0, 0, 1])
    e.reparent([child.id], a.id)
    other = e.add_new("basic.sphere", position=[3, 0, 0])
    fid = e.add_frame("On A", parent=a.id)
    for ref, msg in ((a.id, "itself"), (child.id, "itself"), ("OBJ-999999", "Missing"), ("FRM-9999", "Missing")):
        with pytest.raises(PlacementError, match=msg):
            place(s, [a.id], a.id, ref, [0, 0, 0])
    with pytest.raises(PlacementError, match="attached"):
        place(s, [a.id], a.id, fid, [0, 0, 0])
    with pytest.raises(PlacementError, match="primary"):
        place(s, [a.id], other.id, None, [0, 0, 0])
    with pytest.raises(PlacementError, match="finite"):
        place(s, [a.id], a.id, other.id, [0, math.nan, 0])
    with pytest.raises(PlacementError, match="non-zero"):
        place(s, [a.id], a.id, other.id, [0, 0, 0], [0, 0, 0, 0])
    with pytest.raises(PlacementError, match="Select"):
        place(s, [], None, None, [0, 0, 0])
    e.set_property([other.id], "locked", True)
    with pytest.raises(PlacementError, match="locked"):
        place(s, [other.id], other.id, None, [0, 0, 0])


def test_place_relative_dialog(scene, owned):  # noqa: F811
    from psim.gui.dialogs.placement_dialog import PlaceRelativeDialog
    from psim.units.unit_registry import UnitRegistry
    s, e = scene
    ref = e.add_new("basic.cube", position=[1, 0, 0])
    obj = e.add_new("basic.sphere", position=[1, 0, 2])
    child = e.add_new("basic.sphere", position=[1, 0, 3])
    e.reparent([child.id], obj.id)
    dlg = PlaceRelativeDialog(s, e, UnitRegistry(), [obj.id])
    owned(dlg)
    refs = [dlg.reference.itemData(k) for k in range(dlg.reference.count())]
    assert obj.id not in refs and child.id not in refs and None in refs   # never itself or its children
    assert dlg.reference.currentData() == ref.id                          # first other object by default
    np.testing.assert_allclose(dlg.offset.si(), [0, 0, 2], atol=1e-12)     # pre-filled with the current offset
    assert not dlg.euler.isEnabled()
    dlg.offset.set_si([0.5, 0, 0])
    dlg.orient_mode.setCurrentIndex(dlg.orient_mode.findData("set"))
    assert dlg.euler.isEnabled()
    dlg.euler.set_si([0, 0, math.pi / 2])
    assert dlg.apply()
    p, q = s.world_pose(obj.id)
    np.testing.assert_allclose(p, [1.5, 0, 0], atol=1e-12)
    assert _same_rotation(q, _q([0, 0, 1], math.pi / 2))
    np.testing.assert_allclose(s.world_pose(child.id)[0], [1.5, 0, 1], atol=1e-12)
    assert "Place Relative to" in e.stack.undoText()
    dlg.reference.setCurrentIndex(dlg.reference.findData(None))           # switching refills the offset
    np.testing.assert_allclose(dlg.offset.si(), [1.5, 0, 0], atol=1e-12)
    e.set_property([obj.id], "locked", True)
    dlg2 = PlaceRelativeDialog(s, e, UnitRegistry(), [obj.id])
    owned(dlg2)
    assert "locked" in dlg2.status.text() and not dlg2.buttons.buttons()[0].isEnabled()


def test_place_relative_action_in_window(window):  # noqa: F811
    w = window
    a = w.editor.add_new("basic.cube", position=[0, 0, 0])
    w.scene.set_selection([a.id])
    assert w.actions_["obj.place_relative"].isEnabled()
    menu_texts = [act.text() for m in w.menuBar().actions() if m.menu() for act in m.menu().actions()]
    assert "&Place Relative To…" in menu_texts


from test_gui import window  # noqa: E402,F401  (fixture)
