"""Reference-frame GUI: frames dialog, inspector frames and vector representation, Relative Motion panel."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim import numerics
from psim.units.unit_registry import UnitRegistry


@pytest.fixture
def owned(qapp):
    """Destroy widgets while the QApplication is alive (not during interpreter shutdown)."""
    import shiboken6
    widgets = []
    yield widgets.append
    for w in widgets:
        if shiboken6.isValid(w):
            shiboken6.delete(w)


def test_frames_dialog_applies_as_one_undo_step(scene, owned):
    from psim.gui.dialogs.frames_dialog import ReferenceFramesDialog
    s, e = scene
    o = e.add_new("basic.cube", position=[2, 0, 0])
    o.set_orientation(numerics.quat_from_axis_angle([0, 0, 1], math.pi / 2))
    s.set_selection([o.id])
    depth = e.stack.count()
    dlg = ReferenceFramesDialog(s, e, UnitRegistry())
    owned(dlg)
    dlg._add()
    dlg.position.committed.emit([1.0, 1.0, 0.0])          # global frame at (1, 1, 0)
    dlg._add_at_selection()
    assert dlg.frames[1]["parent"] == o.id and dlg.frames[1]["name"] == "Frame 2"
    assert dlg.apply()
    assert e.stack.count() == depth + 1
    f1, f2 = (f.id for f in s.frames)
    np.testing.assert_allclose(s.frame_world_pose(f1)[0], [1, 1, 0])
    np.testing.assert_allclose(s.frame_world_pose(f2)[0], [2, 0, 0])
    # re-attaching keeps the frame where it is (pose converted into the new parent's frame)
    dlg.list.setCurrentRow(0)
    dlg.parent_combo.setCurrentIndex(dlg.parent_combo.findData(o.id))
    assert dlg.apply()
    np.testing.assert_allclose(s.frame_world_pose(f1)[0], [1, 1, 0], atol=1e-12)
    # stored as an offset in the cube's (rotated) axes: Rz(90°)ᵀ · ((1,1,0) − (2,0,0)) = (1, 1, 0)
    r = np.asarray(numerics.quat_to_matrix(o.orientation), float)
    np.testing.assert_allclose(s.frame(f1).position, r.T @ np.array([-1.0, 1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(s.frame(f1).orientation, numerics.quat_conjugate(o.orientation), atol=1e-12)
    # turning off "rotate with parent" keeps the global orientation of the axes
    before = s.frame_world_pose(f1)[1]
    dlg.follow.setChecked(False)
    assert dlg.apply()
    after = s.frame_world_pose(f1)[1]
    assert abs(abs(float(np.dot(before, after))) - 1) < 1e-12
    # duplicate names are reported, nothing changes
    dlg.list.setCurrentRow(1)
    dlg.name.setText(s.frame(f1).name)
    dlg.name.editingFinished.emit()
    count = e.stack.count()
    assert not dlg.apply() and "Duplicate" in dlg.error.text() and e.stack.count() == count
    e.stack.undo()
    e.stack.undo()
    e.stack.undo()
    assert s.frames == []


def test_inspector_pose_and_vectors_in_named_frames(scene, owned):
    from psim.gui.properties_panel import PropertiesPanel
    from psim.gui.widgets.inputs import VectorField
    s, e = scene
    o = e.add_new("basic.sphere", position=[1, 2, 3])
    o.linear_velocity = [1.0, 0.0, 0.0]
    fid = e.add_frame("Turned", None, [1, 0, 0], numerics.quat_from_axis_angle([0, 0, 1], math.pi / 2))
    panel = PropertiesPanel(s, e, UnitRegistry())
    owned(panel)
    s.set_selection([o.id])
    panel._sync()
    panel._frame = fid
    panel._vec_frame = fid
    panel._signature = None
    panel._sync()
    fields = panel.findChildren(VectorField)
    pos = fields[0]
    np.testing.assert_allclose(pos.si(), [2, 0, 3], atol=1e-9)          # (1,2,3) seen from the turned frame
    vel = next(f for f in fields if len(f.si()) == 3 and np.allclose(f.si(), [0, -1, 0], atol=1e-9))
    vel.committed.emit([0.0, 2.0, 0.0])                                   # entered in the frame's axes …
    np.testing.assert_allclose(o.linear_velocity, [-2, 0, 0], atol=1e-12)  # … stored globally
    pos.committed.emit([0.0, 0.0, 0.0])                                   # move to the frame origin
    np.testing.assert_allclose(s.world_pose(o.id)[0], [1, 0, 0], atol=1e-12)
    # removing the frame falls back to global
    e.remove_frames([fid])
    panel._sync()
    assert panel._frame is None and panel._vec_frame is None


def test_relative_motion_panel_on_a_turntable(scene, owned):
    from psim.gui.relative_motion_panel import UNKNOWN, RelativeMotionPanel
    s, e = scene
    table = e.add_new("basic.cylinder")
    ball = e.add_new("body.point_mass", position=[1, 0, 0])
    table.angular_velocity = [0, 0, 2.0]
    ball.linear_velocity = [0, 2.0, 0]   # initial velocity: co-moving with the deck
    fid = e.add_frame("Deck", table.id)
    panel = RelativeMotionPanel(s, UnitRegistry())
    owned(panel)
    panel.show()
    s.set_selection([ball.id])
    panel.set_frame(fid)
    values = panel.compute()
    np.testing.assert_allclose(values["velocity"][0], [0, 2, 0], atol=1e-12)     # absolute velocity, deck axes
    np.testing.assert_allclose(values["velocity"][1], [0, 0, 0], atol=1e-12)     # carried along: no relative motion
    assert values["acceleration"][1] is None
    panel.refresh()
    assert panel.table.item(3, 2).text() == UNKNOWN
    assert "initial values" in panel.status.text()
    # during a run the table shows live accelerations (the ball is at rest, the deck spins steadily)
    s.simulating = True
    s.runtime = {table.id: {"acceleration": np.zeros(3), "angular_acceleration": np.zeros(3)}}
    values = panel.compute()
    # the ball is not simulated, so it is at rest; the deck observer sees it sweep backwards on a circle
    np.testing.assert_allclose(values["velocity"][0], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(values["velocity"][1], [0, -2, 0], atol=1e-12)
    np.testing.assert_allclose(values["centripetal_acceleration"][1], [-4, 0, 0], atol=1e-12)
    np.testing.assert_allclose(values["coriolis_acceleration"][1], [8, 0, 0], atol=1e-12)
    np.testing.assert_allclose(values["acceleration"][1], [-4, 0, 0], atol=1e-12)  # -ω²r toward the axis
    panel.refresh()
    assert "m/s²" in panel.table.item(3, 2).text() and "Live" in panel.status.text()
    s.simulating, s.runtime = False, {}
