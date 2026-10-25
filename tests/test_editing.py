"""Scene editing: unique persistent IDs, undo/redo, hierarchy, clipboard, align/distribute."""
from __future__ import annotations

import numpy as np
import pytest

from psim.core.ids import DuplicateIdError, IdAllocator
from psim.properties.property_system import PropertyError
from psim.scene.analysis import combined_mass_properties


def test_id_allocator():
    a = IdAllocator()
    assert [a.allocate() for _ in range(3)] == ["OBJ-000001", "OBJ-000002", "OBJ-000003"]
    a.release("OBJ-000002")
    assert a.allocate() == "OBJ-000004"  # never reused
    a.reserve("OBJ-000010")
    assert a.allocate() == "OBJ-000011"
    with pytest.raises(DuplicateIdError):
        a.reserve("OBJ-000010")
    with pytest.raises(ValueError):
        a.reserve("XYZ-1")


def test_ids_are_unique_and_stable(scene):
    s, e = scene
    objs = [e.add_new("basic.cube") for _ in range(5)]
    ids = [o.id for o in objs]
    assert len(set(ids)) == 5
    e.set_property([ids[0]], "name", "Renamed")
    e.set_property([ids[0]], "position", [3, 2, 1])
    assert s.get(ids[0]).id == ids[0] and s.get(ids[0]).name == "Renamed"
    e.delete([ids[1]])
    new = e.add_new("basic.sphere")
    assert new.id not in ids  # deleted IDs are not reused


def test_undo_redo_restores_exact_state(scene):
    s, e = scene
    a = e.add_new("basic.cube", position=[1, 0, 0])
    e.set_property([a.id], "params.size", 2.0)
    e.set_property([a.id], "color", [1, 0, 0], merge=False)
    assert s.get(a.id).params["size"] == 2.0
    e.stack.undo()
    e.stack.undo()
    assert s.get(a.id).params["size"] == 1.0
    e.stack.redo()
    e.stack.redo()
    assert s.get(a.id).color == [1.0, 0.0, 0.0]
    e.delete([a.id])
    assert a.id not in s
    e.stack.undo()
    assert a.id in s and s.get(a.id).params["size"] == 2.0


def test_invalid_edit_is_rejected_and_rolled_back(scene):
    s, e = scene
    a = e.add_new("basic.torus")
    n = e.stack.count()
    with pytest.raises(PropertyError):
        e.set_property([a.id], "params.minor_radius", 5.0)
    assert s.get(a.id).params["minor_radius"] == pytest.approx(0.15)
    assert e.stack.count() == n


def test_consecutive_edits_merge(scene):
    s, e = scene
    a = e.add_new("basic.cube")
    n = e.stack.count()
    for v in (1.0, 2.0, 3.0):
        e.set_property([a.id], "position", [v, 0, 0])
    assert e.stack.count() == n + 1
    e.stack.undo()
    assert s.get(a.id).position == [0.0, 0.0, 0.0]


def test_group_ungroup_undo(scene):
    s, e = scene
    a = e.add_new("basic.cube", position=[2, 0, 0])
    b = e.add_new("basic.sphere", position=[-2, 0, 1])
    g = e.group([a.id, b.id])
    assert s.get(a.id).parent_id == g.id and s.get(b.id).parent_id == g.id
    np.testing.assert_allclose(s.world_pose(a.id)[0], [2, 0, 0], atol=1e-12)
    e.stack.undo()
    assert g.id not in s and s.get(a.id).parent_id is None
    e.stack.redo()
    assert s.get(a.id).parent_id == g.id
    freed = e.ungroup([g.id])
    assert set(freed) == {a.id, b.id} and g.id not in s
    np.testing.assert_allclose(s.world_pose(b.id)[0], [-2, 0, 1], atol=1e-12)


def test_delete_removes_descendants_and_undo_restores_hierarchy(scene):
    s, e = scene
    p = e.add_new("group.assembly")
    c = e.add_new("basic.cube", parent_id=p.id)
    gc = e.add_new("basic.cone", parent_id=c.id)
    assert e.delete([p.id]) == 3
    assert len(s) == 0
    e.stack.undo()
    assert s.get(gc.id).parent_id == c.id and s.get(c.id).parent_id == p.id


def test_duplicate_and_paste_give_new_ids(scene):
    s, e = scene
    p = e.add_new("group.assembly")
    c = e.add_new("basic.cube", parent_id=p.id, position=[1, 0, 0])
    copies = e.duplicate([p.id])
    assert len(copies) == 2
    new_parent = next(o for o in copies if o.type_id == "group.assembly")
    new_child = next(o for o in copies if o.type_id == "basic.cube")
    assert new_parent.id not in (p.id, c.id) and new_child.parent_id == new_parent.id
    states = e.serialize([c.id])
    pasted = e.instantiate(states, offset=(0, 0, 1))
    assert pasted[0].id != c.id and pasted[0].parent_id == p.id


def test_align_and_distribute(scene):
    s, e = scene
    objs = [e.add_new("basic.cube", position=[x, y, 0]) for x, y in ((0, 0), (5, 1), (1, 3))]
    ids = [o.id for o in objs]
    e.align(ids, 1, "min")
    lows = [s.world_bounds([i])[0][1] for i in ids]
    assert max(lows) - min(lows) == pytest.approx(0, abs=1e-12)
    e.distribute(ids, 0)
    xs = sorted(s.world_pose(i)[0][0] for i in ids)
    assert xs[1] - xs[0] == pytest.approx(xs[2] - xs[1])


def test_custom_property_edit_cycle(scene):
    s, e = scene
    a = e.add_new("basic.cube")
    e.add_custom_property([a.id], {"key": "cd", "name": "Drag", "type": "float", "value": 0.5})
    e.set_property([a.id], "custom.cd", 0.8)
    e.replace_custom_property(a.id, "cd", {"key": "drag", "name": "Drag coefficient", "type": "float", "value": 0.9})
    assert list(s.get(a.id).custom) == ["drag"]
    e.stack.undo()
    assert list(s.get(a.id).custom) == ["cd"] and s.get(a.id).custom["cd"].value == 0.8


def test_combined_mass_properties_parallel_axis(scene):
    s, e = scene
    # two 1 kg point masses at x = ±1: total 2 kg, COM at origin, Iyy = Izz = 2 kg·m², Ixx = 0
    for x in (-1.0, 1.0):
        o = e.add_new("body.point_mass", position=[x, 0, 0])
        o.mass_input = 1.0
    r = combined_mass_properties(s, list(s.objects))
    assert r["mass"] == pytest.approx(2.0)
    np.testing.assert_allclose(r["com"], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(np.diag(r["inertia"]), [0, 2, 2], atol=1e-12)
