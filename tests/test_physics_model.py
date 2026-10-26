"""Collision geometry and joint definitions (data model, undo, project files)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from psim import numerics
from psim.physics.colliders import collider_for
from psim.physics.joints import JointError, validate_joints


def test_colliders_are_explicit_and_separate_from_the_mesh(scene):
    s, e = scene
    ball = e.add_new("basic.sphere", params={"radius": 0.25})
    assert collider_for(s.get(ball.id)) is None                  # mass alone does not make a collider
    e.set_property([ball.id], "physics.collider", "auto")
    c = collider_for(s.get(ball.id))
    assert c.kind == "sphere" and c.radius == pytest.approx(0.25) and not c.approximate
    box = e.add_new("body.rigid_box", params={"length": 1.0, "width": 0.5, "height": 0.2})
    e.set_property([box.id], "scale", [2.0, 1.0, 1.0])
    c = collider_for(s.get(box.id))                               # rigid-body types default to "auto"
    assert c.kind == "box" and np.allclose(c.half_extents, [1.0, 0.25, 0.1])
    cyl = e.add_new("basic.cylinder", params={"radius": 0.3, "height": 1.0})
    e.set_property([cyl.id], "physics.collider", "auto")
    c = collider_for(s.get(cyl.id))
    assert c.kind == "convex" and not c.approximate and "convex hull" in c.describe()   # the faceted mesh is exact
    e.set_property([cyl.id], "physics.collider", "box")
    c = collider_for(s.get(cyl.id))
    assert c.kind == "box" and np.allclose(c.half_extents, [0.3, 0.3, 0.5], atol=1e-6)
    e.set_property([cyl.id], "physics.collider", "capsule")
    e.set_property([cyl.id], "physics.radius", 0.3)
    e.set_property([cyl.id], "physics.length", 0.4)
    c = collider_for(s.get(cyl.id))
    assert c.kind == "capsule" and c.radius == 0.3 and c.half_length == pytest.approx(0.2)
    ground = e.add_new("basic.plane")
    e.set_property([ground.id], "physics.collider", "auto")
    assert collider_for(s.get(ground.id)).kind == "plane"
    # settings are undoable and saved
    e.set_property([ball.id], "physics.restitution", 0.9)
    assert s.get(ball.id).to_dict()["physics"] == {"collider": "auto", "restitution": 0.9}
    e.stack.undo()
    assert s.get(ball.id).physics_setting("restitution") == pytest.approx(0.3)
    with pytest.raises(Exception):
        e.set_property([ball.id], "physics.friction", -1)


def test_joint_creation_keeps_local_anchors(scene):
    s, e = scene
    a = e.add_new("body.rigid_box", position=[0, 0, 1])
    s.get(a.id).set_orientation(numerics.quat_from_axis_angle([0, 0, 1], math.pi / 2))
    b = e.add_new("body.rigid_box", position=[2, 0, 1])
    jid = e.add_joint("revolute", a.id, b.id, [1, 0, 1], [0, 1, 0], limit_enabled=True, lower=-0.5, upper=0.5)
    j = s.joint(jid)
    assert j.name == "Revolute 1" and j.type == "revolute"
    np.testing.assert_allclose(j.anchor_a, [0, -1, 0], atol=1e-12)   # (1,0,0) offset seen in A's turned axes
    np.testing.assert_allclose(j.anchor_b, [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(j.axis_a, [1, 0, 0], atol=1e-12)
    wid = e.add_joint("distance", b.id, None, [2, 0, 1], anchor_world_b=[2, 0, 3], rope=True)
    w = s.joint(wid)
    assert w.body_b is None and w.length == pytest.approx(2.0) and w.rope
    np.testing.assert_allclose(w.anchor_b, [2, 0, 3])
    e.update_joint(jid, motor_enabled=True, motor_speed=1.0, motor_max=5.0)
    assert s.joint(jid).motor_enabled
    e.stack.undo()
    assert not s.joint(jid).motor_enabled
    e.remove_joints([jid])
    assert s.joint(jid) is None
    e.stack.undo()
    assert s.joint(jid) is not None
    assert e.add_joint("fixed", a.id, b.id, [1, 0, 1]) == "JNT-0003"


def test_joint_validation():
    good = {"id": "JNT-0001", "name": "Hinge", "type": "revolute", "body_a": "OBJ-000001", "body_b": None}
    j = validate_joints([good])[0]
    assert j.axis_a == [0.0, 0.0, 1.0]
    for bad, msg in (({**good, "type": "weld"}, "unknown joint type"), ({**good, "body_b": "OBJ-000001"},
                                                                         "two different"),
                     ({**good, "axis_a": [0, 0, 0]}, "non-zero"), ({**good, "limit_enabled": True, "lower": 1,
                                                                    "upper": 0}, "lower limit"),
                     ({**good, "motor_max": -1}, "negative"), ({**good, "spring": 3}, "unknown field"),
                     ({**good, "id": "J1"}, "Invalid joint id")):
        with pytest.raises(JointError, match=msg):
            validate_joints([bad])
    with pytest.raises(JointError, match="Duplicate joint name"):
        validate_joints([good, {**good, "id": "JNT-0002", "name": "hinge"}])


def test_joints_and_physics_in_project_files(scene):
    from psim.project import serialization as ser
    from psim.scene.scene_manager import Scene
    from psim.units.unit_registry import UnitRegistry
    s, e = scene
    a = e.add_new("body.rigid_box")
    e.set_property([a.id], "physics.friction", 0.8)
    e.add_joint("ball", a.id, None, [0, 0, 1])
    d = json.loads(json.dumps(ser.project_to_dict(s, UnitRegistry(), ser.ProjectData())))
    s2 = Scene()
    data = ser.load_project_dict(d, s2, UnitRegistry())
    assert s2.joints_snapshot() == s.joints_snapshot() and data.warnings == []
    assert s2.get(a.id).physics_setting("friction") == 0.8 and s2.allocate_joint_id() == "JNT-0002"
    d["objects"] = [o for o in d["objects"] if o["id"] != a.id]
    assert any("connects the missing object" in w for w in ser.load_project_dict(d, Scene(), UnitRegistry()).warnings)
