"""Coordinate transformations: rotations, hierarchy (local/global) and relative frames."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim import numerics


def test_euler_convention_extrinsic_xyz():
    # R = Rz * Ry * Rx: rotating (1,0,0) by 90° about Z gives (0,1,0)
    q = numerics.quat_from_euler([0, 0, math.pi / 2])
    np.testing.assert_allclose(numerics.quat_rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-12)
    rx, ry, rz = 0.3, -0.4, 1.1
    r = numerics.quat_to_matrix(numerics.quat_from_euler([rx, ry, rz]))
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    np.testing.assert_allclose(r, Rz @ Ry @ Rx, atol=1e-12)


def test_euler_roundtrip_including_gimbal_lock():
    for e in ([0.1, 0.2, 0.3], [2.0, -1.0, -2.5], [0.4, math.pi / 2, 0.2], [0, -math.pi / 2, 1.0]):
        q = numerics.quat_from_euler(e)
        q2 = numerics.quat_from_euler(numerics.quat_to_euler(q))
        assert abs(float(np.dot(q, q2))) == pytest.approx(1.0, abs=1e-9)


def test_world_and_local_poses(scene):
    s, e = scene
    parent = e.add_new("basic.cube", position=[1, 0, 0])
    parent.set_euler([0, 0, math.pi / 2])
    child = e.add_new("basic.sphere", position=[2, 0, 0], parent_id=parent.id)
    p, q = s.world_pose(child.id)
    np.testing.assert_allclose(p, [1, 2, 0], atol=1e-12)  # (2,0,0) rotated 90° about Z then offset
    # set a world pose and read back the local pose
    s.set_world_pose(child.id, [1, 0, 5], numerics.quat_from_euler([0, 0, 0]))
    np.testing.assert_allclose(child.position, [0, 0, 5], atol=1e-12)
    np.testing.assert_allclose(child.euler(), [0, 0, -math.pi / 2], atol=1e-12)
    # moving the parent moves the child
    parent.position = [3, 0, 0]
    np.testing.assert_allclose(s.world_pose(child.id)[0], [3, 0, 5], atol=1e-12)


def test_reparent_keeps_world_pose(scene):
    s, e = scene
    a = e.add_new("basic.cube", position=[1, 2, 3])
    b = e.add_new("basic.cube", position=[-4, 0, 1])
    b.set_euler([0.3, 0.2, 0.1])
    before = s.world_pose(a.id)
    e.reparent([a.id], b.id)
    after = s.world_pose(a.id)
    np.testing.assert_allclose(before[0], after[0], atol=1e-12)
    assert abs(float(np.dot(before[1], after[1]))) == pytest.approx(1.0)
    with pytest.raises(Exception):
        e.reparent([b.id], a.id)  # cycle


def test_relative_frame_representation(scene):
    """A pose expressed in another object's frame, and converted back, is identical."""
    s, e = scene
    ref = e.add_new("frame.reference", position=[2, -1, 0.5])
    ref.set_euler([0.2, -0.7, 1.3])
    obj = e.add_new("basic.cube", position=[0.3, 4.0, -2.0])
    obj.set_euler([1.0, 0.1, -0.4])
    rp, rq = s.pose_in_frame(obj.id, ref.id)
    # independent check: p_rel = R_ref^T (p - p_ref)
    r_ref = numerics.quat_to_matrix(ref.orientation)
    np.testing.assert_allclose(rp, r_ref.T @ (np.array(obj.position) - np.array(ref.position)), atol=1e-12)
    np.testing.assert_allclose(numerics.quat_to_matrix(rq), r_ref.T @ numerics.quat_to_matrix(obj.orientation),
                               atol=1e-12)
    # editing in the relative frame
    s.set_pose_in_frame(obj.id, ref.id, [1, 0, 0], None)
    np.testing.assert_allclose(s.world_pose(obj.id)[0], np.array(ref.position) + r_ref @ [1, 0, 0], atol=1e-12)


def test_world_matrix_includes_only_own_scale(scene):
    s, e = scene
    parent = e.add_new("basic.cube")
    parent.scale = [2, 2, 2]
    child = e.add_new("basic.cube", position=[1, 0, 0], parent_id=parent.id)
    np.testing.assert_allclose(s.world_matrix(child.id)[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(s.world_pose(child.id)[0], [1, 0, 0])
