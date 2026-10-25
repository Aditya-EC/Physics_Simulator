"""Camera, picking, placement and gizmo mathematics (no OpenGL required)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim import numerics
from psim.rendering import gizmo as gz
from psim.rendering import picking
from psim.rendering.camera import OrbitCamera
from psim.rendering.view_settings import CameraState, SnapSettings, ViewSettings


def test_screen_ray_and_projection_are_consistent():
    cam = OrbitCamera(CameraState(target=[1, 2, 0], distance=8, azimuth_deg=-40, elevation_deg=30))
    w, h = 800, 600
    pts = np.array([[1, 2, 0], [0, 0, 0], [2, -1, 1.5]])
    xy, ok = cam.project(pts, w, h)
    assert ok.all()
    for p, (x, y) in zip(pts, xy):
        o, d = cam.screen_ray(x, y, w, h)
        closest = o + d * float((p - o) @ d)
        assert np.linalg.norm(closest - p) < 1e-6
    assert xy[0] == pytest.approx([w / 2, h / 2])  # the target projects to the centre


def test_orthographic_ray():
    cam = OrbitCamera(CameraState(orthographic=True, azimuth_deg=-90, elevation_deg=0, distance=5))
    o1, d1 = cam.screen_ray(100, 300, 800, 600)
    o2, d2 = cam.screen_ray(700, 300, 800, 600)
    np.testing.assert_allclose(d1, d2)  # parallel rays
    np.testing.assert_allclose(d1, [0, 1, 0], atol=1e-9)  # front view looks along +Y


def test_standard_views_and_fit():
    cam = OrbitCamera()
    cam.set_view("top")
    np.testing.assert_allclose(cam.forward(), [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(cam.up(), [0, 1, 0], atol=1e-12)
    cam.fit_bounds(np.array([-1, -1, -1]), np.array([1, 1, 1]), aspect=1.0)
    xy, ok = cam.project(np.array([[1, 1, 1], [-1, -1, -1]]), 500, 500)
    assert ok.all() and np.all((xy > 0) & (xy < 500))


def test_pick_returns_closest_object(scene):
    s, e = scene
    near = e.add_new("basic.cube", position=[0, 0, 0])
    e.add_new("basic.cube", position=[0, 5, 0])
    hit = picking.pick(s, [0, -10, 0], [0, 1, 0])
    assert hit.object_id == near.id
    assert hit.point[1] == pytest.approx(-0.5)
    np.testing.assert_allclose(hit.normal, [0, -1, 0], atol=1e-6)
    s.get(near.id).visible = False
    assert picking.pick(s, [0, -10, 0], [0, 1, 0]).object_id != near.id
    assert picking.pick(s, [0, -10, 0], [1, 0, 0]) is None


def test_placement_rests_on_work_plane_and_surfaces(scene):
    s, e = scene
    lo, hi = np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])
    pos, sid = picking.placement_position(s, [0.3, 0.1, 10], [0, 0, -1], lo, hi, "XY", 0.0, True, 0.25, True)
    assert sid is None
    np.testing.assert_allclose(pos, [0.25, 0.0, 0.5])  # snapped in X/Y, resting on z = 0
    base = e.add_new("basic.cuboid", params={"length": 4, "width": 4, "height": 1}, position=[0, 0, 0.5])
    pos, sid = picking.placement_position(s, [1.1, 0.0, 10], [0, 0, -1], lo, hi, "XY", 0.0, False, 0.25, True)
    assert sid == base.id
    np.testing.assert_allclose(pos, [1.1, 0.0, 1.5], atol=1e-6)  # stacked on top of the cuboid
    pos, _ = picking.placement_position(s, [10, 0.2, 0.3], [-1, 0, 0], lo, hi, "YZ", 0.0, False, 0.25, False)
    np.testing.assert_allclose(pos, [0.5, 0.2, 0.3], atol=1e-9)


def test_closest_param_and_ray_plane():
    s = gz.closest_param_on_axis([0, 0, 0], [1, 0, 0], [3, -5, 1], [0, 1, 0])
    assert s == pytest.approx(3.0)
    assert gz.closest_param_on_axis([0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]) is None
    p = gz.ray_plane([0, 0, 5], [0, 0, -1], [0, 0, 1], [0, 0, 1])
    np.testing.assert_allclose(p, [0, 0, 1])


def _drag(handle_name, start_ray, end_ray, snap=None, axes=np.eye(3), global_frame=True):
    handles = {h.name: h for hs in gz.build_handles().values() for h in hs}
    start = {"A": (np.zeros(3), np.array([1.0, 0, 0, 0]), np.ones(3))}
    st = gz.begin_drag(handles[handle_name], np.zeros(3), axes, np.array([0, 1.0, 0]), start, "A", global_frame,
                       *start_ray)
    return gz.update_drag(st, *end_ray, snap or SnapSettings(grid_snap=False))


def test_translate_drag_along_axis_with_snapping():
    r0 = (np.array([0.5, -5, 0]), np.array([0, 1.0, 0]))
    r1 = (np.array([1.87, -5, 0.3]), np.array([0, 1.0, 0]))
    res = _drag("t_x", r0, r1)
    np.testing.assert_allclose(res["A"][0], [1.37, 0, 0], atol=1e-9)
    res = _drag("t_x", r0, r1, SnapSettings(grid_snap=True, translate_increment=0.25))
    np.testing.assert_allclose(res["A"][0], [1.25, 0, 0], atol=1e-9)


def test_rotate_drag_quarter_turn():
    # drag on the Z ring from +X to +Y (viewed from above): +90° about Z
    r0 = (np.array([1.0, 0, 5]), np.array([0, 0, -1.0]))
    r1 = (np.array([0.0, 1.0, 5]), np.array([0, 0, -1.0]))
    res = _drag("r_z", r0, r1)
    np.testing.assert_allclose(numerics.quat_to_euler(res["A"][1]), [0, 0, math.pi / 2], atol=1e-9)
    snap = SnapSettings(rotate_snap=True, rotate_increment_deg=40)
    res = _drag("r_z", r0, r1, snap)
    np.testing.assert_allclose(numerics.quat_to_euler(res["A"][1])[2], math.radians(80), atol=1e-9)


def test_scale_drag():
    r0 = (np.array([0.8, -5, 0]), np.array([0, 1.0, 0]))
    r1 = (np.array([1.6, -5, 0]), np.array([0, 1.0, 0]))
    res = _drag("s_x", r0, r1)
    np.testing.assert_allclose(res["A"][2], [2.0, 1.0, 1.0])


def test_gizmo_handle_picking():
    handles = gz.build_handles()["translate"]
    model = gz.gizmo_matrix(np.zeros(3), np.eye(3), 1.0)
    assert gz.pick_handle(handles, model, [0.7, -5, 0], [0, 1, 0]) == "t_x"
    assert gz.pick_handle(handles, model, [0, -5, 0.7], [0, 1, 0]) == "t_z"
    assert gz.pick_handle(handles, model, [5, 5, 5], [1, 0, 0]) is None


def test_view_settings_round_trip():
    v = ViewSettings()
    v.grids["XZ"].visible = True
    v.grids["XY"].major_spacing = 0.5
    v.snap.work_plane = "YZ"
    v.bookmarks["a"] = CameraState(distance=3)
    v2 = ViewSettings.from_dict(v.to_dict())
    assert v2.to_dict() == v.to_dict()
    bad = v.to_dict()
    bad["grids"]["XY"]["major_spacing"] = "oops"
    assert ViewSettings.from_dict(bad).grids["XY"].major_spacing == 1.0
