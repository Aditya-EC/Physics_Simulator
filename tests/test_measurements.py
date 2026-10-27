"""Distance and angle measurements: values, live updates, undo, project files, panel and overlays."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim import numerics
from psim.project import serialization as ser
from psim.rendering import physics_overlay
from psim.scene.measurements import MeasurementError, evaluate, validate_measurements
from psim.scene.scene_manager import Scene
from psim.units.unit_registry import UnitRegistry

from test_coordinates_gui import owned  # noqa: F401  (fixture)


def _three(e):
    a = e.add_new("basic.sphere", position=[1, 0, 0])
    v = e.add_new("basic.sphere", position=[0, 0, 0])
    b = e.add_new("basic.sphere", position=[1, 1, 0])
    return a, v, b


def test_distance_and_angle_values(scene):
    s, e = scene
    a, v, b = _three(e)
    d = e.add_measurement("distance", [a.id, b.id])
    e.add_measurement("angle", [a.id, v.id, b.id])
    md, ma = s.measurements
    assert (md.id, md.name, ma.name) == (d, "Distance 1", "Angle 1")
    val = evaluate(s, md)
    assert val.value == pytest.approx(1.0) and np.allclose(val.delta, [0, 1, 0])
    assert evaluate(s, ma).value == pytest.approx(math.pi / 4)
    e.set_property([b.id], "position", [-1, 1e-9, 0])                 # nearly straight: precise near 180°
    assert evaluate(s, ma).value == pytest.approx(math.pi - 1e-9, abs=1e-12)
    e.set_property([b.id], "position", [0, 0, 0])                     # coincides with the vertex
    assert math.isnan(evaluate(s, ma).value)


def test_centre_of_mass_points_and_children(scene):
    s, e = scene
    cone = e.add_new("basic.cone", position=[0, 0, 0])
    cone.set_orientation(numerics.quat_from_axis_angle([1, 0, 0], math.pi))     # upside down
    ref = e.add_new("basic.sphere", position=[0, 0, 5])
    e.add_measurement("distance", [cone.id, ref.id], at="com")
    com = cone.mass_properties()["com"]
    assert com[2] != 0
    assert evaluate(s, s.measurements[0]).value == pytest.approx(5 + com[2])     # COM flipped below the origin


def test_undo_validation_and_dangling_objects(scene):
    s, e = scene
    a, v, b = _three(e)
    with pytest.raises(MeasurementError, match="needs 3 objects"):
        e.add_measurement("angle", [a.id, b.id])
    mid = e.add_measurement("distance", [a.id, b.id])
    depth = e.stack.count()
    data = s.measurements_snapshot()
    data[0]["name"] = "Gap"
    e.set_measurements(data)
    assert s.measurements[0].name == "Gap" and e.stack.count() == depth + 1
    e.stack.undo()
    assert s.measurements[0].name == "Distance 1"
    e.remove_measurements([mid])
    assert not s.measurements
    e.stack.undo()
    e.delete([b.id])
    assert any("missing object" in p for p in s.validate())
    with pytest.raises(Exception):
        evaluate(s, s.measurements[0])
    e.stack.undo()
    assert evaluate(s, s.measurements[0]).value == pytest.approx(1.0)
    for bad in ([{"id": "X", "name": "a", "kind": "distance", "points": []}],
                [{"id": "MSR-0001", "name": "a", "kind": "volume", "points": []}],
                [{"id": "MSR-0001", "name": "a", "kind": "distance", "points": [{"object": "A"}]}],
                [{"id": "MSR-0001", "name": "a", "kind": "distance",
                  "points": [{"object": "A", "at": "top"}, {"object": "B"}]}]):
        with pytest.raises(MeasurementError):
            validate_measurements(bad)


def test_measurements_saved_in_projects(scene, tmp_path):
    s, e = scene
    a, v, b = _three(e)
    e.add_measurement("angle", [a.id, v.id, b.id], at="com", name="Opening")
    path = str(tmp_path / "m.psim")
    ser.save_project(path, s, UnitRegistry(), ser.ProjectData())
    s2 = Scene()
    data = ser.load_project(path, s2, UnitRegistry())
    assert data.warnings == [] and s2.measurements_snapshot() == s.measurements_snapshot()
    assert s2.next_measurement_number == 2
    raw = ser.read_project_file(path)
    raw["measurements"][0]["points"][0]["object"] = "OBJ-999999"
    s3 = Scene()
    data = ser.load_project_dict(raw, s3, UnitRegistry())
    assert not s3.measurements and "missing object" in data.warnings[0]


def test_overlay_geometry_and_labels(scene):
    s, e = scene
    a, v, b = _three(e)
    e.add_measurement("distance", [a.id, b.id])
    e.add_measurement("angle", [a.id, v.id, b.id])
    pts, cols = physics_overlay.measurement_lines(s)
    assert len(pts) == 2 and np.allclose(pts[0], [[1, 0, 0], [1, 1, 0]])
    arc = pts[1][4:]
    np.testing.assert_allclose(np.linalg.norm(arc, axis=1), 0.3, atol=1e-12)    # arc at 0.3 × shorter leg
    labels = dict((t.split(":")[0], t) for _a, t in physics_overlay.measurement_labels(s, UnitRegistry()))
    assert labels["Distance 1"] == "Distance 1: 1 m" and labels["Angle 1"].startswith("Angle 1: 0.7854")
    data = s.measurements_snapshot()
    data[0]["visible"] = False
    e.set_measurements(data)
    assert len(physics_overlay.measurement_lines(s)[0]) == 1


def test_measurements_panel(scene, owned):  # noqa: F811
    from psim.gui.measurements_panel import MeasurementsPanel
    s, e = scene
    a, v, b = _three(e)
    panel = MeasurementsPanel(s, e, UnitRegistry())
    owned(panel)
    s.set_selection([a.id, b.id])
    assert panel.add_distance.isEnabled() and not panel.add_angle.isEnabled()
    panel.add_distance.click()
    s.set_selection([a.id, v.id, b.id])
    panel.add_angle.click()
    assert panel.table.rowCount() == 2
    assert panel.table.item(0, 3).text() == "1 m" and panel.table.item(0, 5).text() == "1 m"
    assert panel.table.item(1, 3).text().startswith("0.785398")
    e.set_property([b.id], "position", [1, 3, 0])
    panel.refresh()
    assert panel.table.item(0, 3).text() == "3 m"
    panel.table.item(0, 1).setText("Gap")
    assert s.measurements[0].name == "Gap"
    panel.table.item(1, 1).setText("gap")              # duplicate (case-insensitive) is rejected
    assert s.measurements[1].name == "Angle 1" and "Duplicate" in panel.hint.text()
    panel.table.selectRow(1)
    panel._remove()
    assert [m.name for m in s.measurements] == ["Gap"]


def test_points_fixed_on_objects_follow_motion_rotation_and_scale(scene):
    from psim.scene.measurements import describe_point, surface_point
    s, e = scene
    a = e.add_new("basic.cube", position=[0, 0, 0.5])
    b = e.add_new("basic.cube", position=[3, 0, 0.5])
    pa = surface_point(s, a.id, [0.5, 0.0, 0.5])                   # centre of A's +x face
    pb = surface_point(s, b.id, [2.5, 0.0, 0.5])                   # centre of B's −x face
    assert pa["at"] == "point" and np.allclose(pa["local"], [0.5, 0, 0])
    mid = e.add_measurement("distance", [], points=[pa, pb])
    m = next(x for x in s.measurements if x.id == mid)
    assert evaluate(s, m).value == pytest.approx(2.0)
    assert "point 0.5, 0, 0" in describe_point(s, m.points[0])
    e.set_property([b.id], "position", [4.0, 0, 0.5])               # moves with B
    assert evaluate(s, m).value == pytest.approx(3.0)
    e.set_property([b.id], "orientation", list(numerics.quat_from_axis_angle([0, 0, 1], math.pi)))
    assert evaluate(s, m).value == pytest.approx(4.0)               # B's −x face now points away
    e.set_property([a.id], "scale", [2.0, 1.0, 1.0])                # stays on A's (scaled) face
    np.testing.assert_allclose(evaluate(s, m).points[0], [1.0, 0, 0.5], atol=1e-12)
    for _ in range(3):
        e.stack.undo()
    assert evaluate(s, m).value == pytest.approx(2.0)


def test_point_validation_and_project_files(scene, tmp_path):
    from psim.scene.measurements import surface_point
    s, e = scene
    a = e.add_new("basic.cube", position=[0, 0, 0.5])
    b = e.add_new("basic.sphere", position=[2, 0, 0.5])
    good = {"id": "MSR-0001", "name": "Gap", "kind": "distance",
            "points": [{"object": a.id, "at": "point", "local": [0.5, 0, 0]}, {"object": b.id, "at": "com"}]}
    assert validate_measurements([good])[0].points[0]["local"] == [0.5, 0, 0]
    for pts, msg in ((({"object": a.id, "at": "point"}, {"object": b.id}), "local coordinates"),
                     (({"object": a.id, "at": "point", "local": [1, "x", 0]}, {"object": b.id}), "local coordinates"),
                     (({"object": a.id, "at": "point", "local": [1, float("nan"), 0]}, {"object": b.id}), "finite"),
                     (({"object": a.id, "at": "com", "local": [1, 0, 0]}, {"object": b.id}), "only a point"),
                     (({"object": a.id, "at": "point", "local": [1, 0, 0], "x": 1}, {"object": b.id}), "unknown")):
        with pytest.raises(MeasurementError, match=msg):
            validate_measurements([dict(good, points=list(pts))])
    a.scale = [0.0, 1.0, 1.0]                                       # (the inspector never allows this)
    with pytest.raises(MeasurementError, match="zero scale"):
        surface_point(s, a.id, [0, 0, 0])
    a.scale = [1.0, 1.0, 1.0]
    e.add_measurement("distance", [], points=[surface_point(s, a.id, [0.5, 0.1, 0.6]), {"object": b.id,
                                                                                      "at": "origin"}])
    path = str(tmp_path / "p.psim")
    ser.save_project(path, s, UnitRegistry(), ser.ProjectData())
    s2 = Scene()
    data = ser.load_project(path, s2, UnitRegistry())
    assert data.warnings == [] and s2.measurements_snapshot() == s.measurements_snapshot()
    assert evaluate(s2, s2.measurements[0]).value == pytest.approx(evaluate(s, s.measurements[0]).value)


def test_snapping_to_corners_and_real_edges(scene):
    from psim.rendering import picking
    from psim.rendering.camera import OrbitCamera
    s, e = scene
    cube = e.add_new("basic.cube", position=[0, 0, 0.5])
    cam = OrbitCamera()
    cam.state.target = [0.0, 0.0, 0.5]
    cam.state.distance, cam.state.azimuth_deg, cam.state.elevation_deg = 4.0, 30.0, 35.0
    w, h = 800, 600

    def at(world, offset=(0, 0)):
        xy, _ = cam.project(np.asarray(world, float)[None, :], w, h)
        x, y = xy[0][0] + offset[0], xy[0][1] + offset[1]
        o, d = cam.screen_ray(x, y, w, h)
        hit = picking.pick(s, o, d)
        assert hit is not None and hit.object_id == cube.id
        return picking.snap_point(hit, cam, w, h, x, y)

    top = [0.0, 0.0, 1.0]
    p, kind = at(top)                                                  # middle of the top face: no snapping
    assert kind == "surface" and np.allclose(p, top, atol=1e-9)
    corner = np.array([0.5, 0.5, 1.0])
    p, kind = at(corner - [0.03, 0.03, 0], (0, 0))                     # just inside a corner
    assert kind == "vertex" and np.allclose(p, corner)
    p, kind = at([0.5 - 0.02, 0.1, 1.0])                               # next to the top-front edge
    assert kind == "edge" and p[0] == pytest.approx(0.5) and p[2] == pytest.approx(1.0)
    p, kind = at([0.1, 0.1 + 1e-3, 1.0])                               # on the diagonal seam of the top face
    assert kind == "surface"
