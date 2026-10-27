"""Project serialization: round trip, validation and safety."""
from __future__ import annotations

import json

import numpy as np
import pytest

from psim.project import serialization as ser
from psim.properties.custom_properties import CustomProperty
from psim.scene import geometry as g
from psim.scene.builtin_types import mesh_assets
from psim.scene.scene_manager import Scene
from psim.units.unit_registry import UnitRegistry


def _populate(s, e):
    a = e.add_new("basic.cube", position=[1, 2, 3])
    a.set_euler([0.1, 0.2, 0.3])
    a.scale = [1, 2, 0.5]
    a.color = [0.1, 0.2, 0.3]
    a.add_custom(CustomProperty.create(key="k", name="K", vtype="vec3", value=[1, 2, 3], unit="m"))
    b = e.add_new("basic.sphere", parent_id=a.id, position=[0, 0, 1])
    b.overlays.update(name=True, mass=True)
    key = mesh_assets.add(g.box(1, 1, 2), "tall box")
    c = e.add_new("user.mesh", params={"asset": key})
    return a, b, c


def test_round_trip(tmp_path, scene):
    s, e = scene
    a, b, c = _populate(s, e)
    units = UnitRegistry()
    units.set_preferred_unit("length", "mm")
    data = ser.ProjectData()
    data.view.grids["YZ"].visible = True
    data.view.camera.distance = 42.0
    data.simulation.delta_time = 0.001
    data.simulation.update_rate_hz = 250.0
    path = tmp_path / "p.psim"
    ser.save_project(str(path), s, units, data)

    s2, u2 = Scene(), UnitRegistry()
    mesh_assets.clear()
    d2 = ser.load_project(str(path), s2, u2)
    assert d2.warnings == []
    assert list(s2.objects) == list(s.objects)
    for oid in s.objects:
        assert s2.get(oid).to_dict() == s.get(oid).to_dict()
    np.testing.assert_allclose(s2.world_pose(b.id)[0], s.world_pose(b.id)[0])
    assert s2.ids.next_value == s.ids.next_value
    assert u2.preferred_unit("length") == "mm"
    assert d2.view.grids["YZ"].visible and d2.view.camera.distance == 42.0
    assert d2.simulation.delta_time == 0.001 and d2.simulation.update_rate_hz == 250.0
    assert s2.get(c.id).mesh().triangle_count == 12
    assert s2.get(a.id).custom["k"].value == [1.0, 2.0, 3.0]


def test_new_ids_continue_after_load(tmp_path, scene):
    s, e = scene
    for _ in range(3):
        e.add_new("basic.cube")
    e.delete(["OBJ-000003"])
    path = tmp_path / "p.psim"
    ser.save_project(str(path), s, UnitRegistry(), ser.ProjectData())
    s2 = Scene()
    ser.load_project(str(path), s2, UnitRegistry())
    assert s2.create_object("basic.cube").id == "OBJ-000004"


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d.update(format="other"), "Not a PSim project"),
    (lambda d: d.update(format_version=99), "Unsupported project version"),
    (lambda d: d["objects"].append(dict(d["objects"][0])), "Duplicate object ID"),
    (lambda d: d["objects"][0].update(type="evil.code"), "Unknown object type"),
    (lambda d: d["objects"][0].update(position=["x", 0, 0]), "OBJ-000001"),
    (lambda d: d["objects"][0]["params"].update(size=-5), "below the minimum"),
])
def test_invalid_projects_are_rejected(scene, mutate, message):
    s, e = scene
    e.add_new("basic.cube")
    d = ser.project_to_dict(s, UnitRegistry(), ser.ProjectData())
    d = json.loads(json.dumps(d))
    mutate(d)
    target = Scene()
    with pytest.raises(ser.ProjectError, match=message):
        ser.load_project_dict(d, target, UnitRegistry())


def test_malformed_files(tmp_path):
    bad = tmp_path / "bad.psim"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ser.ProjectError, match="JSON"):
        ser.read_project_file(str(bad))
    with pytest.raises(ser.ProjectError, match="not found"):
        ser.read_project_file(str(tmp_path / "missing.psim"))


def test_missing_parent_and_asset_produce_warnings(scene):
    s, e = scene
    e.add_new("basic.cube")
    d = json.loads(json.dumps(ser.project_to_dict(s, UnitRegistry(), ser.ProjectData())))
    d["objects"][0]["parent"] = "OBJ-000999"
    d["objects"].append({"id": "OBJ-000002", "type": "user.mesh", "params": {"asset": "mesh-missing"}})
    s2 = Scene()
    data = ser.load_project_dict(d, s2, UnitRegistry())
    assert any("parent" in w for w in data.warnings)
    assert any("missing mesh asset" in w for w in data.warnings)
    assert s2.get("OBJ-000001").parent_id is None


def test_project_file_contains_no_code(tmp_path, scene):
    """Saved projects are plain data: only JSON primitives are written."""
    s, e = scene
    _populate(s, e)
    path = tmp_path / "p.psim"
    ser.save_project(str(path), s, UnitRegistry(), ser.ProjectData())
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["format"] == "psim-project" and raw["format_version"] == ser.FORMAT_VERSION

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            assert isinstance(x, (str, int, float, bool, type(None)))
    walk(raw)


def test_example_projects_load():
    import glob
    import os
    files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "examples", "*.psim"))
    assert files
    for f in files:
        s = Scene()
        data = ser.load_project(f, s, UnitRegistry())
        assert len(s) > 0 and data.warnings == []


def test_atomic_write_retries_a_briefly_locked_target(tmp_path, monkeypatch):
    """Windows refuses os.replace while another process holds the target open; the write waits and retries,
    and gives up (leaving no temporary file) if the target stays locked."""
    import json
    import os

    from psim.project import serialization as ser

    path = tmp_path / "p.json"
    path.write_text("{}", encoding="utf-8")
    real, calls = os.replace, []

    def locked(n):
        def replace(src, dst):
            calls.append(dst)
            if len(calls) <= n:
                raise PermissionError(13, "The process cannot access the file", dst)
            real(src, dst)
        return replace

    monkeypatch.setattr(ser, "REPLACE_WAIT", 0.0)
    monkeypatch.setattr(ser.os, "replace", locked(3))
    ser.write_json_atomic(str(path), {"a": 1})
    assert len(calls) == 4 and json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    calls.clear()
    monkeypatch.setattr(ser.os, "replace", locked(10 ** 6))
    with pytest.raises(PermissionError):
        ser.write_json_atomic(str(path), {"a": 2})
    assert len(calls) == ser.REPLACE_ATTEMPTS
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["p.json"]
