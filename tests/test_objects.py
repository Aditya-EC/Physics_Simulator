"""Object registry, geometry generators and mass properties."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim import numerics
from psim.properties.property_system import PropertyDef, PropertyError, ValueType
from psim.scene import geometry as g
from psim.scene.object_base import SceneObject
from psim.scene.object_registry import ObjectRegistry, ObjectTypeDef, RegistryError, registry

ALL_TYPES = [t.type_id for t in registry().types()]


def test_registry_contains_required_types():
    required = ["basic.cube", "basic.cuboid", "basic.sphere", "basic.ellipsoid", "basic.cylinder", "basic.cone",
                "basic.frustum", "basic.torus", "basic.plane", "basic.disk", "basic.capsule", "basic.pyramid",
                "basic.prism", "basic.hemisphere", "user.mesh", "body.point_mass", "body.rigid_box",
                "frame.reference", "frame.marker", "measure.line_segment", "measure.vector_arrow", "mech.spring",
                "mech.rope", "group.assembly"]
    for t in required:
        assert registry().has(t), t


def test_example_plugin_registered():
    assert registry().has("plugin.wedge")


@pytest.mark.parametrize("type_id", ALL_TYPES)
def test_every_type_builds_valid_geometry(type_id):
    td = registry().get(type_id)
    mesh = td.build_mesh(td.default_params())
    assert mesh.triangle_count > 0
    assert mesh.vertices.dtype == np.float32 and mesh.indices.dtype == np.uint32
    assert int(mesh.indices.max()) < len(mesh.vertices)
    n = np.linalg.norm(mesh.normals, axis=1)
    assert np.allclose(n, 1.0, atol=1e-3), "normals must be unit length"
    lo, hi = mesh.bounds()
    assert np.all(hi - lo >= 0) and np.all(np.isfinite(lo))
    if mesh.solid:
        vol = numerics.mass_properties_solid(mesh.vertices, mesh.indices, 1.0)["measure"]
        assert vol > 0, "closed meshes must be outward oriented (positive volume)"


@pytest.mark.parametrize("mesh,expected", [
    (g.box(1, 2, 3), 6.0),
    (g.sphere(1.0, 192, 96), 4 / 3 * math.pi),
    (g.cylinder(1.0, 2.0, 256), 2 * math.pi),
    (g.cone(1.0, 3.0, 256), math.pi),
    (g.frustum(2.0, 1.0, 3.0, 256), math.pi * 7),
    (g.torus(2.0, 0.5, 256, 64), 2 * math.pi ** 2 * 2 * 0.25),
    (g.capsule(1.0, 2.0, 256, 64), 2 * math.pi + 4 / 3 * math.pi),
    (g.hemisphere(1.0, 256, 64), 2 / 3 * math.pi),
    (g.pyramid(2, 3, 4), 8.0),
    (g.prism(6, 1.0, 2.0), 3 * math.sqrt(3)),
    (g.ellipsoid(1, 2, 3, 192, 96), 8 * math.pi),
])
def test_volumes_match_analytic(mesh, expected):
    vol = numerics.mass_properties_solid(mesh.vertices, mesh.indices, 1.0)["measure"]
    assert vol == pytest.approx(expected, rel=2e-3)


def test_solid_inertia_matches_analytic():
    # cuboid a x b x c, mass m: Ixx = m (b^2 + c^2) / 12
    m = g.box(1, 2, 3)
    mp = numerics.mass_properties_solid(m.vertices, m.indices, 2.0)
    mass = 12.0
    assert mp["inertia"][0, 0] == pytest.approx(mass * (4 + 9) / 12)
    assert mp["inertia"][1, 1] == pytest.approx(mass * (1 + 9) / 12)
    assert mp["inertia"][2, 2] == pytest.approx(mass * (1 + 4) / 12)
    # solid sphere: 2/5 m r^2
    s = g.sphere(0.5, 192, 96)
    mp = numerics.mass_properties_solid(s.vertices, s.indices, 1000.0)
    assert mp["inertia"][2, 2] == pytest.approx(0.4 * mp["measure"] * 1000 * 0.25, rel=3e-3)
    # solid cylinder about its axis: 1/2 m r^2
    c = g.cylinder(0.3, 1.0, 256)
    mp = numerics.mass_properties_solid(c.vertices, c.indices, 1.0)
    assert mp["inertia"][2, 2] == pytest.approx(0.5 * mp["measure"] * 0.09, rel=1e-3)


def test_shell_plane_inertia():
    p = g.plane(2.0, 4.0, 3)
    mp = numerics.mass_properties_shell(p.vertices, p.indices, 1.0)
    m = 8.0
    assert mp["measure"] == pytest.approx(8.0)
    assert mp["inertia"][0, 0] == pytest.approx(m * 16 / 12)  # about X: m b^2 / 12
    assert mp["inertia"][2, 2] == pytest.approx(m * (4 + 16) / 12)


def test_scene_object_mass_modes_and_scale():
    cube = SceneObject.new("OBJ-000001", registry().get("basic.cube"), params={"size": 1.0})
    cube.density = 7850.0
    assert cube.mass() == pytest.approx(7850.0)
    cube.scale = [2.0, 1.0, 1.0]
    assert cube.mass() == pytest.approx(15700.0)
    assert cube.mass_properties()["inertia"][0, 0] == pytest.approx(15700 * 2 / 12)
    cube.set("mass_mode", "mass")
    cube.set("mass_input", 3.0)
    mp = cube.mass_properties()
    assert mp["mass"] == pytest.approx(3.0) and mp["density"] == pytest.approx(1.5)


def test_point_mass_and_massless_types():
    pm = SceneObject.new("OBJ-000001", registry().get("body.point_mass"))
    pm.set("mass_input", 2.5)
    pm.set("mass_mode", "density")  # ignored for point masses
    assert pm.mass_mode == "mass" and pm.mass() == pytest.approx(2.5)
    assert np.allclose(pm.mass_properties()["inertia"], 0)
    frame = SceneObject.new("OBJ-000002", registry().get("frame.reference"))
    assert frame.mass() is None and not frame.typedef.physical


def test_geometry_validation_rules():
    td = registry().get("basic.torus")
    with pytest.raises(PropertyError):
        td.coerce_params({"major_radius": 0.2, "minor_radius": 0.3})
    with pytest.raises(PropertyError):
        td.coerce_params({"unknown": 1})
    with pytest.raises(PropertyError):
        registry().get("basic.sphere").coerce_params({"radius": -1})


def test_registry_rejects_bad_definitions():
    reg = ObjectRegistry()
    td = ObjectTypeDef("test.box", "Box", "Basic Solids", "", [PropertyDef("s", "S", ValueType.FLOAT, 1.0)],
                       lambda p: g.box(p["s"], p["s"], p["s"]))
    reg.register(td)
    with pytest.raises(RegistryError):
        reg.register(td)
    with pytest.raises(RegistryError):
        reg.register(ObjectTypeDef("nodot", "X", "Basic Solids", "", [], lambda p: g.box(1, 1, 1)))
    with pytest.raises(RegistryError):
        reg.register(ObjectTypeDef("test.cat", "X", "Not a category", "", [], lambda p: g.box(1, 1, 1)))


def test_obj_import_roundtrip(tmp_path):
    text = g.mesh_to_obj(g.box(1, 2, 3))
    m = g.load_obj(text)
    assert m.solid
    assert numerics.mass_properties_solid(m.vertices, m.indices, 1.0)["measure"] == pytest.approx(6.0, rel=1e-5)
    with pytest.raises(g.MeshImportError):
        g.load_obj("v 0 0 0\nf 1 2 3\n")
    with pytest.raises(g.MeshImportError):
        g.load_obj("# nothing")


def test_feature_edges_of_cube():
    assert len(g.box(1, 1, 1).feature_edges()) == 24  # 12 edges x 2 endpoints
    assert len(g.sphere(1.0, 32, 16).feature_edges()) == 0
