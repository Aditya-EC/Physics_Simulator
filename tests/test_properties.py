"""Typed property definitions, custom properties and property paths."""
from __future__ import annotations

import pytest

from psim.properties.custom_properties import CustomProperty, make_key
from psim.properties.property_system import Origin, PropertyDef, PropertyError, ValueType
from psim.scene.object_base import STANDARD_PROPERTIES, SceneObject
from psim.scene.object_registry import registry


def test_coercion_and_validation():
    f = PropertyDef("x", "X", ValueType.FLOAT, minimum=0, maximum=10)
    assert f.coerce("2.5") == 2.5
    with pytest.raises(PropertyError):
        f.coerce(11)
    with pytest.raises(PropertyError):
        f.coerce(float("nan"))
    i = PropertyDef("n", "N", ValueType.INT)
    assert i.coerce(3.0) == 3
    with pytest.raises(PropertyError):
        i.coerce(3.5)
    b = PropertyDef("b", "B", ValueType.BOOL)
    assert b.coerce("yes") is True and b.coerce("0") is False
    with pytest.raises(PropertyError):
        b.coerce("maybe")
    v = PropertyDef("v", "V", ValueType.VEC3)
    assert v.coerce((1, 2, 3)) == [1.0, 2.0, 3.0]
    with pytest.raises(PropertyError):
        v.coerce([1, 2])
    c = PropertyDef("c", "C", ValueType.COLOR)
    assert c.coerce([2, -1, 0.5]) == [1.0, 0.0, 0.5]
    e = PropertyDef("e", "E", ValueType.ENUM, choices=("a", "b"))
    with pytest.raises(PropertyError):
        e.coerce("c")


def test_custom_property_creation_and_rules():
    cp = CustomProperty.create(key="drag_coefficient", name="Drag coefficient", vtype="float", value=0.47,
                               minimum=0.0, maximum=2.0, precision=3)
    assert cp.value == 0.47 and cp.definition.origin == Origin.USER
    with pytest.raises(PropertyError):
        cp.set_value(5.0)
    vec = CustomProperty.create(key="offset", name="Offset", vtype="vec3", value=[1, 2, 3], unit="mm")
    assert vec.definition.quantity == "length"
    with pytest.raises(PropertyError):
        CustomProperty.create(key="Bad Key", name="x", vtype="float")
    with pytest.raises(PropertyError):
        CustomProperty.create(key="k", name="  ", vtype="float")
    with pytest.raises(PropertyError):
        CustomProperty.create(key="k", name="K", vtype="color")
    with pytest.raises(PropertyError):
        CustomProperty.create(key="k", name="K", vtype="float", minimum=3, maximum=1)
    for t in ("float", "int", "bool", "string", "vec2", "vec3"):
        p = CustomProperty.create(key="k", name="K", vtype=t)
        assert CustomProperty.from_dict(p.to_dict()).to_dict() == p.to_dict()


def test_make_key():
    assert make_key("Drag coeff. (Cd)") == "drag_coeff_cd"
    assert make_key("2nd value")[0].isalpha()


def test_property_paths_on_objects():
    o = SceneObject.new("OBJ-000001", registry().get("basic.sphere"))
    o.set("name", "Ball")
    o.set("params.radius", 0.25)
    o.set("position", [1, 2, 3])
    o.set("overlay.velocity", True)
    o.set("tags", "a, b, a ,")
    assert o.get("name") == "Ball" and o.get("params.radius") == 0.25 and o.get("position") == [1, 2, 3]
    assert o.get("overlay.velocity") is True and o.tags == ["a", "b"]
    o.add_custom(CustomProperty.create(key="ro", name="RO", vtype="int", value=1, read_only=True))
    with pytest.raises(PropertyError):
        o.set("custom.ro", 2)
    o.set("custom.ro", 2, force=True)
    assert o.get("custom.ro") == 2
    with pytest.raises(PropertyError):
        o.set("nonexistent", 1)
    with pytest.raises(PropertyError):
        o.set("name", "   ")
    with pytest.raises(PropertyError):
        o.add_custom(CustomProperty.create(key="ro", name="dup", vtype="int"))


def test_material_presets_set_metallic_roughness():
    o = SceneObject.new("OBJ-000001", registry().get("basic.cube"))
    o.set("material", "Polished Metal")
    assert o.metallic == 1.0 and o.roughness == pytest.approx(0.15)
    o.set("roughness", 0.3)
    assert o.material == "Custom"


def test_value_origins_are_declared():
    for key in ("linear_velocity", "density", "position"):
        assert STANDARD_PROPERTIES[key].origin == Origin.USER
    assert Origin.CALCULATED.badge == "C" and Origin.SIMULATED.badge == "S"
