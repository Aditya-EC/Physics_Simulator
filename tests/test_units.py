"""Unit registry and dimensional consistency."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim.units import dimensions as D
from psim.units.unit_registry import UnitError, UnitRegistry


@pytest.fixture
def u():
    return UnitRegistry()


@pytest.mark.parametrize("value,src,dst,expected", [
    (1, "ft", "m", 0.3048), (1, "in", "mm", 25.4), (1, "lb", "kg", 0.45359237), (90, "deg", "rad", math.pi / 2),
    (60, "rpm", "rad/s", 2 * math.pi), (1, "kWh", "J", 3.6e6), (1, "psi", "Pa", 6894.757293168361),
    (1, "g/cm³", "kg/m³", 1000.0), (1, "lbf", "N", 4.4482216152605), (36, "km/h", "m/s", 10.0),
    (1, "g₀", "m/s²", 9.80665), (1, "hp", "W", 745.69987158227022), (1, "N·s", "kg·m/s", 1.0),
])
def test_conversions(u, value, src, dst, expected):
    assert u.convert(value, src, dst) == pytest.approx(expected)
    assert u.convert(u.convert(value, src, dst), dst, src) == pytest.approx(value)


def test_vector_conversion(u):
    np.testing.assert_allclose(u.convert([1, 2, 3], "m", "mm"), [1000, 2000, 3000])


def test_incompatible_units_rejected(u):
    with pytest.raises(UnitError):
        u.convert(1, "m", "kg")
    with pytest.raises(UnitError):
        u.convert(1, "rad/s", "Hz")  # angle is a tracked pseudo-dimension
    with pytest.raises(UnitError):
        u.unit("parsec")


def test_dimensions_algebra():
    assert D.FORCE == D.MASS * D.LENGTH / D.TIME ** 2
    assert str(D.VELOCITY) == "L·T^-1"
    assert D.DIMENSIONLESS.is_dimensionless


def test_torque_and_energy_lists_are_separate(u):
    assert "N·m" in u.units_for("torque") and "N·m" not in u.units_for("energy")
    assert "J" in u.units_for("energy") and "J" not in u.units_for("torque")


def test_custom_units_and_preferences(u):
    u.add_custom_unit("furlong", "furlong", "m", 201.168)
    assert u.convert(1, "furlong", "ft") == pytest.approx(660.0)
    assert "furlong" in u.units_for("length")
    with pytest.raises(UnitError):
        u.add_custom_unit("furlong", "dup", "m", 1)
    with pytest.raises(UnitError):
        u.add_custom_unit("bad", "x", "m", -1)
    with pytest.raises(UnitError):
        u.remove_custom_unit("m")
    u.set_preferred_unit("length", "mm")
    assert u.preferred_unit("length") == "mm"
    with pytest.raises(UnitError):
        u.set_preferred_unit("length", "kg")
    u.set_display_unit("position", "length", "cm")
    assert u.display_unit("position", "length") == "cm"
    assert u.display_unit("other", "length") == "mm"
    data = u.to_dict()
    v = UnitRegistry()
    assert v.load_dict(data) == []
    assert v.convert(2, "furlong", "m") == pytest.approx(402.336)
    assert v.preferred_unit("length") == "mm" and v.display_unit("position", "length") == "cm"


def test_display_units_do_not_change_values(u):
    """Changing the display unit is presentation only: SI round trip is exact."""
    si = 1.2345
    for sym in u.units_for("length"):
        assert u.to_si(u.from_si(si, sym), sym) == pytest.approx(si, rel=1e-12)
