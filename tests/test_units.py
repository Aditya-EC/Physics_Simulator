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


# ---- Phase 5: unit expressions, derived custom units, formula unit inspection ------------------------------------

def test_unit_expressions():
    from psim.units import dimensions as D
    from psim.units.unit_registry import UnitError, UnitRegistry
    r = UnitRegistry()
    for text, factor, dim in (("kg·m²/s²", 1.0, D.ENERGY), ("m·s⁻¹", 1.0, D.VELOCITY), ("km/h", 1 / 3.6, D.VELOCITY),
                              ("N*m", 1.0, D.TORQUE), ("kg*m^2/s", 1.0, D.INERTIA / D.TIME), ("1/s", 1.0, D.FREQUENCY),
                              ("m^-2", 1.0, D.LENGTH ** -2), ("(lbf·in)/rad", 0.1129848290276167, D.TORQUE / D.ANGLE),
                              ("rev/min", 2 * math.pi / 60, D.ANGULAR_VELOCITY), ("mm³", 1e-9, D.VOLUME)):
        f, d = r.parse_expression(text)
        assert f == pytest.approx(factor) and d == dim, text
    for bad in ("m s", "m^x", "furlong", "(m", "m^", "", "*m"):
        with pytest.raises(UnitError):
            r.parse_expression(bad)


def test_si_unit_names_of_dimensions():
    from psim.units import dimensions as D
    from psim.units.unit_expressions import si_unit_text
    assert si_unit_text(D.FORCE) == "kg·m/s²"
    assert si_unit_text(D.PRESSURE) == "kg/(m·s²)"
    assert si_unit_text(D.FREQUENCY) == "1/s"
    assert si_unit_text(D.DIMENSIONLESS) == "1"
    assert si_unit_text(D.ANGULAR_VELOCITY) == "rad/s"


def test_custom_units_from_expressions():
    from psim.units import dimensions as D
    from psim.units.unit_registry import UnitError, UnitRegistry
    r = UnitRegistry()
    r.add_custom_unit("furlong", "furlong", "m", 201.168)
    r.add_custom_unit("fortnight", "fortnight", "s", 1209600)
    fpf = r.add_custom_unit("fpf", "furlong per fortnight", "furlong/fortnight", 1)
    assert fpf.dimension == D.VELOCITY and "fpf" in r.units_for("velocity")
    assert r.convert(1, "fpf", "mm/s") == pytest.approx(201.168 / 1209600 * 1000)
    r.add_custom_unit("kN·cm", "kilonewton centimetre", "kN*cm", 1)
    assert "kN·cm" in r.units_for("torque") and r.to_si(1, "kN·cm") == pytest.approx(10.0)
    with pytest.raises(UnitError, match="not a quantity"):
        r.add_custom_unit("odd", "odd", "kg·m", 1)
    # saved as SI factor + dimension: survives a round trip without the units it was defined from
    r2 = UnitRegistry()
    assert r2.load_dict(r.to_dict()) == []
    assert r2.convert(2, "fpf", "m/s") == pytest.approx(r.convert(2, "fpf", "m/s"))


def test_inspect_expression_and_requirements():
    from psim.processes.dimensional_analysis import CheckEnv
    from psim.processes.process_registry import process_registry
    from psim.processes.unit_inspection import env_for, inspect_expression, process_requirements
    from psim.units.unit_registry import UnitRegistry
    units = UnitRegistry()
    rep = inspect_expression("0.5 * self.mass * dot(self.velocity, self.velocity)", CheckEnv(), units)
    assert rep.ok and rep.result.si_unit == "kg·m²/s²" and "Energy" in rep.result.quantities
    texts = [row.text for row in rep.rows]
    assert texts[0] == "0.5 * self.mass * dot(self.velocity, self.velocity)" and "self.velocity" in texts
    assert next(r for r in rep.rows if r.text == "self.mass").si_unit == "kg"
    rep = inspect_expression("2[km/h] * 30[min]", CheckEnv(), units)
    assert rep.result.si_unit == "m" and "Length" in rep.result.quantities
    rep = inspect_expression("self.angular_velocity", CheckEnv(), units)
    assert "Angular velocity" in rep.result.quantities  # angles are dimensionless in formulas
    bad = inspect_expression("self.position + self.velocity", CheckEnv(), units)
    assert not bad.ok and "Incompatible dimensions" in bad.message and bad.error_pos == 14
    drag = process_registry().get("lib.wind_drag")
    reqs = {r.name: r for r in process_requirements(drag, units)}
    assert reqs["global.air_density"].si_unit == "kg/m³" and reqs["param.area"].quantity == "Area"
    assert reqs["force"].kind == "output" and reqs["force"].si_unit == "kg·m/s²"
    assert inspect_expression("param.cd * global.air_density * param.area", env_for(drag), units).result.si_unit == "kg/m"
    spring = process_registry().get("dyn.spring")
    assert inspect_expression("param.stiffness * param.rest_length", env_for(spring), units).result.quantities == "Force"


def test_units_dialogs(qapp):
    import shiboken6
    from psim.gui.dialogs.misc_dialogs import UnitsDialog
    from psim.gui.dialogs.unit_inspector_dialog import FormulaUnitInspector
    from psim.processes.process_registry import process_registry
    from psim.units.unit_registry import UnitRegistry
    units = UnitRegistry()
    dlg = UnitsDialog(units)
    try:
        row = next(r for r in range(dlg.table.rowCount()) if dlg.table.item(r, 0).text() == "Force")
        assert dlg.table.item(row, 2).text() == "kg·m/s²"
        dlg.sym.setText("kip")
        dlg.name.setText("kilopound-force")
        dlg.factor.setText("1000")
        dlg.base.setCurrentText("lbf")
        assert "Force" in dlg.preview.text()
        dlg._add()
        assert units.to_si(1, "kip") == pytest.approx(4448.2216152605)
        dlg.base.setCurrentText("kg·m")
        assert "no known quantity" in dlg.preview.text()
    finally:
        shiboken6.delete(dlg)
    insp = FormulaUnitInspector(process_registry(), units, "lib.wind_drag")
    try:
        assert insp.requirements.rowCount() >= 4
        insp.expression.setText("global.air_density * norm(self.velocity - global.wind)^2")
        assert "kg/(m·s²)" in insp.result.text() and "Pressure" in insp.result.text()
        assert insp.terms.rowCount() > 3
        insp.expression.setText("self.mass + 1")
        assert "Incompatible" in insp.result.text() and insp.terms.rowCount() == 0
    finally:
        shiboken6.delete(insp)
