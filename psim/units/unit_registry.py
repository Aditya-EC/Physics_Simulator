"""Unit registry: units, quantity kinds, conversions and project display preferences.

All values are stored internally in canonical SI units (radians for angles).
Display units only affect presentation: converting for display never changes
the stored physical quantity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import dimensions as D
from .dimensions import Dimension
from .unit_expressions import UnitExpressionError, parse_unit_expression, si_unit_text


class UnitError(ValueError):
    pass


@dataclass(frozen=True)
class Unit:
    symbol: str
    name: str
    dimension: Dimension
    factor: float  # SI value = value * factor + offset
    offset: float = 0.0
    custom: bool = False

    def to_si(self, value):
        return np.asarray(value, dtype=float) * self.factor + self.offset if isinstance(value, (list, tuple, np.ndarray)) \
            else value * self.factor + self.offset

    def from_si(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return (np.asarray(value, dtype=float) - self.offset) / self.factor
        return (value - self.offset) / self.factor


@dataclass
class QuantityKind:
    key: str
    label: str
    dimension: Dimension
    si_unit: str
    units: list[str] = field(default_factory=list)


_BUILTIN_UNITS: list[tuple[str, str, Dimension, float]] = [
    ("1", "dimensionless", D.DIMENSIONLESS, 1.0),
    ("%", "percent", D.DIMENSIONLESS, 0.01),
    # length
    ("m", "metre", D.LENGTH, 1.0), ("km", "kilometre", D.LENGTH, 1e3), ("cm", "centimetre", D.LENGTH, 1e-2),
    ("mm", "millimetre", D.LENGTH, 1e-3), ("µm", "micrometre", D.LENGTH, 1e-6), ("in", "inch", D.LENGTH, 0.0254),
    ("ft", "foot", D.LENGTH, 0.3048), ("yd", "yard", D.LENGTH, 0.9144), ("mi", "mile", D.LENGTH, 1609.344),
    # area / volume
    ("m²", "square metre", D.AREA, 1.0), ("cm²", "square centimetre", D.AREA, 1e-4),
    ("mm²", "square millimetre", D.AREA, 1e-6), ("in²", "square inch", D.AREA, 0.0254 ** 2),
    ("m³", "cubic metre", D.VOLUME, 1.0), ("L", "litre", D.VOLUME, 1e-3), ("cm³", "cubic centimetre", D.VOLUME, 1e-6),
    ("mm³", "cubic millimetre", D.VOLUME, 1e-9), ("in³", "cubic inch", D.VOLUME, 0.0254 ** 3),
    # mass
    ("kg", "kilogram", D.MASS, 1.0), ("g", "gram", D.MASS, 1e-3), ("mg", "milligram", D.MASS, 1e-6),
    ("t", "tonne", D.MASS, 1e3), ("lb", "pound", D.MASS, 0.45359237), ("oz", "ounce", D.MASS, 0.028349523125),
    # time
    ("s", "second", D.TIME, 1.0), ("ms", "millisecond", D.TIME, 1e-3), ("µs", "microsecond", D.TIME, 1e-6),
    ("min", "minute", D.TIME, 60.0), ("h", "hour", D.TIME, 3600.0),
    ("Hz", "hertz", D.FREQUENCY, 1.0), ("kHz", "kilohertz", D.FREQUENCY, 1e3),
    # angle
    ("rad", "radian", D.ANGLE, 1.0), ("deg", "degree", D.ANGLE, np.pi / 180), ("rev", "revolution", D.ANGLE, 2 * np.pi),
    # velocity
    ("m/s", "metre per second", D.VELOCITY, 1.0), ("km/h", "kilometre per hour", D.VELOCITY, 1 / 3.6),
    ("cm/s", "centimetre per second", D.VELOCITY, 1e-2), ("mm/s", "millimetre per second", D.VELOCITY, 1e-3),
    ("ft/s", "foot per second", D.VELOCITY, 0.3048), ("mph", "mile per hour", D.VELOCITY, 0.44704),
    ("kn", "knot", D.VELOCITY, 1852 / 3600),
    # acceleration
    ("m/s²", "metre per second squared", D.ACCELERATION, 1.0), ("ft/s²", "foot per second squared", D.ACCELERATION, 0.3048),
    ("g₀", "standard gravity", D.ACCELERATION, 9.80665), ("mm/s²", "millimetre per second squared", D.ACCELERATION, 1e-3),
    # force
    ("N", "newton", D.FORCE, 1.0), ("kN", "kilonewton", D.FORCE, 1e3), ("mN", "millinewton", D.FORCE, 1e-3),
    ("lbf", "pound-force", D.FORCE, 4.4482216152605), ("kgf", "kilogram-force", D.FORCE, 9.80665),
    ("dyn", "dyne", D.FORCE, 1e-5),
    # energy
    ("J", "joule", D.ENERGY, 1.0), ("kJ", "kilojoule", D.ENERGY, 1e3), ("MJ", "megajoule", D.ENERGY, 1e6),
    ("Wh", "watt-hour", D.ENERGY, 3600.0), ("kWh", "kilowatt-hour", D.ENERGY, 3.6e6), ("cal", "calorie", D.ENERGY, 4.184),
    ("eV", "electronvolt", D.ENERGY, 1.602176634e-19), ("ft·lbf", "foot-pound", D.ENERGY, 1.3558179483314004),
    # power
    ("W", "watt", D.POWER, 1.0), ("kW", "kilowatt", D.POWER, 1e3), ("MW", "megawatt", D.POWER, 1e6),
    ("hp", "horsepower (mechanical)", D.POWER, 745.69987158227022),
    # pressure
    ("Pa", "pascal", D.PRESSURE, 1.0), ("kPa", "kilopascal", D.PRESSURE, 1e3), ("MPa", "megapascal", D.PRESSURE, 1e6),
    ("GPa", "gigapascal", D.PRESSURE, 1e9), ("bar", "bar", D.PRESSURE, 1e5), ("atm", "atmosphere", D.PRESSURE, 101325.0),
    ("psi", "pound per square inch", D.PRESSURE, 6894.757293168361),
    # angular velocity / acceleration
    ("rad/s", "radian per second", D.ANGULAR_VELOCITY, 1.0), ("deg/s", "degree per second", D.ANGULAR_VELOCITY, np.pi / 180),
    ("rpm", "revolution per minute", D.ANGULAR_VELOCITY, 2 * np.pi / 60),
    ("rad/s²", "radian per second squared", D.ANGULAR_ACCELERATION, 1.0),
    ("deg/s²", "degree per second squared", D.ANGULAR_ACCELERATION, np.pi / 180),
    # torque (N·m) – stored with torque dimension
    ("N·m", "newton metre", D.TORQUE, 1.0), ("kN·m", "kilonewton metre", D.TORQUE, 1e3),
    ("N·mm", "newton millimetre", D.TORQUE, 1e-3), ("lbf·ft", "pound-force foot", D.TORQUE, 1.3558179483314004),
    ("lbf·in", "pound-force inch", D.TORQUE, 0.1129848290276167),
    # momentum
    ("kg·m/s", "kilogram metre per second", D.MOMENTUM, 1.0), ("N·s", "newton second", D.MOMENTUM, 1.0),
    ("kg·m²/s", "kilogram square metre per second", D.ANGULAR_MOMENTUM, 1.0),
    # density
    ("kg/m³", "kilogram per cubic metre", D.DENSITY, 1.0), ("g/cm³", "gram per cubic centimetre", D.DENSITY, 1e3),
    ("kg/L", "kilogram per litre", D.DENSITY, 1e3), ("lb/ft³", "pound per cubic foot", D.DENSITY, 16.018463373960138),
    ("lb/in³", "pound per cubic inch", D.DENSITY, 27679.904710203125),
    ("kg/m²", "kilogram per square metre", D.AREAL_DENSITY, 1.0), ("g/cm²", "gram per square centimetre", D.AREAL_DENSITY, 10.0),
    # inertia
    ("N/m", "newton per metre", D.STIFFNESS, 1.0), ("kN/m", "kilonewton per metre", D.STIFFNESS, 1e3),
    ("N/mm", "newton per millimetre", D.STIFFNESS, 1e3), ("lbf/in", "pound-force per inch", D.STIFFNESS, 175.12683524647636),
    ("N·s/m", "newton second per metre", D.LINEAR_DAMPING, 1.0), ("kg/s", "kilogram per second", D.LINEAR_DAMPING, 1.0),
    ("kg·m²", "kilogram square metre", D.INERTIA, 1.0), ("g·cm²", "gram square centimetre", D.INERTIA, 1e-7),
    ("lb·ft²", "pound square foot", D.INERTIA, 0.0421401100938048), ("lb·in²", "pound square inch", D.INERTIA, 0.00029263965343),
]

_QUANTITIES: list[tuple[str, str, Dimension, str]] = [
    ("dimensionless", "Dimensionless", D.DIMENSIONLESS, "1"),
    ("length", "Length", D.LENGTH, "m"),
    ("area", "Area", D.AREA, "m²"),
    ("volume", "Volume", D.VOLUME, "m³"),
    ("mass", "Mass", D.MASS, "kg"),
    ("time", "Time", D.TIME, "s"),
    ("frequency", "Frequency", D.FREQUENCY, "Hz"),
    ("angle", "Angle", D.ANGLE, "rad"),
    ("velocity", "Velocity", D.VELOCITY, "m/s"),
    ("acceleration", "Acceleration", D.ACCELERATION, "m/s²"),
    ("force", "Force", D.FORCE, "N"),
    ("energy", "Energy", D.ENERGY, "J"),
    ("power", "Power", D.POWER, "W"),
    ("pressure", "Pressure", D.PRESSURE, "Pa"),
    ("angular_velocity", "Angular velocity", D.ANGULAR_VELOCITY, "rad/s"),
    ("angular_acceleration", "Angular acceleration", D.ANGULAR_ACCELERATION, "rad/s²"),
    ("torque", "Torque", D.TORQUE, "N·m"),
    ("momentum", "Momentum", D.MOMENTUM, "kg·m/s"),
    ("angular_momentum", "Angular momentum", D.ANGULAR_MOMENTUM, "kg·m²/s"),
    ("density", "Density", D.DENSITY, "kg/m³"),
    ("areal_density", "Areal density", D.AREAL_DENSITY, "kg/m²"),
    ("inertia", "Moment of inertia", D.INERTIA, "kg·m²"),
    ("stiffness", "Stiffness", D.STIFFNESS, "N/m"),
    ("damping", "Damping coefficient", D.LINEAR_DAMPING, "N·s/m"),
]

# Torque and energy share an SI dimension; keep their unit lists separate for display.
_EXCLUSIVE = {"torque": {"N·m", "kN·m", "N·mm", "lbf·ft", "lbf·in"}}
_SYMBOL_RE = re.compile(r"^[^\s\d][^\s]{0,15}$")


class UnitRegistry:
    def __init__(self) -> None:
        self._units: dict[str, Unit] = {}
        self._quantities: dict[str, QuantityKind] = {}
        self._preferred: dict[str, str] = {}
        self._property_units: dict[str, str] = {}  # property path -> display unit override
        for sym, name, dim, factor in _BUILTIN_UNITS:
            self._units[sym] = Unit(sym, name, dim, float(factor))
        for key, label, dim, si in _QUANTITIES:
            self._quantities[key] = QuantityKind(key, label, dim, si)
        self._rebuild_quantity_units()

    # ---- lookup -----------------------------------------------------------
    def unit(self, symbol: str) -> Unit:
        try:
            return self._units[symbol]
        except KeyError:
            raise UnitError(f"Unknown unit '{symbol}'") from None

    def has_unit(self, symbol: str) -> bool:
        return symbol in self._units

    def quantity(self, key: str) -> QuantityKind:
        try:
            return self._quantities[key]
        except KeyError:
            raise UnitError(f"Unknown quantity '{key}'") from None

    def quantities(self) -> list[QuantityKind]:
        return list(self._quantities.values())

    def units_for(self, quantity: str) -> list[str]:
        return list(self.quantity(quantity).units)

    def quantity_for_unit(self, symbol: str) -> str | None:
        for q in self._quantities.values():
            if symbol in q.units:
                return q.key
        return None

    def quantities_for_dimension(self, dim: Dimension) -> list[QuantityKind]:
        return [q for q in self._quantities.values() if q.dimension == dim]

    def parse_expression(self, text: str) -> tuple[float, Dimension]:
        """SI factor and dimension of a unit expression such as 'kN·mm', 'furlong/fortnight' or 'kg·m²/s'."""
        try:
            return parse_unit_expression(text, self)
        except UnitExpressionError as exc:
            raise UnitError(str(exc)) from None

    def describe_dimension(self, dim: Dimension) -> str:
        """'[L·T^-1] = m/s (Velocity)' style description of a dimension."""
        names = ", ".join(q.label for q in self.quantities_for_dimension(dim))
        return f"[{dim}] = {si_unit_text(dim)}" + (f" ({names})" if names else "")

    def custom_units(self) -> list[Unit]:
        return [u for u in self._units.values() if u.custom]

    # ---- conversion -------------------------------------------------------
    def convert(self, value, from_unit: str, to_unit: str):
        a, b = self.unit(from_unit), self.unit(to_unit)
        if a.dimension != b.dimension:
            raise UnitError(f"Cannot convert '{from_unit}' [{a.dimension}] to '{to_unit}' [{b.dimension}]")
        return b.from_si(a.to_si(value))

    def to_si(self, value, unit: str):
        return self.unit(unit).to_si(value)

    def from_si(self, value, unit: str):
        return self.unit(unit).from_si(value)

    # ---- preferences --------------------------------------------------------
    def preferred_unit(self, quantity: str) -> str:
        return self._preferred.get(quantity, self.quantity(quantity).si_unit)

    def set_preferred_unit(self, quantity: str, symbol: str) -> None:
        q = self.quantity(quantity)
        if symbol not in q.units:
            raise UnitError(f"Unit '{symbol}' is not valid for {q.label.lower()}")
        self._preferred[quantity] = symbol

    def preferences(self) -> dict[str, str]:
        return dict(self._preferred)

    def reset_preferences(self) -> None:
        self._preferred.clear()
        self._property_units.clear()

    def display_unit(self, path: str, quantity: str) -> str:
        """Display unit for a property path: per-property override, else the project preference."""
        sym = self._property_units.get(path)
        if sym and sym in self._units and sym in self.quantity(quantity).units:
            return sym
        return self.preferred_unit(quantity)

    def set_display_unit(self, path: str, quantity: str, symbol: str | None) -> None:
        if symbol is None or symbol == self.preferred_unit(quantity):
            self._property_units.pop(path, None)
            return
        if symbol not in self.quantity(quantity).units:
            raise UnitError(f"Unit '{symbol}' is not valid for {quantity}")
        self._property_units[path] = symbol

    # ---- custom units -----------------------------------------------------
    def add_custom_unit(self, symbol: str, name: str, base_unit: str, factor: float) -> Unit:
        """Define `1 symbol = factor × base` where base is a unit or a unit expression ('kN·mm', 'ft/min').

        The resulting dimension must belong to a known quantity, so the unit can be selected for display.
        """
        symbol = symbol.strip()
        if not _SYMBOL_RE.match(symbol):
            raise UnitError("Unit symbol must be 1-16 non-space characters and not start with a digit")
        if symbol in self._units:
            raise UnitError(f"Unit '{symbol}' already exists")
        if not np.isfinite(factor) or factor <= 0:
            raise UnitError("Conversion factor must be a positive finite number")
        base_unit = base_unit.strip()
        if self.has_unit(base_unit):
            base = self.unit(base_unit)
            if base.offset:
                raise UnitError("Custom units cannot be based on offset units")
            base_factor, dim = base.factor, base.dimension
        else:
            base_factor, dim = self.parse_expression(base_unit)
        if not self.quantities_for_dimension(dim):
            raise UnitError(f"'{base_unit}' has the dimension {si_unit_text(dim)} [{dim}], which is not a quantity "
                            "known to the application; custom units must be compatible with an existing quantity")
        unit = Unit(symbol, name.strip() or symbol, dim, base_factor * float(factor), custom=True)
        self._units[symbol] = unit
        self._rebuild_quantity_units()
        return unit

    def remove_custom_unit(self, symbol: str) -> None:
        u = self.unit(symbol)
        if not u.custom:
            raise UnitError("Built-in units cannot be removed")
        del self._units[symbol]
        self._preferred = {k: v for k, v in self._preferred.items() if v != symbol}
        self._property_units = {k: v for k, v in self._property_units.items() if v != symbol}
        self._rebuild_quantity_units()

    def _rebuild_quantity_units(self) -> None:
        for q in self._quantities.values():
            q.units = []
            for sym, u in self._units.items():
                if u.dimension != q.dimension:
                    continue
                exclusive_owner = next((k for k, s in _EXCLUSIVE.items() if sym in s), None)
                if exclusive_owner is not None and exclusive_owner != q.key:
                    continue
                if q.key in _EXCLUSIVE and sym not in _EXCLUSIVE[q.key] and not u.custom:
                    continue
                q.units.append(sym)

    # ---- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "preferred": dict(self._preferred),
            "property_units": dict(self._property_units),
            "custom_units": [
                {"symbol": u.symbol, "name": u.name, "dimension": list(u.dimension.exponents), "si_factor": u.factor}
                for u in self.custom_units()
            ],
        }

    def load_dict(self, data: dict) -> list[str]:
        """Restore custom units and preferences. Returns a list of warnings."""
        warnings: list[str] = []
        for u in [u for u in self._units.values() if u.custom]:
            del self._units[u.symbol]
        self._preferred.clear()
        self._property_units.clear()
        for cu in data.get("custom_units", []):
            try:
                sym = str(cu["symbol"])
                dim = Dimension(tuple(int(x) for x in cu["dimension"]))
                factor = float(cu["si_factor"])
                if sym in self._units or not _SYMBOL_RE.match(sym) or factor <= 0 or len(dim.exponents) != 8:
                    raise UnitError(f"invalid custom unit '{sym}'")
                self._units[sym] = Unit(sym, str(cu.get("name", sym)), dim, factor, custom=True)
            except (KeyError, TypeError, ValueError) as exc:
                warnings.append(f"Skipped custom unit: {exc}")
        self._rebuild_quantity_units()
        for q, sym in data.get("preferred", {}).items():
            try:
                self.set_preferred_unit(q, sym)
            except UnitError as exc:
                warnings.append(str(exc))
        for path, sym in dict(data.get("property_units", {})).items():
            if isinstance(sym, str) and sym in self._units:
                self._property_units[str(path)] = sym
        return warnings


_default_registry: UnitRegistry | None = None


def default_registry() -> UnitRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = UnitRegistry()
    return _default_registry
