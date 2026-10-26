"""Plot definitions (saved in the project) and the data they show, independent of the plotting widgets."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..units.unit_registry import UnitRegistry
from .variables import Evaluator, Variable, VariableError, info, label

LINE_STYLES = ("solid", "dash", "dot", "dashdot", "none")
MARKERS = ("none", "o", "s", "t", "d", "+", "x")
PALETTE = ([0.30, 0.62, 0.95], [0.95, 0.55, 0.20], [0.35, 0.80, 0.45], [0.85, 0.35, 0.55], [0.65, 0.50, 0.95],
           [0.95, 0.80, 0.30], [0.30, 0.80, 0.80], [0.75, 0.75, 0.75])


class PlotSpecError(ValueError):
    pass


@dataclass
class SeriesSpec:
    x: Variable
    y: Variable
    z: Variable | None = None
    label: str = ""
    color: list[float] = field(default_factory=lambda: list(PALETTE[0]))
    width: float = 2.0
    style: str = "solid"
    marker: str = "none"
    marker_size: float = 6.0

    def to_dict(self) -> dict:
        return {"x": self.x.to_dict(), "y": self.y.to_dict(), "z": self.z.to_dict() if self.z else None,
                "label": self.label, "color": list(self.color), "width": self.width, "style": self.style,
                "marker": self.marker, "marker_size": self.marker_size}

    @staticmethod
    def from_dict(d: dict) -> "SeriesSpec":
        try:
            s = SeriesSpec(Variable.from_dict(d["x"]), Variable.from_dict(d["y"]),
                           Variable.from_dict(d["z"]) if d.get("z") else None, str(d.get("label", "")),
                           [float(c) for c in d.get("color", PALETTE[0])][:3], float(d.get("width", 2.0)),
                           str(d.get("style", "solid")), str(d.get("marker", "none")), float(d.get("marker_size", 6)))
        except (KeyError, TypeError, ValueError) as exc:
            raise PlotSpecError(f"Invalid plot series: {exc}") from None
        if s.style not in LINE_STYLES or s.marker not in MARKERS:
            raise PlotSpecError("Invalid line style or marker")
        if len(s.color) != 3 or not (0.1 <= s.width <= 20) or not (1 <= s.marker_size <= 40):
            raise PlotSpecError("Invalid series colour, width or marker size")
        return s


@dataclass
class PlotSpec:
    title: str = "Plot"
    kind: str = "2d"                          # 2d | 3d
    series: list[SeriesSpec] = field(default_factory=list)
    x_log: bool = False
    y_log: bool = False
    grid: bool = True
    legend: bool = True
    x_limits: tuple[float, float] | None = None   # display units; None = automatic
    y_limits: tuple[float, float] | None = None
    x_label: str = ""                          # empty = automatic (quantity and unit)
    y_label: str = ""
    z_label: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "kind": self.kind, "series": [s.to_dict() for s in self.series],
                "x_log": self.x_log, "y_log": self.y_log, "grid": self.grid, "legend": self.legend,
                "x_limits": list(self.x_limits) if self.x_limits else None,
                "y_limits": list(self.y_limits) if self.y_limits else None,
                "x_label": self.x_label, "y_label": self.y_label, "z_label": self.z_label}

    @staticmethod
    def from_dict(d: dict) -> "PlotSpec":
        if not isinstance(d, dict):
            raise PlotSpecError("A plot definition must be an object")

        def limits(v):
            if v is None:
                return None
            a, b = float(v[0]), float(v[1])
            if not (math.isfinite(a) and math.isfinite(b) and a < b):
                raise PlotSpecError("Axis limits must be finite with min < max")
            return a, b
        try:
            p = PlotSpec(str(d.get("title", "Plot")), str(d.get("kind", "2d")),
                         [SeriesSpec.from_dict(s) for s in d.get("series", [])], bool(d.get("x_log", False)),
                         bool(d.get("y_log", False)), bool(d.get("grid", True)), bool(d.get("legend", True)),
                         limits(d.get("x_limits")), limits(d.get("y_limits")), str(d.get("x_label", "")),
                         str(d.get("y_label", "")), str(d.get("z_label", "")))
        except (TypeError, ValueError) as exc:
            raise PlotSpecError(f"Invalid plot definition: {exc}") from None
        if p.kind not in ("2d", "3d"):
            raise PlotSpecError(f"Unknown plot kind '{p.kind}'")
        if p.kind == "3d" and any(s.z is None for s in p.series):
            raise PlotSpecError("3D plots need X, Y and Z variables for every series")
        return p

    @property
    def x_is_time(self) -> bool:
        return self.kind == "2d" and bool(self.series) and all(s.x.quantity == "time" for s in self.series)


@dataclass
class SeriesData:
    spec: SeriesSpec
    label: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None
    t: np.ndarray            # simulated time of every point (for cursors)
    error: str = ""


def axis_unit(var: Variable, units: UnitRegistry, custom_quantities: dict | None = None) -> str | None:
    q = info(var, custom_quantities).unit_quantity
    return units.preferred_unit(q) if q else None


def axis_title(spec: PlotSpec, axis: str, units: UnitRegistry, names: dict | None = None,
               frame_names: dict | None = None, custom_quantities: dict | None = None) -> str:
    """Automatic axis title: the common quantity and unit of all series on this axis."""
    manual = {"x": spec.x_label, "y": spec.y_label, "z": spec.z_label}[axis]
    if manual:
        return manual
    variables = [getattr(s, axis) for s in spec.series if getattr(s, axis) is not None]
    if not variables:
        return ""
    labels = {label(Variable(None, v.quantity, v.component, v.frame), None, frame_names, custom_quantities)
              for v in variables}
    unit_set = {axis_unit(v, units, custom_quantities) for v in variables}
    text = labels.pop() if len(labels) == 1 else " / ".join(sorted({info(v, custom_quantities).label for v in variables}))
    if len(variables) == 1 and variables[0].object_id:
        text = label(variables[0], names, frame_names, custom_quantities)
    unit = unit_set.pop() if len(unit_set) == 1 else None
    return f"{text} [{unit}]" if unit and unit != "1" else text


def series_data(ev: Evaluator, spec: PlotSpec, units: UnitRegistry, names: dict | None = None,
                frame_names: dict | None = None, custom_quantities: dict | None = None) -> list[SeriesData]:
    """Values of every series in display units. A series that cannot be computed carries an error text."""
    out = []
    t = ev.values(Variable.time()) if ev.n else np.zeros(0)
    for s in spec.series:
        name = s.label or label(s.y, names, frame_names, custom_quantities)
        if spec.kind == "3d":
            name = s.label or f"{label(s.x, names, frame_names, custom_quantities)} / …"
        try:
            vals = []
            for var in (s.x, s.y, s.z):
                if var is None:
                    vals.append(None)
                    continue
                v = ev.values(var)
                unit = axis_unit(var, units, custom_quantities)
                vals.append(np.asarray(units.from_si(v, unit), float) if unit else np.asarray(v, float))
            out.append(SeriesData(s, name, vals[0], vals[1], vals[2], t))
        except VariableError as exc:
            empty = np.zeros(0)
            out.append(SeriesData(s, name, empty, empty, None if spec.kind == "2d" else empty, empty, str(exc)))
    return out


def values_at_time(data: list[SeriesData], time: float) -> list[tuple[str, float, float]]:
    """(series label, x, y) of every series at the sample nearest to `time` (synchronized cursor read-out)."""
    out = []
    for d in data:
        if d.error or not len(d.t):
            continue
        i = int(np.clip(np.searchsorted(d.t, time), 0, len(d.t) - 1))
        if i > 0 and abs(d.t[i - 1] - time) < abs(d.t[i] - time):
            i -= 1
        out.append((d.label, float(d.x[i]), float(d.y[i])))
    return out


# ---- quick plots ---------------------------------------------------------------------------------------------------

def quick_plot(kind: str, object_ids: list[str], names: dict | None = None) -> PlotSpec:
    """Ready-made plots for the selected objects."""
    names = names or {}
    t = Variable.time()

    def per_object(quantity, components=("x", "y", "z"), title=""):
        series = []
        for k, oid in enumerate(object_ids):
            for j, c in enumerate(components):
                color = PALETTE[(k * len(components) + j) % len(PALETTE)]
                series.append(SeriesSpec(t, Variable(oid, quantity, c), color=list(color)))
        return PlotSpec(title, "2d", series)
    if kind == "position":
        return per_object("position", title="Position vs time")
    if kind == "velocity":
        return per_object("velocity", title="Velocity vs time")
    if kind == "acceleration":
        return per_object("acceleration", title="Acceleration vs time")
    if kind == "angular":
        return per_object("angular_velocity", ("x", "y", "z", "mag"), title="Angular velocity vs time")
    if kind == "speed":
        return per_object("speed", ("",), title="Speed vs time")
    if kind == "energy":
        return per_object("kinetic_energy", ("",), title="Kinetic energy vs time")
    if kind == "force_displacement":
        series = [SeriesSpec(Variable(oid, "displacement", "mag"), Variable(oid, "net_force", "mag"),
                             color=list(PALETTE[k % len(PALETTE)])) for k, oid in enumerate(object_ids)]
        return PlotSpec("Net force vs displacement", "2d", series)
    if kind == "trajectory3d":
        series = [SeriesSpec(Variable(oid, "com", "x"), Variable(oid, "com", "y"), Variable(oid, "com", "z"),
                             label=names.get(oid, oid), color=list(PALETTE[k % len(PALETTE)]))
                  for k, oid in enumerate(object_ids)]
        return PlotSpec("Trajectories", "3d", series)
    raise PlotSpecError(f"Unknown quick plot '{kind}'")
