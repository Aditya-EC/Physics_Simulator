"""Viewport settings: grids, global axes, snapping, lighting & environment (changes apply live)."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ...rendering.view_settings import ViewSettings
from ...units.unit_registry import UnitRegistry
from ..widgets.inputs import ColorButton, NumberField, VectorField


class ViewportSettingsDialog(QDialog):
    changed = Signal()

    def __init__(self, view: ViewSettings, units: UnitRegistry, tab: str = "grids", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Viewport Settings")
        self.setMinimumWidth(460)
        self.view, self.units = view, units
        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self._tab_index = {}
        self._add_tab("grids", "Grids", self._grids_tab())
        self._add_tab("axes", "Axes", self._axes_tab())
        self._add_tab("snap", "Snapping", self._snap_tab())
        self._add_tab("environment", "Lighting && Environment", self._env_tab())
        self.tabs.setCurrentIndex(self._tab_index.get(tab, 0))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        lay.addWidget(buttons)

    def _add_tab(self, key: str, title: str, w: QWidget) -> None:
        self._tab_index[key] = self.tabs.addTab(w, title)

    # ---- field helpers ------------------------------------------------------------------------
    def _bool(self, form: QFormLayout, label: str, obj, attr: str) -> QCheckBox:
        cb = QCheckBox()
        cb.setChecked(bool(getattr(obj, attr)))
        cb.toggled.connect(lambda v: self._set(obj, attr, bool(v)))
        form.addRow(label, cb)
        return cb

    def _float(self, form: QFormLayout, label: str, obj, attr: str, lo: float, hi: float, step: float = 0.1,
               decimals: int = 2, suffix: str = "") -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(decimals)
        sb.setSingleStep(step)
        sb.setSuffix(suffix)
        sb.setValue(float(getattr(obj, attr)))
        sb.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        sb.valueChanged.connect(lambda v: self._set(obj, attr, float(v)))
        form.addRow(label, sb)
        return sb

    def _length(self, form: QFormLayout, label: str, obj, attr: str, lo: float, hi: float) -> NumberField:
        unit = self.units.preferred_unit("length")
        f = NumberField(self.units, unit, 4, lo, hi)
        f.set_si(getattr(obj, attr))
        f.committed.connect(lambda v: self._set(obj, attr, float(v)))
        form.addRow(f"{label} [{unit}]", f)
        return f

    def _int(self, form: QFormLayout, label: str, obj, attr: str, lo: int, hi: int) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(int(getattr(obj, attr)))
        sb.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        sb.valueChanged.connect(lambda v: self._set(obj, attr, int(v)))
        form.addRow(label, sb)
        return sb

    def _color(self, form: QFormLayout, label: str, obj, attr: str) -> ColorButton:
        b = ColorButton()
        b.set_rgb(getattr(obj, attr))
        b.colorChanged.connect(lambda v: self._set(obj, attr, list(v)))
        form.addRow(label, b)
        return b

    def _set(self, obj, attr: str, value) -> None:
        setattr(obj, attr, value)
        self.changed.emit()

    # ---- tabs --------------------------------------------------------------------------------------------
    def _grids_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        sub = QTabWidget()
        for key in ("XY", "YZ", "XZ"):
            gs = self.view.grids[key]
            page = QWidget()
            form = QFormLayout(page)
            self._bool(form, "Visible", gs, "visible")
            self._color(form, "Colour", gs, "color")
            self._float(form, "Opacity", gs, "opacity", 0.05, 1.0, 0.05)
            self._length(form, "Major spacing", gs, "major_spacing", 1e-4, 1e4)
            self._int(form, "Subdivisions", gs, "subdivisions", 1, 50)
            self._length(form, "Extent (half size)", gs, "extent", 0.01, 1e5)
            self._float(form, "Line thickness", gs, "line_width", 0.5, 6.0, 0.25, suffix=" px")
            self._bool(form, "Labels", gs, "show_labels")
            self._bool(form, "Emphasise origin lines", gs, "emphasize_origin")
            sub.addTab(page, f"{key} plane")
        lay.addWidget(sub)
        return w

    def _axes_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        ax = self.view.axes
        self._bool(form, "Show global axes", ax, "visible")
        unit = self.units.preferred_unit("length")
        origin = VectorField(3, self.units, unit, 4, -1e7, 1e7)
        origin.set_si(ax.origin)
        origin.committed.connect(lambda v: self._set(ax, "origin", [float(c) for c in v]))
        origin.setToolTip("Where the global axes and origin marker are drawn. Display only: global coordinates "
                          "are always measured from the world origin (0, 0, 0). To measure from another point, "
                          "create a named reference frame (Coordinates ▸ Reference Frames).")
        form.addRow(f"Axes drawn at [{unit}]", origin)
        self._length(form, "Arrow length", ax, "length", 1e-3, 1e5)
        self._float(form, "Arrow-head size", ax, "arrow_size", 0.0, 5.0, 0.1, suffix=" ×")
        self._float(form, "Line thickness", ax, "line_width", 0.5, 10.0, 0.5, suffix=" px")
        self._color(form, "X axis colour", ax, "x_color")
        self._color(form, "Y axis colour", ax, "y_color")
        self._color(form, "Z axis colour", ax, "z_color")
        self._bool(form, "Axis labels", ax, "show_labels")
        self._bool(form, "Origin marker", ax, "show_origin_marker")
        self._bool(form, "View orientation triad", ax, "show_view_triad")
        self._bool(form, "Show local axes on all objects", self.view.environment, "show_local_axes")
        return w

    def _snap_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        sn = self.view.snap
        self._bool(form, "Grid snapping (move / placement)", sn, "grid_snap")
        self._length(form, "Translation increment", sn, "translate_increment", 1e-6, 1e4)
        self._bool(form, "Rotation snapping", sn, "rotate_snap")
        self._float(form, "Rotation increment", sn, "rotate_increment_deg", 0.1, 180.0, 1.0, 1, " °")
        self._bool(form, "Scale snapping", sn, "scale_snap")
        self._float(form, "Scale increment", sn, "scale_increment", 0.001, 10.0, 0.05, 3)
        self._bool(form, "Surface snapping (placement)", sn, "surface_snap")
        plane = QComboBox()
        plane.addItems(["XY", "YZ", "XZ"])
        plane.setCurrentText(sn.work_plane)
        plane.currentTextChanged.connect(lambda v: self._set(sn, "work_plane", v))
        form.addRow("Placement plane", plane)
        self._length(form, "Placement plane offset", sn, "work_plane_offset", -1e5, 1e5)
        return w

    def _env_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        env = self.view.environment
        self._color(form, "Background (top)", env, "background_top")
        self._color(form, "Background (bottom)", env, "background_bottom")
        self._bool(form, "Shadows", env, "shadows")
        self._bool(form, "Ground shadow (XY plane)", env, "ground_shadow")
        self._float(form, "Light azimuth", env, "light_azimuth_deg", -180, 180, 5, 1, " °")
        self._float(form, "Light elevation", env, "light_elevation_deg", 5, 90, 5, 1, " °")
        self._float(form, "Light intensity", env, "light_intensity", 0.1, 10.0, 0.1)
        self._float(form, "Ambient light", env, "ambient", 0.0, 2.0, 0.05)
        self._color(form, "Selection highlight", env, "highlight_color")
        self._bool(form, "Object labels (master switch)", env, "show_labels")
        self._bool(form, "Viewport status overlay", env, "show_stats")
        return w
