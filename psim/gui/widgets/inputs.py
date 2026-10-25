"""Inspector input widgets: unit-aware numeric fields, vector editors, colour buttons, badges."""
from __future__ import annotations

import math

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QValidator
from PySide6.QtWidgets import (QColorDialog, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSizePolicy, QSpinBox,
                               QToolButton, QWidget)

from ...properties.property_system import Origin
from ...units.unit_registry import UnitRegistry
from .. import theme


class NumberField(QDoubleSpinBox):
    """Floating-point field that displays a canonical SI value in a chosen unit.

    The full SI value is kept internally; the display uses `precision`
    significant decimals, switching to scientific notation for very large or
    small magnitudes. `committed` fires only when the user finishes an edit.
    """

    committed = Signal(float)  # SI value

    def __init__(self, units: UnitRegistry | None = None, unit: str | None = None, precision: int = 4,
                 minimum: float | None = None, maximum: float | None = None, parent=None) -> None:
        super().__init__(parent)
        self._units = units
        self._unit = unit
        self._precision = precision
        self._si = 0.0
        self._min_si = -1e15 if minimum is None else minimum
        self._max_si = 1e15 if maximum is None else maximum
        self.setDecimals(15)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._update_range()
        self.editingFinished.connect(self._emit)

    # QDoubleSpinBox sizes itself from the widest value in its (huge) range; use compact hints instead.
    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(84, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(34, super().minimumSizeHint().height())

    def _conv_to_display(self, v: float) -> float:
        return self._units.from_si(v, self._unit) if self._units and self._unit else v

    def _conv_to_si(self, v: float) -> float:
        return self._units.to_si(v, self._unit) if self._units and self._unit else v

    def _update_range(self) -> None:
        a, b = self._conv_to_display(self._min_si), self._conv_to_display(self._max_si)
        self.blockSignals(True)
        self.setRange(min(a, b), max(a, b))
        self.blockSignals(False)

    def set_unit(self, unit: str | None) -> None:
        self._unit = unit
        self._update_range()
        self.set_si(self._si)

    def set_si(self, value: float) -> None:
        self._si = float(value)
        self.blockSignals(True)
        self.setValue(self._conv_to_display(self._si))
        self.blockSignals(False)
        self.lineEdit().setCursorPosition(0)

    def si(self) -> float:
        return self._si

    def _emit(self) -> None:
        disp_old = self._conv_to_display(self._si)
        new = self.value()
        if math.isclose(new, disp_old, rel_tol=1e-12, abs_tol=0.0) or \
                self.textFromValue(new) == self.textFromValue(disp_old):
            return  # unchanged (avoid precision loss from re-committing rounded text)
        self._si = self._conv_to_si(new)
        self.committed.emit(self._si)

    def textFromValue(self, v: float) -> str:  # noqa: N802
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1e7 or a < 10 ** (-self._precision):
            return f"{v:.{max(self._precision - 1, 1)}e}"
        text = f"{v:.{self._precision}f}".rstrip("0").rstrip(".")
        return text if text not in ("-0", "") else "0"

    def valueFromText(self, text: str) -> float:  # noqa: N802
        try:
            return float(text.replace(",", ".").strip() or 0.0)
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        t = text.replace(",", ".").strip()
        if t in ("", "-", "+", ".", "-.", "+."):
            return QValidator.State.Intermediate, text, pos
        try:
            float(t)
            return QValidator.State.Acceptable, text, pos
        except ValueError:
            if t[-1] in "eE+-" and any(c.isdigit() for c in t):
                return QValidator.State.Intermediate, text, pos
            return QValidator.State.Invalid, text, pos

    def wheelEvent(self, e) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()

    def stepBy(self, steps: int) -> None:  # noqa: N802
        super().stepBy(steps)
        self._emit()


class IntField(QSpinBox):
    committed = Signal(int)

    def __init__(self, minimum: int = -2**31, maximum: int = 2**31 - 1, parent=None) -> None:
        super().__init__(parent)
        self.setRange(minimum, maximum)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._last = 0
        self.editingFinished.connect(self._emit)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(84, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(34, super().minimumSizeHint().height())

    def set_int(self, v: int) -> None:
        self._last = int(v)
        self.blockSignals(True)
        self.setValue(int(v))
        self.blockSignals(False)

    def _emit(self) -> None:
        if self.value() != self._last:
            self._last = self.value()
            self.committed.emit(self.value())

    def wheelEvent(self, e) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(e)
            self._emit()
        else:
            e.ignore()


class UnitCombo(QComboBox):
    unitChanged = Signal(str)

    def __init__(self, units: UnitRegistry, quantity: str, current: str, parent=None) -> None:
        super().__init__(parent)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(4)
        self.setMaximumWidth(84)
        self.setToolTip(f"Display unit ({units.quantity(quantity).label.lower()})")
        for sym in units.units_for(quantity):
            self.addItem(sym)
        self.setCurrentText(current)
        self.currentTextChanged.connect(self.unitChanged.emit)

    def wheelEvent(self, e) -> None:  # noqa: N802
        e.ignore()


class VectorField(QWidget):
    committed = Signal(list)  # SI values

    LABELS = ("X", "Y", "Z", "W")

    def __init__(self, n: int, units: UnitRegistry | None = None, unit: str | None = None, precision: int = 4,
                 minimum: float | None = None, maximum: float | None = None, labels: tuple[str, ...] | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.fields: list[NumberField] = []
        labels = labels or self.LABELS[:n]
        for i in range(n):
            lab = QLabel(labels[i])
            lab.setObjectName(f"AxisLabel{labels[i]}" if labels[i] in "XYZ" else "Muted")
            lab.setFixedWidth(11)
            lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            f = NumberField(units, unit, precision, minimum, maximum)
            f.committed.connect(self._emit)
            lay.addWidget(lab)
            lay.addWidget(f, 1)
            self.fields.append(f)

    def set_si(self, values) -> None:
        for f, v in zip(self.fields, values):
            f.set_si(float(v))

    def si(self) -> list[float]:
        return [f.si() for f in self.fields]

    def set_unit(self, unit: str | None) -> None:
        for f in self.fields:
            f.set_unit(unit)

    def setReadOnly(self, ro: bool) -> None:  # noqa: N802
        for f in self.fields:
            f.setReadOnly(ro)

    def _emit(self, _v) -> None:
        self.committed.emit(self.si())


class ColorButton(QToolButton):
    colorChanged = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rgb = [0.5, 0.5, 0.5]
        self.setFixedSize(QSize(48, 22))
        self.clicked.connect(self._pick)
        self.setToolTip("Click to choose a colour")

    def set_rgb(self, rgb) -> None:
        self._rgb = [float(c) for c in rgb[:3]]
        self.update()

    def rgb(self) -> list[float]:
        return list(self._rgb)

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor.fromRgbF(*self._rgb), self, "Choose Colour")
        if c.isValid():
            self.set_rgb([c.redF(), c.greenF(), c.blueF()])
            self.colorChanged.emit(self.rgb())

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        p.setPen(QColor(theme.current().border_strong))
        p.setBrush(QColor.fromRgbF(*self._rgb))
        p.drawRoundedRect(r, 4, 4)
        p.end()


class OriginBadge(QLabel):
    COLORS = {Origin.USER: "#3d8bfd", Origin.CALCULATED: "#9a7cf0", Origin.DERIVED: "#3fb3a6",
              Origin.SIMULATED: "#e39a3b"}

    def __init__(self, origin: Origin, parent=None) -> None:
        super().__init__(origin.badge, parent)
        self.setObjectName("Badge")
        c = self.COLORS[origin]
        self.setStyleSheet(f"background: {c}33; color: {c}; border: 1px solid {c}66;")
        self.setFixedSize(14, 14)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setToolTip(origin.description)
