"""Dialog for creating or editing a custom object property."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
                               QSpinBox, QVBoxLayout)

from ...properties.custom_properties import CUSTOM_TYPES, CustomProperty, make_key
from ...properties.property_system import KEY_RE, PropertyError, ValueType
from ...units.unit_registry import UnitRegistry

TYPE_LABELS = {
    ValueType.FLOAT: "Floating-point scalar", ValueType.INT: "Integer", ValueType.BOOL: "Boolean",
    ValueType.STRING: "Text", ValueType.VEC2: "2D vector", ValueType.VEC3: "3D vector",
}


class CustomPropertyDialog(QDialog):
    def __init__(self, units: UnitRegistry, existing_keys: set[str], data: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Custom Property" if data else "Add Custom Property")
        self.setMinimumWidth(440)
        self.units = units
        self.existing = set(existing_keys) - ({data["key"]} if data else set())
        self._key_touched = bool(data)
        self.result_data: dict | None = None

        lay = QVBoxLayout(self)
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Drag coefficient")
        self.key = QLineEdit()
        self.key.setPlaceholderText("drag_coefficient")
        self.key.setToolTip("Stable internal key used by processes, plots, labels and exports.\n"
                            "Lowercase letters, digits and underscores.")
        self.type = QComboBox()
        for t in CUSTOM_TYPES:
            self.type.addItem(TYPE_LABELS[t], t.value)
        self.quantity = QComboBox()
        self.quantity.addItem("None (dimensionless / no unit)", "")
        for q in units.quantities():
            if q.key != "dimensionless":
                self.quantity.addItem(f"{q.label} [{q.si_unit}]", q.key)
        self.value = QLineEdit()
        self.default = QLineEdit()
        self.description = QLineEdit()
        self.read_only = QCheckBox("Read-only in the inspector")
        self.precision = QSpinBox()
        self.precision.setRange(0, 12)
        self.precision.setValue(4)
        self.minimum = QLineEdit()
        self.maximum = QLineEdit()
        self.minimum.setPlaceholderText("none")
        self.maximum.setPlaceholderText("none")
        form.addRow("Name", self.name)
        form.addRow("Key", self.key)
        form.addRow("Type", self.type)
        form.addRow("Quantity", self.quantity)
        form.addRow("Value", self.value)
        form.addRow("Default", self.default)
        form.addRow("Description", self.description)
        form.addRow("Precision", self.precision)
        form.addRow("Minimum", self.minimum)
        form.addRow("Maximum", self.maximum)
        form.addRow("", self.read_only)
        lay.addLayout(form)
        self.hint = QLabel()
        self.hint.setObjectName("Muted")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        self.error = QLabel()
        self.error.setStyleSheet("color: #e5534b;")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self.name.textEdited.connect(self._name_edited)
        self.key.textEdited.connect(lambda _: setattr(self, "_key_touched", True))
        self.type.currentIndexChanged.connect(self._type_changed)
        self.quantity.currentIndexChanged.connect(self._type_changed)
        if data:
            self._load(data)
        self._type_changed()

    def _load(self, d: dict) -> None:
        self.name.setText(d.get("name", ""))
        self.key.setText(d.get("key", ""))
        self.type.setCurrentIndex(max(0, self.type.findData(d.get("type"))))
        self.quantity.setCurrentIndex(max(0, self.quantity.findData(d.get("quantity") or "")))
        self.value.setText(self._fmt(d.get("value"), d.get("quantity")))
        self.default.setText(self._fmt(d.get("default"), d.get("quantity")) if d.get("default") is not None else "")
        self.description.setText(d.get("description", ""))
        self.read_only.setChecked(bool(d.get("read_only")))
        self.precision.setValue(int(d.get("precision", 4)))
        self.minimum.setText("" if d.get("minimum") is None else repr(d["minimum"]))
        self.maximum.setText("" if d.get("maximum") is None else repr(d["maximum"]))

    def _fmt(self, v, quantity) -> str:
        if v is None:
            return ""
        unit = self.units.preferred_unit(quantity) if quantity else None
        if isinstance(v, list):
            vals = [self.units.from_si(x, unit) if unit else x for x in v]
            return ", ".join(f"{x:g}" for x in vals)
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)) and unit:
            return f"{self.units.from_si(v, unit):g}"
        return str(v)

    def _name_edited(self, text: str) -> None:
        if not self._key_touched:
            self.key.setText(make_key(text))

    def _vtype(self) -> ValueType:
        return ValueType(self.type.currentData())

    def _type_changed(self) -> None:
        t = self._vtype()
        numeric = t in (ValueType.FLOAT, ValueType.INT, ValueType.VEC2, ValueType.VEC3)
        for w in (self.quantity, self.minimum, self.maximum, self.precision):
            w.setEnabled(numeric)
        q = self.quantity.currentData() if numeric else ""
        unit = self.units.preferred_unit(q) if q else ""
        examples = {ValueType.FLOAT: "1.25", ValueType.INT: "3", ValueType.BOOL: "true / false",
                    ValueType.STRING: "any text", ValueType.VEC2: "1.0, 2.0", ValueType.VEC3: "0, 0, 9.81"}
        self.hint.setText(f"Enter values as {examples[t]}" + (f" in {unit}" if unit else "") +
                          ". Values are stored in canonical SI units.")

    def _parse(self, text: str, t: ValueType, unit: str | None):
        text = text.strip()
        if text == "":
            return None
        if t in (ValueType.VEC2, ValueType.VEC3):
            parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
            vals = [float(p) for p in parts]
            return [self.units.to_si(v, unit) if unit else v for v in vals]
        if t == ValueType.FLOAT:
            v = float(text)
            return self.units.to_si(v, unit) if unit else v
        if t == ValueType.INT:
            return int(text)
        if t == ValueType.BOOL:
            return text
        return text

    def _accept(self) -> None:
        try:
            t = self._vtype()
            key = self.key.text().strip()
            if not KEY_RE.match(key):
                raise PropertyError("Key must start with a letter or underscore and use only a-z, 0-9 and _")
            if key in self.existing:
                raise PropertyError(f"Key '{key}' is already used on this object")
            numeric = t in (ValueType.FLOAT, ValueType.INT, ValueType.VEC2, ValueType.VEC3)
            quantity = (self.quantity.currentData() or None) if numeric else None
            unit = self.units.preferred_unit(quantity) if quantity else None
            lim_unit = unit if t != ValueType.INT else None
            minimum = self._parse(self.minimum.text(), ValueType.FLOAT, lim_unit) if numeric else None
            maximum = self._parse(self.maximum.text(), ValueType.FLOAT, lim_unit) if numeric else None
            data = {
                "key": key, "name": self.name.text().strip(), "type": t.value,
                "value": self._parse(self.value.text(), t, unit), "default": self._parse(self.default.text(), t, unit),
                "quantity": quantity, "description": self.description.text().strip(),
                "read_only": self.read_only.isChecked(), "precision": self.precision.value(),
                "minimum": minimum, "maximum": maximum,
            }
            prop = CustomProperty.from_dict(data)  # full validation
        except (PropertyError, ValueError) as exc:
            self.error.setText(str(exc))
            return
        self.result_data = prop.to_dict()
        self.accept()
