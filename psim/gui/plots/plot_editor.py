"""Dialog to configure a plot: type, axes, and the series with their variables and styles."""
from __future__ import annotations

import copy

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QVBoxLayout,
                               QWidget)

from ...coordinates.scene_kinematics import reference_choices
from ...recording.plot_spec import LINE_STYLES, MARKERS, PALETTE, PlotSpec, PlotSpecError, SeriesSpec
from ...recording.recording import Recording
from ...recording.variables import PROCESS_OUTPUTS, QUANTITIES, Variable, info, label, object_quantities
from ...scene.scene_manager import Scene
from ..widgets.inputs import ColorButton

COMPONENT_LABELS = {"x": "x", "y": "y", "z": "z", "mag": "magnitude"}


def quantities_for(scene: Scene, rec: Recording | None, oid: str) -> list[str]:
    """Quantities offered for an object: from the recording if it has the object, else from the scene setup."""
    if rec is not None and rec.has((oid, "origin")):
        return object_quantities(rec, oid)
    out = [q for q in QUANTITIES if q not in ("time", "step")]
    obj = scene.find(oid)
    if obj is not None:
        out += [f"custom.{k}" for k in obj.custom]
        for a in scene.assignments_for(oid):
            out += [f"proc:{a.uid}:{w}" for w in PROCESS_OUTPUTS]
    return out


class VariableEditor(QWidget):
    changed = Signal()

    def __init__(self, scene: Scene, rec: Recording | None, parent=None) -> None:
        super().__init__(parent)
        self.scene, self.rec = scene, rec
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.obj = QComboBox()
        self.obj.addItem("Time", "@time")
        self.obj.addItem("Step", "@step")
        for o in scene.objects.values():
            self.obj.addItem(f"{o.name} ({o.id})", o.id)
        self.quantity = QComboBox()
        self.component = QComboBox()
        self.frame = QComboBox()
        for text, ref in reference_choices(scene):
            self.frame.addItem(f"in {text}" if ref else "global", ref)
        for w in (self.obj, self.quantity, self.component, self.frame):
            lay.addWidget(w, 2 if w is self.obj or w is self.quantity else 1)
        self.obj.currentIndexChanged.connect(self._obj_changed)
        self.quantity.currentIndexChanged.connect(self._quantity_changed)
        self.component.currentIndexChanged.connect(self._emit_changed)
        self.frame.currentIndexChanged.connect(self._emit_changed)
        self._obj_changed()

    def _emit_changed(self, *_a) -> None:
        self.changed.emit()

    def _custom_quantities(self) -> dict:
        out = {}
        for o in self.scene.objects.values():
            for k, cp in o.custom.items():
                out[(o.id, k)] = (cp.definition.quantity, cp.definition.vtype.value == "vec3")
        return out

    def _obj_changed(self, *_a) -> None:
        oid = self.obj.currentData()
        self.quantity.blockSignals(True)
        self.quantity.clear()
        if oid not in ("@time", "@step"):
            cq = self._custom_quantities()
            for q in quantities_for(self.scene, self.rec, oid):
                self.quantity.addItem(info(Variable(oid, q), cq).label, q)
        self.quantity.blockSignals(False)
        self.quantity.setEnabled(oid not in ("@time", "@step"))
        self._quantity_changed()

    def _quantity_changed(self, *_a) -> None:
        var = self._base()
        vector = var.object_id is not None and info(var, self._custom_quantities()).vector
        cur = self.component.currentData()
        self.component.blockSignals(True)
        self.component.clear()
        if vector:
            for c, text in COMPONENT_LABELS.items():
                self.component.addItem(text, c)
            self.component.setCurrentIndex(max(self.component.findData(cur), 0))
        self.component.blockSignals(False)
        self.component.setEnabled(vector)
        frame_ok = var.object_id is not None and QUANTITIES.get(var.quantity) is not None \
            and QUANTITIES[var.quantity].frame
        self.frame.setEnabled(frame_ok)
        self.changed.emit()

    def _base(self) -> Variable:
        oid = self.obj.currentData()
        if oid == "@time":
            return Variable.time()
        if oid == "@step":
            return Variable(None, "step")
        return Variable(oid, self.quantity.currentData() or "position")

    def variable(self) -> Variable:
        base = self._base()
        if base.object_id is None:
            return base
        comp = self.component.currentData() if self.component.isEnabled() else ""
        frame = self.frame.currentData() if self.frame.isEnabled() else None
        return Variable(base.object_id, base.quantity, comp or "", frame)

    def set_variable(self, var: Variable) -> None:
        key = "@time" if var.quantity == "time" else "@step" if var.quantity == "step" else var.object_id
        i = self.obj.findData(key)
        if i < 0 and var.object_id:
            self.obj.addItem(f"⚠ missing {var.object_id}", var.object_id)
            i = self.obj.count() - 1
        self.obj.setCurrentIndex(max(i, 0))
        if var.object_id:
            j = self.quantity.findData(var.quantity)
            if j < 0:
                self.quantity.addItem(var.quantity, var.quantity)
                j = self.quantity.count() - 1
            self.quantity.setCurrentIndex(j)
            self.component.setCurrentIndex(max(self.component.findData(var.component), 0))
            self.frame.setCurrentIndex(max(self.frame.findData(var.frame), 0))


class PlotEditorDialog(QDialog):
    def __init__(self, scene: Scene, rec: Recording | None, spec: PlotSpec | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot Settings")
        self.resize(900, 620)
        self.scene, self.rec = scene, rec
        self.spec = copy.deepcopy(spec) if spec else PlotSpec("Plot", "2d", [])
        self._loading = False
        lay = QVBoxLayout(self)
        top = QFormLayout()
        self.title = QLineEdit(self.spec.title)
        self.kind = QComboBox()
        self.kind.addItem("2D (line / scatter)", "2d")
        self.kind.addItem("3D (X, Y, Z)", "3d")
        self.kind.setCurrentIndex(self.kind.findData(self.spec.kind))
        self.kind.currentIndexChanged.connect(self._kind_changed)
        top.addRow("Title", self.title)
        top.addRow("Type", self.kind)
        opts = QHBoxLayout()
        self.grid = QCheckBox("Grid")
        self.grid.setChecked(self.spec.grid)
        self.legend = QCheckBox("Legend")
        self.legend.setChecked(self.spec.legend)
        self.x_log = QCheckBox("Logarithmic X")
        self.x_log.setChecked(self.spec.x_log)
        self.y_log = QCheckBox("Logarithmic Y")
        self.y_log.setChecked(self.spec.y_log)
        for w in (self.grid, self.legend, self.x_log, self.y_log):
            opts.addWidget(w)
        opts.addStretch(1)
        top.addRow("Options", opts)
        self.limits = {}
        for axis in ("x", "y"):
            row = QHBoxLayout()
            auto = QCheckBox("automatic")
            lo, hi = QDoubleSpinBox(), QDoubleSpinBox()
            for sb in (lo, hi):
                sb.setRange(-1e12, 1e12)
                sb.setDecimals(6)
            lim = getattr(self.spec, f"{axis}_limits")
            auto.setChecked(lim is None)
            if lim:
                lo.setValue(lim[0])
                hi.setValue(lim[1])
            auto.toggled.connect(lambda on, a=lo, b=hi: (a.setEnabled(not on), b.setEnabled(not on)))
            auto.toggled.emit(auto.isChecked())
            label_edit = QLineEdit(getattr(self.spec, f"{axis}_label"))
            label_edit.setPlaceholderText("automatic (quantity [unit])")
            for w in (auto, QLabel("min"), lo, QLabel("max"), hi, QLabel("title"), label_edit):
                row.addWidget(w)
            self.limits[axis] = (auto, lo, hi, label_edit)
            top.addRow(f"{axis.upper()} axis", row)
        self.z_label = QLineEdit(self.spec.z_label)
        self.z_label.setPlaceholderText("automatic")
        top.addRow("Z axis title", self.z_label)
        lay.addLayout(top)

        box = QGroupBox("Series")
        bl = QHBoxLayout(box)
        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show_series)
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, slot in (("Add", self._add), ("Duplicate", self._duplicate), ("Remove", self._remove)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(slot)
            row.addWidget(b)
        left.addLayout(row)
        bl.addLayout(left, 1)
        self.editor = QWidget()
        form = QFormLayout(self.editor)
        self.axis_editors = {}
        for axis in ("x", "y", "z"):
            ed = VariableEditor(scene, rec)
            ed.changed.connect(self._store)
            self.axis_editors[axis] = ed
            form.addRow(f"{axis.upper()} variable", ed)
        self.series_label = QLineEdit()
        self.series_label.setPlaceholderText("automatic")
        self.series_label.editingFinished.connect(self._store)
        self.color = ColorButton()
        self.color.colorChanged.connect(lambda _c: self._store())
        self.width = QDoubleSpinBox()
        self.width.setRange(0.1, 20)
        self.width.valueChanged.connect(self._store)
        self.style = QComboBox()
        self.style.addItems(LINE_STYLES)
        self.style.currentIndexChanged.connect(self._store)
        self.marker = QComboBox()
        self.marker.addItems(MARKERS)
        self.marker.currentIndexChanged.connect(self._store)
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(1, 40)
        self.marker_size.valueChanged.connect(self._store)
        form.addRow("Legend label", self.series_label)
        style_row = QHBoxLayout()
        for text, w in (("colour", self.color), ("width", self.width), ("line", self.style), ("marker", self.marker),
                        ("size", self.marker_size)):
            style_row.addWidget(QLabel(text))
            style_row.addWidget(w)
        style_row.addStretch(1)
        form.addRow("Style", style_row)
        bl.addWidget(self.editor, 3)
        lay.addWidget(box, 1)
        self.error = QLabel()
        self.error.setStyleSheet("color: #e5534b;")
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.result_spec: PlotSpec | None = None
        self._refresh_list()
        self._kind_changed()

    # ---- series ----------------------------------------------------------------------------------------
    def _names(self) -> dict:
        return {o.id: o.name for o in self.scene.objects.values()}

    def _refresh_list(self, select: int | None = None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for s in self.spec.series:
            ys = label(s.y, self._names())
            self.list.addItem(s.label or (f"{ys} vs {label(s.x, self._names())}" if self.spec.kind == "2d"
                                          else f"{label(s.x, self._names())}, {ys}, …"))
        self.list.blockSignals(False)
        if self.spec.series:
            self.list.setCurrentRow(len(self.spec.series) - 1 if select is None else select)
        self._show_series(self.list.currentRow())

    def _current(self) -> SeriesSpec | None:
        r = self.list.currentRow()
        return self.spec.series[r] if 0 <= r < len(self.spec.series) else None

    def _add(self) -> None:
        sel = self.scene.selection
        oid = sel[-1] if sel else next(iter(self.scene.objects), None)
        k = len(self.spec.series)
        y = Variable(oid, "position", "z") if oid else Variable(None, "step")
        z = Variable(oid, "com", "z") if oid else None
        x = Variable(oid, "com", "x") if self.spec.kind == "3d" and oid else Variable.time()
        if self.spec.kind == "3d" and oid:
            y = Variable(oid, "com", "y")
        self.spec.series.append(SeriesSpec(x, y, z if self.spec.kind == "3d" else None,
                                           color=list(PALETTE[k % len(PALETTE)])))
        self._refresh_list()

    def _duplicate(self) -> None:
        s = self._current()
        if s is not None:
            dup = copy.deepcopy(s)
            dup.color = list(PALETTE[len(self.spec.series) % len(PALETTE)])
            self.spec.series.append(dup)
            self._refresh_list()

    def _remove(self) -> None:
        r = self.list.currentRow()
        if 0 <= r < len(self.spec.series):
            del self.spec.series[r]
            self._refresh_list(max(0, r - 1))

    def _show_series(self, _row: int) -> None:
        s = self._current()
        self.editor.setEnabled(s is not None)
        if s is None:
            return
        self._loading = True
        try:
            self.axis_editors["x"].set_variable(s.x)
            self.axis_editors["y"].set_variable(s.y)
            self.axis_editors["z"].set_variable(s.z or Variable.time())
            self.series_label.setText(s.label)
            self.color.set_rgb(s.color)
            self.width.setValue(s.width)
            self.style.setCurrentText(s.style)
            self.marker.setCurrentText(s.marker)
            self.marker_size.setValue(s.marker_size)
        finally:
            self._loading = False

    def _store(self, *_a) -> None:
        s = self._current()
        if s is None or self._loading:
            return
        s.x = self.axis_editors["x"].variable()
        s.y = self.axis_editors["y"].variable()
        s.z = self.axis_editors["z"].variable() if self.kind.currentData() == "3d" else None
        s.label = self.series_label.text().strip()
        s.color = list(self.color.rgb())
        s.width, s.style = self.width.value(), self.style.currentText()
        s.marker, s.marker_size = self.marker.currentText(), self.marker_size.value()
        item = self.list.currentItem()
        if item is not None:
            item.setText(s.label or label(s.y, self._names()))

    def _kind_changed(self, *_a) -> None:
        three = self.kind.currentData() == "3d"
        self.spec.kind = self.kind.currentData()
        self.axis_editors["z"].setEnabled(three)
        self.z_label.setEnabled(three)
        for w in (self.x_log, self.y_log):
            w.setEnabled(not three)
        if three:
            for s in self.spec.series:
                if s.z is None:
                    s.z = Variable(s.y.object_id, "com", "z") if s.y.object_id else Variable.time()
        else:
            for s in self.spec.series:
                s.z = None

    # ---- result ------------------------------------------------------------------------------------------
    def collect(self) -> PlotSpec:
        d = self.spec.to_dict()
        d.update(title=self.title.text().strip() or "Plot", kind=self.kind.currentData(), grid=self.grid.isChecked(),
                 legend=self.legend.isChecked(), x_log=self.x_log.isChecked() and self.x_log.isEnabled(),
                 y_log=self.y_log.isChecked() and self.y_log.isEnabled(), z_label=self.z_label.text().strip())
        for axis, (auto, lo, hi, label_edit) in self.limits.items():
            d[f"{axis}_limits"] = None if auto.isChecked() else [lo.value(), hi.value()]
            d[f"{axis}_label"] = label_edit.text().strip()
        return PlotSpec.from_dict(d)

    def _accept(self) -> None:
        try:
            spec = self.collect()
        except PlotSpecError as exc:
            self.error.setText(str(exc))
            return
        if not spec.series:
            self.error.setText("Add at least one series")
            return
        self.result_spec = spec
        self.accept()
