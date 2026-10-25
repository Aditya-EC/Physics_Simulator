"""Process assignment, documentation and dependency dialogs (ordinary-user tools).

These dialogs select approved processes, configure their permitted
parameters and targets, and show documentation and diagnostics. They never
edit process definitions (formulas), which are application code.
"""
from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QSplitter, QTableWidget,
                               QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from ...processes.process_base import ProcessAssignment, ProcessDef, TargetFilter
from ...processes.process_registry import process_registry
from ...processes.process_scheduler import ExecutionPlan
from ...properties.property_system import PropertyDef, PropertyError, ValueType
from ...scene.scene_manager import Scene
from ...units.unit_registry import UnitRegistry
from ..widgets.inputs import NumberField, VectorField


def process_doc_html(p: ProcessDef, units: UnitRegistry | None = None) -> str:
    rows = []
    for d in p.parameters:
        unit = units.preferred_unit(d.quantity) if units and d.quantity else ""
        rows.append(f"<tr><td>{html.escape(d.label)}</td><td><code>{d.key}</code></td>"
                    f"<td>{d.vtype.value}</td><td>{html.escape(unit or d.description)}</td></tr>")
    params = ("<table cellspacing='4'><tr><th align='left'>Parameter</th><th align='left'>Key</th>"
              "<th align='left'>Type</th><th align='left'>Unit / note</th></tr>" + "".join(rows) + "</table>"
              if rows else "<i>none</i>")
    targets = (f"exactly {p.min_targets}" if p.max_targets == p.min_targets else f"{p.min_targets} or more")
    return (f"<h3>{html.escape(p.name)}</h3>"
            f"<p style='color:gray'>{p.process_id} · version {p.version} · {p.category} · "
            f"{'per-object' if p.mode == 'per_object' else 'global'} {p.kind}</p>"
            f"<p>{html.escape(p.description)}</p>"
            f"<p><b>Mathematical definition</b><br><code>{html.escape(p.formula)}</code></p>"
            f"<p><b>Reads:</b> {html.escape(', '.join(p.reads) or '—')}<br>"
            f"<b>Writes:</b> {html.escape(', '.join(p.writes) or 'integration only')}"
            f" {'(summed with other contributions)' if p.kind == 'contribution' and p.writes else ''}"
            f"{'(exclusive)' if p.kind == 'prescribed' else ''}<br>"
            f"<b>Targets:</b> {targets} object(s){' with mass' if p.requires_mass else ''}</p>"
            f"<p><b>Parameters</b><br>{params}</p>"
            + (f"<p><b>Preconditions:</b> {html.escape(p.preconditions)}</p>" if p.preconditions else "")
            + f"<p><b>Error handling:</b> {html.escape(p.error_handling)}</p>")


class ParameterForm(QWidget):
    """Editors for a list of PropertyDefs (values in SI, shown in project units)."""

    def __init__(self, defs: list[PropertyDef], values: dict, units: UnitRegistry, parent=None) -> None:
        super().__init__(parent)
        self.defs = defs
        self.units = units
        self._getters = {}
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        for d in defs:
            unit = units.preferred_unit(d.quantity) if d.quantity else None
            v = values.get(d.key, d.default_value())
            if d.vtype == ValueType.FLOAT:
                w = NumberField(units, unit, 6, d.minimum, d.maximum)
                w.set_si(v)
                self._getters[d.key] = w.si
            elif d.vtype in (ValueType.VEC2, ValueType.VEC3):
                w = VectorField(d.vtype.size, units, unit, 5)
                w.set_si(v)
                self._getters[d.key] = w.si
            elif d.vtype == ValueType.BOOL:
                w = QCheckBox()
                w.setChecked(bool(v))
                self._getters[d.key] = w.isChecked
            elif d.vtype == ValueType.ENUM:
                w = QComboBox()
                w.addItems(list(d.choices))
                w.setCurrentText(str(v))
                self._getters[d.key] = w.currentText
            else:
                w = QLineEdit(str(v))
                self._getters[d.key] = w.text
            label = d.label + (f" [{unit}]" if unit and unit != "1" else "")
            w.setToolTip(d.description)
            form.addRow(label, w)
        if not defs:
            form.addRow(QLabel("This process has no parameters."))

    def values(self) -> dict:
        return {k: g() for k, g in self._getters.items()}


class AssignProcessDialog(QDialog):
    def __init__(self, scene: Scene, units: UnitRegistry, assignment: ProcessAssignment | None = None,
                 preset_ids: list[str] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Process Assignment" if assignment else "Assign Process")
        self.resize(900, 600)
        self.scene, self.units = scene, units
        self.assignment = assignment
        self.result: dict | None = None
        reg = process_registry()

        lay = QVBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        lay.addWidget(split, 1)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.process = QComboBox()
        for cat in reg.categories():
            self.process.addItem(f"— {cat} —")
            self.process.model().item(self.process.count() - 1).setEnabled(False)
            for d in reg.definitions():
                if d.category == cat:
                    self.process.addItem(d.name, d.process_id)
        if assignment:
            self.process.setCurrentIndex(max(self.process.findData(assignment.process_id), 0))
            self.process.setEnabled(False)
        else:
            self.process.setCurrentIndex(1)
        ll.addWidget(QLabel("Process"))
        ll.addWidget(self.process)
        self.doc = QTextBrowser()
        ll.addWidget(self.doc, 1)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        tg = QGroupBox("Targets")
        tl = QVBoxLayout(tg)
        self.objects = QListWidget()
        chosen = set(assignment.targets.ids if assignment else (preset_ids or []))
        for o in scene.objects.values():
            it = QListWidgetItem(f"{o.name}  ({o.id} · {o.typedef.name})")
            it.setData(Qt.ItemDataRole.UserRole, o.id)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if o.id in chosen else Qt.CheckState.Unchecked)
            self.objects.addItem(it)
        tl.addWidget(self.objects, 1)
        tf = QFormLayout()
        self.types = QLineEdit(", ".join(assignment.targets.types) if assignment else "")
        self.types.setPlaceholderText("e.g. basic.sphere, body.point_mass")
        self.types.setToolTip("Also target every object of these type ids (comma separated)")
        self.tags = QLineEdit(", ".join(assignment.targets.tags) if assignment else "")
        self.tags.setPlaceholderText("e.g. projectile")
        self.tags.setToolTip("Also target every object carrying one of these tags (comma separated)")
        tf.addRow("Types", self.types)
        tf.addRow("Tags", self.tags)
        tl.addLayout(tf)
        rl.addWidget(tg, 1)
        self.params_box = QGroupBox("Parameters")
        self.params_lay = QVBoxLayout(self.params_box)
        rl.addWidget(self.params_box)
        lf = QFormLayout()
        self.label = QLineEdit(assignment.label if assignment else "")
        self.label.setPlaceholderText("optional")
        lf.addRow("Label", self.label)
        rl.addLayout(lf)
        self.error = QLabel()
        self.error.setStyleSheet("color: #e5534b;")
        self.error.setWordWrap(True)
        rl.addWidget(self.error)
        split.addWidget(right)
        split.setSizes([420, 480])

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.form: ParameterForm | None = None
        self.process.currentIndexChanged.connect(self._process_changed)
        self._process_changed()

    def _pdef(self) -> ProcessDef | None:
        pid = self.process.currentData()
        return process_registry().get(pid) if pid else None

    def _process_changed(self) -> None:
        p = self._pdef()
        if p is None:
            return
        self.doc.setHtml(process_doc_html(p, self.units))
        if self.form is not None:
            self.form.deleteLater()
        values = self.assignment.params if self.assignment else p.default_params()
        self.form = ParameterForm(p.parameters, values, self.units)
        self.params_lay.addWidget(self.form)

    def _accept(self) -> None:
        p = self._pdef()
        if p is None:
            return
        ids = [self.objects.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.objects.count())
               if self.objects.item(i).checkState() == Qt.CheckState.Checked]
        split = lambda text: [t.strip() for t in text.split(",") if t.strip()]  # noqa: E731
        targets = TargetFilter(ids, split(self.types.text()), split(self.tags.text()))
        if not (targets.ids or targets.types or targets.tags):
            self.error.setText("Choose at least one target object, type or tag.")
            return
        try:
            params = p.coerce_params(self.form.values())
        except PropertyError as exc:
            self.error.setText(str(exc))
            return
        self.result = {"process_id": p.process_id, "targets": targets, "params": params,
                       "label": self.label.text().strip()}
        self.accept()


class ProcessDocumentationDialog(QDialog):
    def __init__(self, units: UnitRegistry, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Process Documentation")
        self.resize(860, 560)
        lay = QHBoxLayout(self)
        self.list = QListWidget()
        self.list.setMaximumWidth(280)
        self.doc = QTextBrowser()
        defs = process_registry().definitions()
        for d in defs:
            it = QListWidgetItem(f"{d.name}\n{d.category} · {d.process_id}")
            it.setData(Qt.ItemDataRole.UserRole, d.process_id)
            self.list.addItem(it)
        self.list.currentRowChanged.connect(
            lambda r: self.doc.setHtml(process_doc_html(defs[r], units)) if r >= 0 else None)
        lay.addWidget(self.list)
        lay.addWidget(self.doc, 1)
        self.list.setCurrentRow(0)


class DependencyDialog(QDialog):
    """Reads/writes of every enabled process, per-object write map and diagnostics."""

    def __init__(self, plan: ExecutionPlan, scene: Scene, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Process Dependencies & Diagnostics")
        self.resize(820, 620)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("<b>Process inputs and outputs</b> (contributions are summed; prescribed values are "
                             "exclusive; every process reads the shared state of the current solver stage)"))
        rows = plan.reads_writes_table()
        t = QTableWidget(len(rows), 4)
        t.setHorizontalHeaderLabels(["Assignment", "Process", "Reads", "Writes"])
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                t.setItem(r, c, QTableWidgetItem(v))
        lay.addWidget(t, 2)
        lay.addWidget(QLabel("<b>Writers per object</b>"))
        items = sorted(plan.writers.items())
        w = QTableWidget(len(items), 3)
        w.setHorizontalHeaderLabels(["Object", "Quantity", "Written by"])
        w.verticalHeader().setVisible(False)
        w.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for r, ((oid, q), uids) in enumerate(items):
            name = scene.get(oid).name if oid in scene else oid
            for c, v in enumerate((f"{name} ({oid})", q, ", ".join(uids))):
                w.setItem(r, c, QTableWidgetItem(v))
        lay.addWidget(w, 2)
        lay.addWidget(QLabel("<b>Diagnostics</b>"))
        diag = QTextBrowser()
        if plan.diagnostics:
            diag.setHtml("<br>".join(
                f"<span style='color:{'#e5534b' if d.level == 'error' else '#d9a33a'}'>{d.level.upper()}</span> "
                f"{html.escape(d.message)}" for d in plan.diagnostics))
        else:
            diag.setHtml("<span style='color:#4fb36b'>No problems: the plan is valid.</span>")
        lay.addWidget(diag, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        lay.addWidget(buttons)
