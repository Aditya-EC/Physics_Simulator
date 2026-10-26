"""Units ▸ Formula Unit Inspector: unit requirements of processes and dimensions of formula expressions."""
from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...processes.dimensional_analysis import CheckEnv
from ...processes.process_registry import ProcessRegistry
from ...processes.unit_inspection import env_for, inspect_expression, process_requirements
from ...units.unit_registry import UnitRegistry


def _table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for c in range(1, len(headers)):
        t.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    return t


def _fill(t: QTableWidget, rows: list[list[str]]) -> None:
    t.setRowCount(len(rows))
    for r, values in enumerate(rows):
        for c, v in enumerate(values):
            t.setItem(r, c, QTableWidgetItem(v))


class FormulaUnitInspector(QDialog):
    def __init__(self, registry: ProcessRegistry, units: UnitRegistry, process_id: str | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Formula Unit Inspector")
        self.resize(900, 640)
        self.registry, self.units = registry, units
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Context"))
        self.context = QComboBox()
        self.context.addItem("No process (constants, literals and self only)", None)
        for d in sorted(registry.definitions(), key=lambda d: (d.category, d.name)):
            self.context.addItem(f"{d.category} ▸ {d.name}  ({d.process_id})", d.process_id)
        self.context.currentIndexChanged.connect(self._context_changed)
        top.addWidget(self.context, 1)
        lay.addLayout(top)

        split = QSplitter(Qt.Orientation.Vertical)
        req = QWidget()
        rl = QVBoxLayout(req)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("<b>Unit requirements of the process</b> — inputs a formula reads and outputs it may "
                            "write (values are SI inside formulas; angles are dimensionless, in radians)"))
        self.requirements = _table(["Name", "Kind", "Type", "Quantity", "SI unit", "Display unit"])
        rl.addWidget(self.requirements)
        split.addWidget(req)

        expr = QWidget()
        el = QVBoxLayout(expr)
        el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(QLabel("<b>Expression</b> — e.g. <code>0.5 * self.mass * dot(self.velocity, self.velocity)"
                            "</code> or <code>9.81[m/s^2] * 2[s]</code>"))
        self.expression = QLineEdit()
        self.expression.setPlaceholderText("Type a formula expression")
        self.expression.textChanged.connect(self.analyse)
        el.addWidget(self.expression)
        self.result = QLabel()
        self.result.setTextFormat(Qt.TextFormat.RichText)
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        el.addWidget(self.result)
        self.terms = _table(["Sub-expression", "Type", "Dimension", "SI unit", "Quantity"])
        el.addWidget(self.terms, 1)
        split.addWidget(expr)
        split.setSizes([220, 380])
        lay.addWidget(split, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        lay.addWidget(buttons)
        self.env = CheckEnv()
        i = self.context.findData(process_id)
        self.context.setCurrentIndex(max(i, 0))
        self._context_changed()

    def _context_changed(self, *_a) -> None:
        pid = self.context.currentData()
        if pid is None:
            self.env = CheckEnv()
            _fill(self.requirements, [])
        else:
            pdef = self.registry.get(pid)
            self.env = env_for(pdef)
            _fill(self.requirements, [[r.name, r.kind, r.type, r.quantity, r.si_unit, r.display_unit]
                                      for r in process_requirements(pdef, self.units)])
        self.analyse()

    def analyse(self, *_a) -> None:
        text = self.expression.text()
        if not text.strip():
            self.result.setText("")
            _fill(self.terms, [])
            return
        rep = inspect_expression(text, self.env, self.units)
        if not rep.ok:
            caret = ""
            if rep.error_pos >= 0:
                caret = (f"<pre style='margin:2px'>{html.escape(text)}\n{' ' * rep.error_pos}^</pre>")
            self.result.setText(f"<span style='color:#e5534b'>✗ {html.escape(rep.message)}</span>{caret}")
            _fill(self.terms, [])
            return
        r = rep.result
        unit = f" = <b>{html.escape(r.si_unit)}</b>" if r.si_unit else ""
        q = f" — {html.escape(r.quantities)}" if r.quantities else ""
        self.result.setText(f"<span style='color:#4fb36b'>✓</span> {html.escape(r.type)} "
                            f"{html.escape(r.dimension)}{unit}{q}")
        _fill(self.terms, [["   " * t.depth + t.text, t.type, t.dimension, t.si_unit, t.quantities]
                           for t in rep.rows])
