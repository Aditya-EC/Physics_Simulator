"""Simulation ▸ Stop Conditions: Boolean formulas that pause or end a run when they become true."""
from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from ...simulation.conditions import ACTIONS, ConditionError, attach_conditions, validate_conditions

COLUMNS = ("On", "Name", "Condition (formula)", "Action")


class StopConditionsDialog(QDialog):
    def __init__(self, scene, conditions: list[dict], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stop Conditions")
        self.resize(860, 420)
        self.scene = scene
        self.result_conditions: list[dict] | None = None
        lay = QVBoxLayout(self)
        note = QLabel("A stop condition is a Boolean formula checked after every simulation step, for example "
                      "<code>object(\"OBJ-000001\").position.z &lt; 0[m]</code> or "
                      "<code>norm(object(\"OBJ-000002\").velocity) &gt; 20[m/s] &amp;&amp; t &gt; 1[s]</code>. "
                      "When it <i>becomes</i> true the run pauses or ends, and the event is recorded. Units and "
                      "object references are checked before the run starts.")
        note.setWordWrap(True)
        note.setObjectName("Muted")
        lay.addWidget(note)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(1, 160)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, slot in (("Add", self.add_row), ("Remove", self._remove), ("Check", self.check)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        for c in conditions:
            self.add_row(c)

    # ---- rows ------------------------------------------------------------------------------------------
    def add_row(self, cond: dict | None = None) -> None:
        if not isinstance(cond, dict):
            names = {self._text(r, 1).lower() for r in range(self.table.rowCount())}
            k = 1
            while f"condition {k}" in names:
                k += 1
            cond = {"name": f"Condition {k}", "expression": "t > 1[s]", "action": "pause", "enabled": True}
        r = self.table.rowCount()
        self.table.insertRow(r)
        on = QTableWidgetItem()
        on.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        on.setCheckState(Qt.CheckState.Checked if cond.get("enabled", True) else Qt.CheckState.Unchecked)
        self.table.setItem(r, 0, on)
        self.table.setItem(r, 1, QTableWidgetItem(str(cond.get("name", ""))))
        self.table.setItem(r, 2, QTableWidgetItem(str(cond.get("expression", ""))))
        combo = QComboBox()
        combo.addItems(list(ACTIONS))
        combo.setCurrentText(str(cond.get("action", "pause")))
        self.table.setCellWidget(r, 3, combo)

    def _remove(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _text(self, r: int, c: int) -> str:
        item = self.table.item(r, c)
        return item.text().strip() if item is not None else ""

    def conditions(self) -> list[dict]:
        return [{"name": self._text(r, 1), "expression": self._text(r, 2),
                 "action": self.table.cellWidget(r, 3).currentText(),
                 "enabled": self.table.item(r, 0).checkState() == Qt.CheckState.Checked}
                for r in range(self.table.rowCount())]

    # ---- checking ----------------------------------------------------------------------------------------
    def check(self) -> bool:
        """Validate names/actions and type-check every enabled condition against the scene."""
        try:
            conds = validate_conditions(self.conditions())
        except ConditionError as exc:
            self._show(False, str(exc))
            return False
        plan = SimpleNamespace(diagnostics=[], conditions=[])
        attach_conditions(plan, self.scene, SimpleNamespace(stop_conditions=conds))
        if plan.diagnostics:
            self._show(False, "\n".join(d.message for d in plan.diagnostics))
            return False
        n = sum(c["enabled"] for c in conds)
        self._show(True, f"{n} enabled condition{'s' if n != 1 else ''} checked: units and references are valid.")
        return True

    def _show(self, ok: bool, text: str) -> None:
        self.status.setStyleSheet("" if ok else "color: #e5534b;")
        self.status.setText(text)

    def _accept(self) -> None:
        if self.check():
            self.result_conditions = validate_conditions(self.conditions())
            self.accept()
