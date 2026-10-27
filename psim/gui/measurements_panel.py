"""Measurements panel: live distances and angles between objects (origins, centres of mass or surface points)."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..scene.editing import SceneEditor
from ..scene.measurements import MeasurementError, describe_point, evaluate
from ..scene.scene_manager import Scene, SceneError
from ..units.unit_registry import UnitRegistry

HINT = ("Select two objects for a distance or three for an angle (the vertex second). Values follow the objects "
        "live, also while a simulation runs.")
POINT_HINT = ("Press Distance or Angle, then click the points on object surfaces in the viewport (for an angle: A, "
              "the vertex, B). Each point stays fixed on its object. Esc cancels.")
COLUMNS = ("Show", "Name", "Points", "Value", "ΔX", "ΔY", "ΔZ")


class MeasurementsPanel(QWidget):
    errorOccurred = Signal(str)
    pickRequested = Signal(str)        # kind: pick surface points in the viewport

    def __init__(self, scene: Scene, editor: SceneEditor, units: UnitRegistry, parent=None) -> None:
        super().__init__(parent)
        self.scene, self.editor, self.units = scene, editor, units
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        self.at = QComboBox()
        self.at.addItem("Object origins", "origin")
        self.at.addItem("Centres of mass", "com")
        self.at.addItem("Points on surfaces", "point")
        self.at.currentIndexChanged.connect(lambda _i: self._update_buttons())
        self.at.setToolTip("Which point of each selected object new measurements use")
        self.add_distance = QPushButton("Distance")
        self.add_distance.setToolTip("Distance between the two selected objects (in selection order)")
        self.add_distance.clicked.connect(lambda: self._add("distance"))
        self.add_angle = QPushButton("Angle")
        self.add_angle.setToolTip("Angle at the second of three selected objects (selection order A, vertex, B)")
        self.add_angle.clicked.connect(lambda: self._add("angle"))
        self.remove = QPushButton("Remove")
        self.remove.clicked.connect(self._remove)
        copy_btn = QPushButton("Copy")
        copy_btn.setToolTip("Copy the table as tab-separated text")
        copy_btn.clicked.connect(self.copy)
        for w in (QLabel("Measure"), self.at, self.add_distance, self.add_angle, self.remove):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(copy_btn)
        lay.addLayout(bar)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._item_changed)
        lay.addWidget(self.table, 1)
        self.hint = QLabel(HINT)
        self._showing_error = False
        self.hint.setObjectName("Muted")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(50)    # coalesce the many change signals of a running simulation
        self._timer.timeout.connect(self.refresh)
        self._loading = False
        scene.measurementsChanged.connect(self.rebuild)
        scene.sceneReset.connect(self.rebuild)
        scene.objectsChanged.connect(lambda _ids: self._timer.start())
        scene.objectsRemoved.connect(lambda _ids: self._timer.start())
        scene.objectsAdded.connect(lambda _ids: self._timer.start())
        scene.hierarchyChanged.connect(self._timer.start)
        scene.selectionChanged.connect(lambda _ids: self._update_buttons())
        self.rebuild()

    # ---- building ------------------------------------------------------------------------------------------
    def rebuild(self) -> None:
        self._loading = True
        try:
            self.table.setRowCount(len(self.scene.measurements))
            for r, m in enumerate(self.scene.measurements):
                show = QTableWidgetItem()
                show.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                              | Qt.ItemFlag.ItemIsSelectable)
                show.setCheckState(Qt.CheckState.Checked if m.visible else Qt.CheckState.Unchecked)
                show.setData(Qt.ItemDataRole.UserRole, m.id)
                self.table.setItem(r, 0, show)
                self.table.setItem(r, 1, QTableWidgetItem(m.name))
                for c in range(2, len(COLUMNS)):
                    item = QTableWidgetItem("")
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    self.table.setItem(r, c, item)
        finally:
            self._loading = False
        self.refresh()
        self._update_buttons()

    def refresh(self) -> None:
        """Recompute every value at the current object poses."""
        lu, au = self.units.preferred_unit("length"), self.units.preferred_unit("angle")
        self._loading = True
        try:
            for r, m in enumerate(self.scene.measurements):
                if r >= self.table.rowCount():
                    break
                names = []
                for p in m.points:
                    names.append(describe_point(self.scene, p))
                sep = " – " if m.kind == "distance" else " ∠ "
                self.table.item(r, 2).setText(sep.join(names) if m.kind == "distance" else
                                              f"{names[0]} – {names[1]} – {names[2]}  (at {names[1]})")
                try:
                    val = evaluate(self.scene, m)
                except SceneError:
                    self.table.item(r, 3).setText("missing object")
                    for c in (4, 5, 6):
                        self.table.item(r, c).setText("")
                    continue
                if m.kind == "distance":
                    self.table.item(r, 3).setText(self._fmt(val.value, lu))
                    for c, comp in zip((4, 5, 6), val.delta):
                        self.table.item(r, c).setText(self._fmt(float(comp), lu))
                else:
                    text = "undefined (point at the vertex)" if math.isnan(val.value) else self._fmt(val.value, au)
                    self.table.item(r, 3).setText(text)
                    for c in (4, 5, 6):
                        self.table.item(r, c).setText("")
        finally:
            self._loading = False

    def _fmt(self, si: float, unit: str) -> str:
        return f"{float(self.units.from_si(si, unit)):.6g} {unit}"

    def _update_buttons(self) -> None:
        n = len([i for i in self.scene.selection if i in self.scene.objects])
        points = self.at.currentData() == "point"
        self.add_distance.setEnabled(points or n == 2)
        self.add_angle.setEnabled(points or n == 3)
        if not self._showing_error:
            self.hint.setText(POINT_HINT if points else HINT)
        self.remove.setEnabled(bool(self.scene.measurements))

    # ---- editing ---------------------------------------------------------------------------------------------
    def _add(self, kind: str) -> None:
        if self.at.currentData() == "point":
            self.pickRequested.emit(kind)
            return
        ids = [i for i in self.scene.selection if i in self.scene.objects]
        try:
            self.editor.add_measurement(kind, ids, self.at.currentData())
        except (MeasurementError, SceneError) as exc:
            self._message(str(exc))

    def _remove(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        ids = [self.scene.measurements[r].id for r in rows if r < len(self.scene.measurements)]
        if ids:
            self.editor.remove_measurements(ids)

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() not in (0, 1):
            return
        data = self.scene.measurements_snapshot()
        r = item.row()
        if r >= len(data):
            return
        if item.column() == 0:
            data[r]["visible"] = item.checkState() == Qt.CheckState.Checked
        else:
            data[r]["name"] = item.text().strip()
        try:
            self.editor.set_measurements(data, f"Edit Measurement '{data[r]['name']}'")
        except (MeasurementError, SceneError) as exc:
            self.errorOccurred.emit(str(exc))
            self._message(str(exc))
            self.rebuild()

    def _message(self, text: str) -> None:
        """Show a problem in the panel itself (no modal dialog while the user is working in the table)."""
        self._showing_error = True
        self.hint.setStyleSheet("color: #e5534b;")
        self.hint.setText(text)
        QTimer.singleShot(6000, self._reset_hint)

    def _reset_hint(self) -> None:
        self._showing_error = False
        self.hint.setStyleSheet("")
        self._update_buttons()

    def copy(self) -> None:
        rows = ["\t".join(COLUMNS[1:])]
        for r in range(self.table.rowCount()):
            rows.append("\t".join(self.table.item(r, c).text() for c in range(1, len(COLUMNS))))
        QGuiApplication.clipboard().setText("\n".join(rows))
