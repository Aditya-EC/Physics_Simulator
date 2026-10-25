"""Bottom panels: console, errors & warnings, object table, performance monitor."""
from __future__ import annotations

import csv
import datetime as _dt
import logging
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import numerics
from ..rendering.renderer import FrameStats
from ..scene.scene_manager import Scene
from ..units.unit_registry import UnitRegistry
from . import icons, theme


class LogBus(QObject):
    """Application-wide message bus feeding the console and the problems list."""

    message = Signal(str, str, str)  # level, text, source

    def info(self, text: str, source: str = "") -> None:
        self.message.emit("info", text, source)

    def warning(self, text: str, source: str = "") -> None:
        self.message.emit("warning", text, source)

    def error(self, text: str, source: str = "") -> None:
        self.message.emit("error", text, source)


class _LogHandler(logging.Handler):
    def __init__(self, bus: LogBus) -> None:
        super().__init__(logging.INFO)
        self.bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        level = "error" if record.levelno >= logging.ERROR else ("warning" if record.levelno >= logging.WARNING else "info")
        self.bus.message.emit(level, record.getMessage(), record.name)


def install_logging(bus: LogBus) -> None:
    logging.getLogger("psim").addHandler(_LogHandler(bus))
    logging.getLogger("psim").setLevel(logging.INFO)


def _now() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S")


class ConsolePanel(QWidget):
    def __init__(self, bus: LogBus, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        bar.addStretch(1)
        clear = QPushButton(icons.icon("trash"), "Clear")
        bar.addWidget(clear)
        lay.addLayout(bar)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(5000)
        self.text.setStyleSheet("font-family: 'Cascadia Mono', 'Consolas', 'DejaVu Sans Mono', monospace;")
        lay.addWidget(self.text)
        clear.clicked.connect(self.text.clear)
        bus.message.connect(self.append)

    def append(self, level: str, text: str, source: str = "") -> None:
        t = theme.current()
        color = {"info": t.text_muted, "warning": t.warning, "error": t.danger}.get(level, t.text)
        src = f"[{source}] " if source else ""
        esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.text.appendHtml(f'<span style="color:{t.text_disabled}">{_now()}</span> '
                             f'<span style="color:{color}">{level.upper():7s}</span> {src}{esc}')


class ProblemsPanel(QWidget):
    countChanged = Signal(int, int)

    def __init__(self, bus: LogBus, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        self.summary = QLabel("No problems")
        self.summary.setObjectName("Muted")
        bar.addWidget(self.summary, 1)
        clear = QPushButton(icons.icon("trash"), "Clear")
        clear.clicked.connect(self.clear)
        bar.addWidget(clear)
        lay.addLayout(bar)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Message", "Source", "Time"])
        self.tree.setRootIsDecorated(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.tree)
        bus.message.connect(self.add)
        self.errors = self.warnings = 0

    def add(self, level: str, text: str, source: str = "") -> None:
        if level not in ("warning", "error"):
            return
        it = QTreeWidgetItem([text, source, _now()])
        it.setIcon(0, icons.icon("error" if level == "error" else "warning",
                                 theme.current().danger if level == "error" else theme.current().warning))
        it.setToolTip(0, text)
        self.tree.insertTopLevelItem(0, it)
        if level == "error":
            self.errors += 1
        else:
            self.warnings += 1
        self._update()

    def clear(self) -> None:
        self.tree.clear()
        self.errors = self.warnings = 0
        self._update()

    def _update(self) -> None:
        self.summary.setText(f"{self.errors} error(s), {self.warnings} warning(s)" if self.errors or self.warnings
                             else "No problems")
        self.countChanged.emit(self.errors, self.warnings)


class ObjectTablePanel(QWidget):
    frameRequested = Signal(str)

    HEADERS = ["ID", "Name", "Type", "Parent", "Position X", "Position Y", "Position Z", "Mass", "Tags", "Visible",
               "Locked"]

    def __init__(self, scene: Scene, units: UnitRegistry, parent=None) -> None:
        super().__init__(parent)
        self.scene, self.units = scene, units
        self._syncing = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        bar.addWidget(self.filter, 1)
        export = QPushButton(icons.icon("export"), "Export CSV…")
        export.clicked.connect(self.export_csv)
        bar.addWidget(export)
        lay.addLayout(bar)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._table_selection)
        self.table.cellDoubleClicked.connect(lambda r, c: self.frameRequested.emit(self.table.item(r, 0).text()))
        lay.addWidget(self.table)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.rebuild)
        for sig in (scene.objectsAdded, scene.objectsRemoved):
            sig.connect(lambda _ids: self._timer.start(0))
        # throttle (not debounce): continuous changes such as a running simulation must still refresh
        scene.objectsChanged.connect(lambda _ids: self._timer.isActive() or self._timer.start(250))
        scene.sceneReset.connect(lambda: self._timer.start(0))
        scene.hierarchyChanged.connect(lambda: self._timer.start(0))
        scene.selectionChanged.connect(self._scene_selection)
        self.rebuild()

    def rows(self) -> list[list[str]]:
        lu = self.units.preferred_unit("length")
        mu = self.units.preferred_unit("mass")
        out = []
        for o in self.scene.objects.values():
            p, _ = self.scene.world_pose(o.id)
            p = self.units.from_si(np.asarray(p), lu)
            m = o.mass() if o.typedef.physical else None
            out.append([o.id, o.name, o.typedef.name, o.parent_id or "", f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}",
                        "" if m is None else f"{self.units.from_si(m, mu):.6g}", ", ".join(o.tags),
                        "yes" if o.visible else "no", "yes" if o.locked else "no"])
        return out

    def rebuild(self) -> None:
        self._syncing = True
        lu = self.units.preferred_unit("length")
        mu = self.units.preferred_unit("mass")
        self.table.setHorizontalHeaderLabels(self.HEADERS[:4] + [f"Position X [{lu}]", f"Position Y [{lu}]",
                                                                  f"Position Z [{lu}]", f"Mass [{mu}]"] + self.HEADERS[8:])
        self.table.setSortingEnabled(False)
        rows = self.rows()
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem()
                if 4 <= c <= 7 and val:
                    item.setData(Qt.ItemDataRole.DisplayRole, float(val))
                else:
                    item.setText(val)
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self._syncing = False
        self._scene_selection(self.scene.selection)
        self._apply_filter(self.filter.text())

    def _apply_filter(self, text: str) -> None:
        text = text.lower()
        for r in range(self.table.rowCount()):
            hay = " ".join(self.table.item(r, c).text() for c in (0, 1, 2, 8)).lower()
            self.table.setRowHidden(r, bool(text) and text not in hay)

    def _scene_selection(self, ids) -> None:
        self._syncing = True
        self.table.clearSelection()
        sel = set(ids)
        mode = self.table.selectionMode()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).text() in sel:
                self.table.selectRow(r)
        self.table.setSelectionMode(mode)
        self._syncing = False

    def _table_selection(self) -> None:
        if self._syncing:
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        self.scene.set_selection([self.table.item(r, 0).text() for r in rows])

    def export_csv(self, path: str | None = None) -> str | None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export Object Table", "objects.csv", "CSV files (*.csv)")
            if not path:
                return None
        headers = [self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(headers)
            w.writerows(self.rows())
        return path


class _Sparkline(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.values: deque[float] = deque(maxlen=240)
        self.setMinimumHeight(56)
        self.setMinimumWidth(160)

    def push(self, v: float) -> None:
        self.values.append(v)
        self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = theme.current()
        r = self.rect().adjusted(1, 1, -1, -1)
        p.setPen(QPen(QColor(t.border), 1))
        p.setBrush(QColor(t.input))
        p.drawRoundedRect(r, 4, 4)
        if len(self.values) < 2:
            p.end()
            return
        vals = list(self.values)
        top = max(max(vals) * 1.15, 1.0)
        path = QPainterPath()
        n = self.values.maxlen
        for i, v in enumerate(vals):
            x = r.left() + (n - len(vals) + i) / (n - 1) * r.width()
            y = r.bottom() - v / top * (r.height() - 4)
            (path.lineTo if i else path.moveTo)(QPointF(x, y))
        p.setPen(QPen(QColor(t.accent), 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setPen(QColor(t.text_muted))
        p.drawText(r.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
                   f"frame CPU time (ms), max {max(vals):.1f}")
        p.end()


class PerformancePanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(3)
        self.labels: dict[str, QLabel] = {}
        entries = (("fps", "Render rate"), ("cpu", "Frame CPU time"), ("draws", "Draw calls"), ("tris", "Triangles"),
                   ("objects", "Objects drawn"), ("shadow", "Shadow pass"), ("backend", "Numerical core"),
                   ("gl", "OpenGL"), ("sim_state", "Simulation"), ("sim_rate", "Update rate (actual / requested)"),
                   ("sim_speed", "Simulated s per real s"), ("sim_cost", "Step cost / dropped steps"))
        for k, (key, label) in enumerate(entries):
            name = QLabel(label)
            name.setObjectName("Muted")
            lab = QLabel("—")
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row, col = (k % 4, (k // 4) * 2)  # three label/value column pairs
            grid.addWidget(name, row, col)
            grid.addWidget(lab, row, col + 1)
            self.labels[key] = lab
        grid.setColumnStretch(5, 1)
        lay.addLayout(grid, 3)
        self.spark = _Sparkline()
        lay.addWidget(self.spark, 2)
        self.labels["backend"].setText(numerics.backend_description())

    def update_stats(self, stats: FrameStats, fps: float, continuous: bool) -> None:
        self.labels["fps"].setText(f"{fps:.1f} FPS (continuous)" if continuous else "on demand (redraw on change)")
        self.labels["cpu"].setText(f"{stats.cpu_ms:.2f} ms")
        self.labels["draws"].setText(str(stats.draw_calls))
        self.labels["tris"].setText(f"{stats.triangles:,}")
        self.labels["objects"].setText(str(stats.objects_drawn))
        self.labels["shadow"].setText("on" if stats.shadow else "off")
        self.spark.push(stats.cpu_ms)

    def update_sim_stats(self, st) -> None:
        self.labels["sim_state"].setText(f"{st.state} · t = {st.time:.6g} s · {st.steps:,} steps · Δt = {st.delta_time:g} s")
        if st.pacing == "afap":
            self.labels["sim_rate"].setText(f"{st.achieved_hz:,.0f} steps/s (as fast as possible)")
        else:
            warn = "  ⚠ requested rate not achieved" if st.overrun else ""
            self.labels["sim_rate"].setText(f"{st.achieved_hz:,.2f} / {st.requested_hz:g} Hz{warn}")
        self.labels["sim_speed"].setText(f"{st.sim_speed:.4g}")
        self.labels["sim_cost"].setText(f"{st.step_ms:.3f} ms · {st.dropped_steps:,} dropped")

    def set_gl_info(self, info: dict) -> None:
        self.labels["gl"].setText(f"{info.get('renderer', '?')} · {info.get('version', '?')}")
