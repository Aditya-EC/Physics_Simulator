"""Processes table panel and simulation toolbar widgets."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..processes.process_base import ProcessError
from ..processes.process_registry import process_registry
from ..scene.editing import SceneEditor
from ..scene.scene_manager import Scene
from ..simulation.runner import RunStats
from . import icons, theme
from .widgets.inputs import NumberField


def _fmt_param(v) -> str:
    if isinstance(v, list):
        return "(" + ", ".join(f"{c:g}" for c in v) + ")"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


class ProcessesPanel(QWidget):
    """All process assignments in evaluation order, with enable toggles."""

    addRequested = Signal()
    editRequested = Signal(str)
    validateRequested = Signal()
    dependenciesRequested = Signal()
    documentationRequested = Signal()

    HEADERS = ["On", "ID", "Process", "Category", "Targets", "Parameters", "Label"]

    def __init__(self, scene: Scene, editor: SceneEditor, parent=None) -> None:
        super().__init__(parent)
        self.scene, self.editor = scene, editor
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        buttons = [("plus", "Assign…", self.addRequested.emit), ("edit", "Edit…", self._edit),
                   ("trash", "Remove", self._remove), ("arrow_up", "Up", lambda: self._move(-1)),
                   ("arrow_down", "Down", lambda: self._move(1)), ("check", "Validate", self.validateRequested.emit),
                   ("tree", "Dependencies…", self.dependenciesRequested.emit),
                   ("help", "Documentation…", self.documentationRequested.emit)]
        self.buttons = {}
        for ic, text, slot in buttons:
            b = QPushButton(icons.icon(ic), text)
            b.setAutoDefault(False)
            b.clicked.connect(slot)
            bar.addWidget(b)
            self.buttons[text] = b
        bar.addStretch(1)
        self.summary = QLabel()
        self.summary.setObjectName("Muted")
        bar.addWidget(self.summary)
        lay.addLayout(bar)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(lambda r, c: c != 0 and self.editRequested.emit(self._uid(r)))
        self.table.itemSelectionChanged.connect(self._update_buttons)
        lay.addWidget(self.table)
        scene.processesChanged.connect(self.rebuild)
        scene.sceneReset.connect(self.rebuild)
        scene.objectsAdded.connect(lambda _ids: self.rebuild())
        scene.objectsRemoved.connect(lambda _ids: self.rebuild())
        self.rebuild()

    def _uid(self, row: int) -> str:
        return self.table.item(row, 1).text()

    def selected_uids(self) -> list[str]:
        return [self._uid(r) for r in sorted({i.row() for i in self.table.selectedIndexes()})]

    def rebuild(self) -> None:
        reg = process_registry()
        sel = set(self.selected_uids())
        rows = self.scene.assignments
        self.table.setRowCount(len(rows))
        for r, a in enumerate(rows):
            try:
                p = reg.get(a.process_id)
                name, cat = p.name, p.category
            except ProcessError:
                name, cat = f"{a.process_id} (unavailable)", "—"
            cb = QCheckBox()
            cb.setChecked(a.enabled)
            cb.setToolTip("Enable or disable this process")
            cb.toggled.connect(lambda on, uid=a.uid: self.editor.update_assignment(uid, enabled=on))
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(6, 0, 0, 0)
            hl.addWidget(cb)
            self.table.setCellWidget(r, 0, holder)
            n_targets = sum(1 for o in self.scene.objects.values() if a.targets.matches(o))
            values = [a.uid, name, cat, f"{a.targets.describe()}  [{n_targets} object(s)]",
                      ", ".join(f"{k} = {_fmt_param(v)}" for k, v in a.params.items()) or "—", a.label]
            for c, v in enumerate(values, start=1):
                it = QTableWidgetItem(v)
                it.setToolTip(v)
                if not a.enabled:
                    it.setForeground(self.palette().placeholderText())
                self.table.setItem(r, c, it)
            if a.uid in sel:
                self.table.selectRow(r)
        self.table.resizeColumnsToContents()
        on = sum(a.enabled for a in rows)
        self.summary.setText(f"{len(rows)} assignment(s), {on} enabled · evaluated in this order")
        self._update_buttons()

    def _update_buttons(self) -> None:
        n = len(self.selected_uids())
        for k in ("Edit…", "Up", "Down"):
            self.buttons[k].setEnabled(n == 1)
        self.buttons["Remove"].setEnabled(n > 0)

    def _edit(self) -> None:
        uids = self.selected_uids()
        if len(uids) == 1:
            self.editRequested.emit(uids[0])

    def _remove(self) -> None:
        uids = self.selected_uids()
        if uids:
            self.editor.remove_assignments(uids)

    def _move(self, delta: int) -> None:
        uids = self.selected_uids()
        if len(uids) == 1:
            self.editor.move_assignment(uids[0], delta)
            row = next(i for i, a in enumerate(self.scene.assignments) if a.uid == uids[0])
            self.table.selectRow(row)


class SimulationFields(QWidget):
    """Δt / rate / duration editors and the live status read-out for the simulation toolbar."""

    deltaTimeChanged = Signal(float)
    updateRateChanged = Signal(float)
    durationChanged = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(4)

        def field(label: str, tip: str, signal, lo: float, hi: float, prec: int) -> NumberField:
            lab = QLabel(label)
            lab.setObjectName("Muted")
            lab.setToolTip(tip)
            f = NumberField(precision=prec, minimum=lo, maximum=hi)
            f.setToolTip(tip)
            f.setFixedWidth(74)
            f.committed.connect(signal.emit)
            lay.addWidget(lab)
            lay.addWidget(f)
            return f

        self.dt = field("Δt [s]", "Fixed simulation time step: simulated time added by every step", self.deltaTimeChanged,
                        1e-9, 10.0, 6)
        self.rate = field("Rate [Hz]", "Requested simulation update rate: steps per real second",
                          self.updateRateChanged, 0.1, 100000.0, 2)
        self.duration = field("Duration [s]", "Total simulated time", self.durationChanged, 1e-9, 1e12, 4)
        self.status = QLabel()
        self.status.setMinimumWidth(360)
        self.status.setStyleSheet("font-family: 'Cascadia Mono', 'Consolas', 'DejaVu Sans Mono', monospace;")
        lay.addSpacing(8)
        lay.addWidget(self.status)
        self.show_idle()

    def set_values(self, dt: float, rate: float, duration: float) -> None:
        self.dt.set_si(dt)
        self.rate.set_si(rate)
        self.duration.set_si(duration)

    def show_idle(self) -> None:
        self.status.setText("t = 0 s · idle")
        self.status.setStyleSheet(self.status.styleSheet().split("color")[0])

    def show_stats(self, st: RunStats) -> None:
        if st.pacing == "afap":
            rate = f"{st.achieved_hz:,.0f} steps/s (max speed)"
        else:
            rate = f"{st.achieved_hz:,.1f}/{st.requested_hz:g} Hz"
        text = (f"t = {st.time:.4f} s · {st.steps:,} steps · {rate} · ×{st.sim_speed:.3g} · {st.state}")
        if st.overrun:
            text += f" · ⚠ behind schedule ({st.dropped_steps:,} dropped)"
        self.status.setText(text)
        base = self.status.styleSheet().split("color")[0]
        if st.overrun:
            self.status.setStyleSheet(base + f"color: {theme.current().warning};")
        elif st.state == "error":
            self.status.setStyleSheet(base + f"color: {theme.current().danger};")
        else:
            self.status.setStyleSheet(base)
        self.status.setToolTip(
            f"Simulated time: {st.time:.9g} s of {st.duration:g} s\nSteps completed: {st.steps}\n"
            f"Δt: {st.delta_time:g} s\nRequested rate: {st.requested_hz:g} Hz\nAchieved rate: {st.achieved_hz:.2f} Hz\n"
            f"Simulation speed: {st.sim_speed:.4g} simulated s per real s\nWall-clock time running: {st.wall_time:.2f} s\n"
            f"Dropped (late) steps: {st.dropped_steps}\nMean step cost: {st.step_ms:.3f} ms")

