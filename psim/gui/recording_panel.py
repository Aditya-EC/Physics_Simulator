"""Recording panel: recording settings, status, statistics / object comparison, event log, exports and report."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..coordinates.scene_kinematics import reference_choices
from ..recording.analysis import STAT_COLUMNS, clean_row, statistics
from ..recording.recording import Recording
from ..recording.variables import QUANTITIES, Evaluator, Variable, VariableError, info, recorded_objects
from ..scene.scene_manager import Scene
from ..units.unit_registry import UnitRegistry

STAT_QUANTITIES = [q for q in QUANTITIES if q not in ("time", "step", "mass")]


def _table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.verticalHeader().setDefaultSectionSize(t.fontMetrics().height() + 6)
    return t


class RecordingPanel(QWidget):
    exportCsvRequested = Signal()
    exportJsonRequested = Signal()
    reportRequested = Signal()
    clearRequested = Signal()
    intervalChanged = Signal(int)

    def __init__(self, scene: Scene, units: UnitRegistry, recording_getter, settings_getter, parent=None) -> None:
        super().__init__(parent)
        self.scene, self.units = scene, units
        self._recording, self._settings = recording_getter, settings_getter
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        cfg = QHBoxLayout()
        self.enabled = QCheckBox("Record")
        self.enabled.setToolTip("Record sampled data during simulation runs (plots, trajectories, statistics)")
        self.every = QSpinBox()
        self.every.setRange(1, 1_000_000)
        self.every.setPrefix("every ")
        self.every.setSuffix(" steps")
        self.every.setToolTip("Sampling interval in simulation steps, independent of Δt and the render rate. "
                              "It can be changed while running.")
        self.memory = QSpinBox()
        self.memory.setRange(1, 65536)
        self.memory.setSuffix(" MB")
        self.memory.setToolTip("Memory limit of a recording (applies to the next run)")
        self.overflow = QComboBox()
        for key, text in (("decimate", "when full: halve resolution"), ("window", "when full: keep recent"),
                          ("stop", "when full: stop recording")):
            self.overflow.addItem(text, key)
        self.process_outputs = QCheckBox("Process outputs")
        self.process_outputs.setToolTip("Also record the contribution of every force/acceleration process "
                                        "(one extra evaluation per sample)")
        for w in (self.enabled, self.every, self.memory, self.overflow, self.process_outputs):
            cfg.addWidget(w)
        cfg.addStretch(1)
        for text, sig in (("Export CSV…", self.exportCsvRequested), ("Export JSON…", self.exportJsonRequested),
                          ("Report…", self.reportRequested), ("Clear", self.clearRequested)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(sig.emit)
            cfg.addWidget(b)
        lay.addLayout(cfg)
        self.status = QLabel()
        self.status.setObjectName("Muted")
        lay.addWidget(self.status)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        stats = QWidget()
        sl = QVBoxLayout(stats)
        sl.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        self.quantity = QComboBox()
        for q in STAT_QUANTITIES:
            self.quantity.addItem(QUANTITIES[q].label, q)
        self.component = QComboBox()
        for c, text in (("mag", "magnitude"), ("x", "x"), ("y", "y"), ("z", "z")):
            self.component.addItem(text, c)
        self.frame = QComboBox()
        row.addWidget(QLabel("Compare"))
        row.addWidget(self.quantity)
        row.addWidget(self.component)
        row.addWidget(self.frame)
        row.addStretch(1)
        sl.addLayout(row)
        self.stats = _table(["Object", "Unit"] + STAT_COLUMNS + ["t at min", "t at max"])
        self.stats.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        sl.addWidget(self.stats)
        self.tabs.addTab(stats, "Statistics && Comparison")
        self.events = _table(["t [s]", "Step", "Event", "Message"])
        self.events.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.events, "Event Log")

        self._load_settings()
        self.enabled.toggled.connect(self._store)
        self.every.valueChanged.connect(self._interval)
        self.memory.valueChanged.connect(self._store)
        self.overflow.currentIndexChanged.connect(self._store)
        self.process_outputs.toggled.connect(self._store)
        for combo in (self.quantity, self.component, self.frame):
            combo.currentIndexChanged.connect(lambda _i: self.refresh(force=True))
        self.quantity.currentIndexChanged.connect(self._quantity_changed)
        self._version = None
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self._quantity_changed()

    # ---- settings ------------------------------------------------------------------------------------------
    def _load_settings(self) -> None:
        s = self._settings()
        for w in (self.enabled, self.every, self.memory, self.overflow, self.process_outputs):
            w.blockSignals(True)
        self.enabled.setChecked(s.record_enabled)
        self.every.setValue(s.record_every_n_steps)
        self.memory.setValue(int(s.record_max_mb))
        self.overflow.setCurrentIndex(max(self.overflow.findData(s.record_overflow), 0))
        self.process_outputs.setChecked(s.record_process_outputs)
        for w in (self.enabled, self.every, self.memory, self.overflow, self.process_outputs):
            w.blockSignals(False)

    def reload_settings(self) -> None:
        self._load_settings()

    def _store(self, *_a) -> None:
        s = self._settings()
        s.record_enabled = self.enabled.isChecked()
        s.record_max_mb = float(self.memory.value())
        s.record_overflow = self.overflow.currentData()
        s.record_process_outputs = self.process_outputs.isChecked()

    def _interval(self, value: int) -> None:
        self.intervalChanged.emit(int(value))

    def _quantity_changed(self, *_a) -> None:
        q = QUANTITIES[self.quantity.currentData()]
        self.component.setEnabled(q.vector)
        self.frame.setEnabled(q.frame)
        cur = self.frame.currentData()
        self.frame.blockSignals(True)
        self.frame.clear()
        for text, ref in reference_choices(self.scene):
            self.frame.addItem(f"relative to {text}" if ref else "global", ref)
        self.frame.setCurrentIndex(max(self.frame.findData(cur), 0))
        self.frame.blockSignals(False)

    # ---- data ------------------------------------------------------------------------------------------------
    def refresh(self, force: bool = False) -> None:
        rec: Recording | None = self._recording()
        version = (id(rec), rec.version if rec is not None else None)
        if not force and (version == self._version or not self.isVisible()):
            return
        self._version = version
        if rec is None:
            self.status.setText("No recording. Recording starts with the next simulation run"
                                if self.enabled.isChecked() else "Recording is off")
            self.stats.setRowCount(0)
            self.events.setRowCount(0)
            return
        self.status.setText(rec.status_text())
        self._fill_stats(rec)
        evs = rec.event_list()
        self.events.setRowCount(len(evs))
        for r, e in enumerate(evs):
            for c, v in enumerate((f"{e.time:.6g}", str(e.step), e.kind, e.message)):
                self.events.setItem(r, c, QTableWidgetItem(v))

    def _fill_stats(self, rec: Recording) -> None:
        q = self.quantity.currentData()
        comp = self.component.currentData() if self.component.isEnabled() else ""
        frame = self.frame.currentData() if self.frame.isEnabled() else None
        objs = recorded_objects(rec)
        self.stats.setRowCount(len(objs))
        if not rec.n:
            return
        ev = Evaluator(rec, self.scene)
        t = ev.values(Variable.time())
        names = {**rec.meta.get("names", {}), **{o.id: o.name for o in self.scene.objects.values()}}
        for r, oid in enumerate(objs):
            var = Variable(oid, q, comp, frame)
            qty = info(var).unit_quantity
            unit = self.units.preferred_unit(qty) if qty else ""
            try:
                y = ev.values(var)
                if unit:
                    y = np.asarray(self.units.from_si(y, unit), float)
                st = statistics(t, y)
            except VariableError:
                st = None
            row = [names.get(oid, oid), unit]
            row += [f"{x:.6g}" if math.isfinite(x) else "—" for x in (clean_row(st.as_row()) if st else [math.nan] * 7)]
            row += [f"{st.t_min:.6g}", f"{st.t_max:.6g}"] if st else ["—", "—"]
            for c, v in enumerate(row):
                self.stats.setItem(r, c, QTableWidgetItem(v))
