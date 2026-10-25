"""Simulation settings: fixed Δt, update rate, duration, pacing, integrator, render rate, breakpoints."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
                               QSpinBox, QVBoxLayout)

from ...simulation.integrators import INTEGRATORS
from ...simulation.settings import DurationPolicy, PacingMode, SimulationSettings
from ..widgets.inputs import NumberField

POLICY_TEXT = {
    DurationPolicy.FINAL_PARTIAL_STEP: "Final partial step (stop exactly at the duration)",
    DurationPolicy.NEXT_FULL_STEP: "Next full step (may overshoot by < Δt)",
    DurationPolicy.LAST_FULL_STEP: "Last full step (may stop short by < Δt)",
}


class SimulationSettingsDialog(QDialog):
    def __init__(self, settings: SimulationSettings, running: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Simulation Settings")
        self.setMinimumWidth(520)
        self.base = settings
        lay = QVBoxLayout(self)
        intro = QLabel("Δt is the simulated time added by every step. The update rate is how many steps run per "
                       "real second. They are independent: simulated time advances at rate × Δt seconds per real "
                       "second when the hardware keeps up. Rendering never changes either value.")
        intro.setWordWrap(True)
        intro.setObjectName("Muted")
        lay.addWidget(intro)
        form = QFormLayout()
        self.dt = NumberField(precision=9, minimum=settings.min_delta_time, maximum=settings.max_delta_time)
        self.dt.set_si(settings.delta_time)
        self.rate = NumberField(precision=4, minimum=settings.min_update_rate_hz, maximum=settings.max_update_rate_hz)
        self.rate.set_si(settings.update_rate_hz)
        self.duration = NumberField(precision=6, minimum=1e-9, maximum=1e12)
        self.duration.set_si(settings.duration)
        self.policy = QComboBox()
        for p, text in POLICY_TEXT.items():
            self.policy.addItem(text, p.value)
        self.policy.setCurrentIndex(self.policy.findData(settings.duration_policy.value))
        self.afap = QCheckBox("Run as fast as possible (scientific calculation mode)")
        self.afap.setChecked(settings.pacing == PacingMode.AS_FAST_AS_POSSIBLE)
        self.catch_up = QSpinBox()
        self.catch_up.setRange(1, 100000)
        self.catch_up.setValue(settings.max_catch_up_steps)
        self.catch_up.setToolTip("Maximum extra steps per scheduler tick when real-time execution falls behind. "
                                 "Further backlog is dropped from the schedule and reported (Δt is never changed).")
        self.integrator = QComboBox()
        for key, text in INTEGRATORS.items():
            self.integrator.addItem(text, key)
        self.integrator.setCurrentIndex(self.integrator.findData(settings.integrator))
        self.fps = NumberField(precision=1, minimum=1, maximum=480)
        self.fps.set_si(settings.render_fps)
        self.breakpoints = QLineEdit(", ".join(f"{b:g}" for b in settings.breakpoints))
        self.breakpoints.setPlaceholderText("e.g. 1.5, 3")
        self.breakpoints.setToolTip("Simulated times (s) at which the run pauses automatically")
        form.addRow("Fixed time step Δt [s]", self.dt)
        form.addRow("Update rate [steps/s]", self.rate)
        form.addRow("Duration (simulated) [s]", self.duration)
        form.addRow("Duration boundary", self.policy)
        form.addRow("", self.afap)
        form.addRow("Max catch-up steps per tick", self.catch_up)
        form.addRow("Integrator", self.integrator)
        form.addRow("Render rate [FPS]", self.fps)
        form.addRow("Breakpoints [s]", self.breakpoints)
        lay.addLayout(form)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        if running:
            note = QLabel("A simulation is active: Δt, update rate, duration, pacing and render rate apply at the next "
                          "step; the integrator, catch-up limit and breakpoints apply to the next run.")
            note.setWordWrap(True)
            note.setObjectName("Muted")
            lay.addWidget(note)
        self.error = QLabel()
        self.error.setStyleSheet("color: #e5534b;")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        for w in (self.dt, self.rate, self.duration):
            w.committed.connect(lambda _v: self._update_summary())
        self.policy.currentIndexChanged.connect(self._update_summary)
        self.afap.toggled.connect(self._update_summary)
        self.result_settings: SimulationSettings | None = None
        self._update_summary()

    def _collect(self) -> SimulationSettings:
        s = SimulationSettings.from_dict(self.base.to_dict())
        s.delta_time, s.update_rate_hz, s.duration = self.dt.si(), self.rate.si(), self.duration.si()
        s.duration_policy = DurationPolicy(self.policy.currentData())
        s.pacing = PacingMode.AS_FAST_AS_POSSIBLE if self.afap.isChecked() else PacingMode.REAL_TIME
        s.max_catch_up_steps = self.catch_up.value()
        s.integrator = self.integrator.currentData()
        s.render_fps = self.fps.si()
        text = self.breakpoints.text().replace(";", ",")
        s.breakpoints = sorted(float(t) for t in text.split(",") if t.strip())
        return s

    def _update_summary(self) -> None:
        try:
            s = self._collect()
        except ValueError:
            return
        if s.pacing == PacingMode.AS_FAST_AS_POSSIBLE:
            pace = "Steps run back to back (no real-time pacing)."
        else:
            pace = (f"Target: {s.update_rate_hz:g} steps per real second → {s.time_ratio:g} simulated s per real s "
                    f"(the duration takes ≈ {s.duration / max(s.time_ratio, 1e-300):.4g} s of real time).")
        try:
            steps = f"{s.expected_steps():,}"
        except (ValueError, ZeroDivisionError, OverflowError):
            steps = "—"
        self.summary.setText(f"<b>{steps}</b> steps to reach {s.duration:g} s. {pace}")

    def _accept(self) -> None:
        try:
            s = self._collect()
        except ValueError:
            self.error.setText("Breakpoints must be numbers separated by commas")
            return
        errors = s.validate()
        if errors:
            self.error.setText("; ".join(errors))
            return
        self.result_settings = s
        self.accept()
