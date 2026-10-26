"""Simulation session management between the scene/GUI and the background runner.

Session policy
* Start validates the settings and the process plan, captures the initial
  state of every simulated object, and launches the runner.
* While a session is active (running, paused, finished or failed), object
  edits are applied immediately and forwarded to the engine at the next step
  boundary, but they are not recorded for undo, and structural edits (adding,
  deleting, regrouping) are blocked by the main window.
* Stop ends the session and records the whole run as one undoable command
  ("Simulation run"), so undo returns to the initial conditions.
* Reset ends the session (if any) and restores the initial state captured at Start.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import QObject, Signal

from ..processes.process_scheduler import ExecutionPlan, build_plan
from ..properties.property_system import PropertyError
from ..scene.editing import SceneEditor, StateCommand
from ..scene.scene_manager import Scene
from ..simulation.runner import RunState, RunStats, SimulationRunner
from ..simulation.settings import PacingMode, SimulationSettings
from ..simulation.simulation_engine import SimulationEngine, SimulationError, Snapshot
from ..simulation.state import capture_bodies
from ..recording.recording import Recording
from ..simulation.world import EngineWorld, capture_frames, capture_static, global_values


class SimulationController(QObject):
    stateChanged = Signal(str)        # idle | running | paused | finished | error
    statsUpdated = Signal(object)     # RunStats
    snapshotApplied = Signal(float)   # simulated time
    info = Signal(str)
    warning = Signal(str)
    error = Signal(str)
    recordingChanged = Signal()       # a new recording was started or the recording was cleared

    def __init__(self, scene: Scene, editor: SceneEditor, settings_getter, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.editor = editor
        self._settings = settings_getter
        self.runner = SimulationRunner(self)
        self.runner.snapshotReady.connect(self._apply_snapshot)
        self.runner.statsUpdated.connect(self._on_stats)
        self.runner.stateChanged.connect(self._on_runner_state)
        self.runner.errorOccurred.connect(self._on_runner_error)
        self.runner.message.connect(self.info.emit)
        self._initial: dict | None = None        # object states captured at Start
        self._last_initial: dict | None = None   # kept after Stop for Reset
        self._body_ids: list[str] = []
        self._applying = False
        self._reported: set[str] = set()
        self._state = "idle"
        self.last_stats: RunStats | None = None
        self.recording: Recording | None = None   # data of the current or last run (kept after Stop)
        self.warning.connect(self._log_warning)
        self.error.connect(self._log_error)
        self.info.connect(self._log_info)
        scene.objectsChanged.connect(self._on_objects_changed)
        scene.processesChanged.connect(self._on_processes_changed)
        scene.globalsChanged.connect(self._on_processes_changed)
        scene.framesChanged.connect(self._on_processes_changed)

    # ---- state -------------------------------------------------------------------------------------
    @property
    def settings(self) -> SimulationSettings:
        return self._settings()

    @property
    def state(self) -> str:
        return self._state

    @property
    def session_active(self) -> bool:
        return self._initial is not None

    @property
    def can_reset(self) -> bool:
        return self.session_active or bool(self._last_initial)

    @property
    def body_ids(self) -> list[str]:
        return list(self._body_ids) if self.session_active else []

    def _set_state(self, st: str) -> None:
        if st != self._state:
            prev, self._state = self._state, st
            kind = {"running": "resume" if prev == "paused" else "start", "paused": "pause",
                    "finished": "finished", "idle": "stop", "error": "error"}.get(st)
            if kind:
                self._log(kind, f"{prev} → {st}")
            self.stateChanged.emit(st)

    # ---- data recording ------------------------------------------------------------------------------------
    def _log(self, kind: str, message: str) -> None:
        rec = self.recording
        if rec is None:
            return
        eng = self.runner.engine
        t, step = (eng.clock.time, eng.clock.steps) if eng is not None else (0.0, 0)
        rec.log(t, step, kind, message)

    def _log_warning(self, message: str) -> None:
        if self.session_active:
            self._log("warning", message)

    def _log_error(self, message: str) -> None:
        if self.session_active:
            self._log("error", message)

    def _log_info(self, message: str) -> None:
        if self.session_active and message.startswith(("Applied", "Breakpoint")):
            self._log("change" if message.startswith("Applied") else "breakpoint", message)

    def set_record_interval(self, steps: int) -> None:
        self.settings.record_every_n_steps = max(1, int(steps))
        if self.recording is not None and self.session_active:
            self.recording.interval_steps = self.settings.record_every_n_steps
            self._log("change", f"Recording interval set to {self.settings.record_every_n_steps} steps")

    def clear_recording(self) -> None:
        if self.session_active:
            self.warning.emit("The recording of a running simulation cannot be cleared; stop it first")
            return
        self.recording = None
        self.recordingChanged.emit()

    def validate(self) -> ExecutionPlan:
        return build_plan(self.scene, self.scene.assignments)

    # ---- commands --------------------------------------------------------------------------------------
    def start(self, paused: bool = False) -> bool:
        """Start a new session, or resume the current one."""
        if self.session_active:
            if self._state in ("paused", "finished"):
                eng = self.runner.engine
                if eng is not None and eng.finished:
                    self.warning.emit("The simulation reached its duration. Reset, or increase the duration to "
                                      "continue.")
                    return False
                self.runner.resume()
            return True
        errors = self.settings.validate()
        if errors:
            self.error.emit("Invalid simulation settings: " + "; ".join(errors))
            return False
        plan = self.validate()
        for d in plan.diagnostics:
            (self.error if d.level == "error" else self.warning).emit(d.message)
        if not plan.ok:
            return False
        if not plan.body_ids:
            self.error.emit("Nothing to simulate: assign at least one enabled process to an object "
                            "(Processes ▸ Assign Process).")
            return False
        bodies, state = capture_bodies(self.scene, plan.body_ids)
        statics = capture_static(self.scene)
        world = EngineWorld(statics, plan.body_ids, global_values(self.scene), self.settings.delta_time,
                            capture_frames(self.scene, statics))
        try:
            engine = SimulationEngine(SimulationSettings.from_dict(self.settings.to_dict()), plan, bodies, state, world)
        except SimulationError as exc:
            self.error.emit(str(exc))
            return False
        recording_on = self.settings.record_enabled
        if recording_on:
            rec = Recording(int(self.settings.record_max_mb * 2 ** 20), self.settings.record_overflow,
                            self.settings.record_every_n_steps)
            rec.meta["settings"] = self.settings.to_dict()
            rec.meta["names"] = {o.id: o.name for o in self.scene.objects.values()}
            rec.meta["types"] = {o.id: o.type_id for o in self.scene.objects.values()}
            engine.attach_recording(rec, self.settings.record_process_outputs)
            self.recording = rec
        self._initial = self.scene.snapshot(plan.body_ids)
        self._reported = set()
        self._body_ids = list(plan.body_ids)
        self.editor.recording = False
        self.scene.runtime = {oid: {} for oid in plan.body_ids}  # accelerations unknown until the first snapshot
        self.scene.simulating = True
        self.runner.start(engine, paused=paused)
        if recording_on:
            self.recordingChanged.emit()
        self._set_state("paused" if paused else "running")
        s = self.settings
        mode = "as fast as possible" if s.pacing == PacingMode.AS_FAST_AS_POSSIBLE else f"{s.update_rate_hz:g} Hz"
        self.info.emit(f"Simulation started: {len(plan.body_ids)} bodies, {len(plan.contributions) + len(plan.prescribed)}"
                       f" processes, Δt = {s.delta_time:g} s, {mode}, duration {s.duration:g} s, "
                       f"{s.integrator.replace('_', ' ')}")
        return True

    def pause(self) -> None:
        if self.session_active:
            self.runner.pause()

    def step(self) -> None:
        if not self.session_active:
            if self.start(paused=True):
                self.runner.step(1)
        elif self._state in ("paused", "finished"):
            self.runner.step(1)

    def stop(self, record: bool = True) -> None:
        """End the session, keeping the current state; the run becomes one undoable command."""
        if not self.session_active:
            return
        self.runner.stop()
        if self.runner.engine is not None:
            if self.runner.engine.sampler is not None:   # the worker has stopped: record the end state
                self.runner.engine.sampler.final()
            self._apply_snapshot(self.runner.engine.snapshot())
        before = self._initial
        t = self.runner.engine.clock.time if self.runner.engine else 0.0
        self._last_initial = before  # before ending the session, so listeners see that Reset is possible
        self._end_session()
        after = self.scene.snapshot(before.keys())
        if record and after != before:
            sel = self.scene.selection
            self.editor.stack.push(StateCommand(self.scene, before, after, f"Simulation run (t = {t:.6g} s)", sel, sel))
        self.info.emit(f"Simulation stopped at t = {t:.6g} s")

    def reset(self) -> None:
        """Restore the state captured when the last run started."""
        if self.session_active:
            initial = self._initial
            self.runner.stop()
            self._last_initial = initial
            self._end_session()
            self.scene.apply_states({k: v for k, v in initial.items() if k in self.scene.objects})
            self.info.emit("Simulation reset to the initial conditions")
        elif self._last_initial:
            initial = {k: v for k, v in self._last_initial.items() if k in self.scene.objects}
            self.editor.edit(initial.keys(), lambda: self.scene.apply_states(initial), "Reset Simulation")
            self.info.emit("Simulation reset to the initial conditions of the last run")

    def abandon(self) -> None:
        """End any session without recording (used when the project is closed)."""
        self._last_initial = None
        if self.session_active:
            self.runner.stop()
            self._end_session()

    def _end_session(self) -> None:
        self._initial = None
        self._body_ids = []
        self.scene.runtime = {}
        self.scene.simulating = False
        self.editor.recording = True
        self._set_state("idle")
        self.scene.notify_changed(list(self.scene.objects))

    # ---- live settings ---------------------------------------------------------------------------------
    def set_delta_time(self, dt: float) -> None:
        self.settings.delta_time = dt
        if self.session_active:
            self.runner.set_delta_time(dt)

    def set_update_rate(self, hz: float) -> None:
        self.settings.update_rate_hz = hz
        if self.session_active:
            self.runner.set_update_rate(hz)

    def set_duration(self, duration: float) -> None:
        self.settings.duration = duration
        if self.session_active:
            self.runner.set_duration(duration)

    def set_pacing(self, pacing: PacingMode) -> None:
        self.settings.pacing = PacingMode(pacing)
        if self.session_active:
            self.runner.set_pacing(pacing)

    def set_render_fps(self, fps: float) -> None:
        self.settings.render_fps = fps
        if self.session_active:
            self.runner.set_render_fps(fps)

    # ---- runner callbacks (GUI thread) ---------------------------------------------------------------------
    def _apply_snapshot(self, snap: Snapshot) -> None:
        if not self.session_active:
            return
        self._applying = True
        try:
            changed = []
            for i, oid in enumerate(snap.ids):
                obj = self.scene.find(oid)
                if obj is None:
                    continue
                self.scene.set_world_pose(oid, snap.origin[i], snap.q[i])
                obj.linear_velocity = [float(c) for c in snap.v[i]]
                obj.angular_velocity = [float(c) for c in snap.w[i]]
                self.scene.runtime[oid] = {"acceleration": snap.acc[i].copy(), "angular_acceleration": snap.alpha[i].copy(),
                                           "time": snap.time}
                changed.append(oid)
            for oid, values in (snap.custom or {}).items():
                obj = self.scene.find(oid)
                if obj is None:
                    continue
                for key, value in values.items():
                    prop = obj.custom.get(key)
                    if prop is None:
                        continue
                    try:
                        prop.set_value(value.tolist() if hasattr(value, "tolist") else value)
                    except PropertyError as exc:
                        msg = f"{oid}.custom.{key}: simulated value rejected ({exc})"
                        if msg not in self._reported:
                            self._reported.add(msg)
                            self.warning.emit(msg)
                if oid not in changed:
                    changed.append(oid)
            for msg in snap.warnings or []:
                if msg not in self._reported and len(self._reported) < 200:
                    self._reported.add(msg)
                    self.warning.emit(msg)
            self.scene.notify_changed(changed)
        finally:
            self._applying = False
        self.snapshotApplied.emit(snap.time)

    def _on_stats(self, stats: RunStats) -> None:
        self.last_stats = stats
        self.statsUpdated.emit(stats)

    def _on_runner_state(self, st: str) -> None:
        if self.session_active and st != RunState.IDLE.value:
            self._set_state(st)

    def _on_runner_error(self, msg: str) -> None:
        self.error.emit(msg)

    # ---- edits during a session ---------------------------------------------------------------------------------
    def _on_objects_changed(self, ids) -> None:
        if not self.session_active or self._applying:
            return
        edited = set(ids) & set(self._body_ids)
        if edited:
            self._rebuild(edited)

    def _on_processes_changed(self) -> None:
        if self.session_active:
            self._rebuild(set())

    def _rebuild(self, edited: set[str]) -> None:
        plan = self.validate()
        if not plan.ok:
            for d in plan.errors:
                self.error.emit(f"Change not applied to the running simulation: {d.message}")
            return
        new_ids = [i for i in plan.body_ids if i not in self._initial]
        for oid in new_ids:  # objects that became simulated mid-run: remember their initial state too
            self._initial[oid] = copy.deepcopy(self.scene.get(oid).to_dict())
        bodies, state = capture_bodies(self.scene, plan.body_ids)
        keep = set(plan.body_ids) - edited - set(new_ids)
        self._body_ids = list(plan.body_ids)
        statics = capture_static(self.scene)
        self.runner.replace_plan(plan, bodies, state, keep, statics, global_values(self.scene),
                                 capture_frames(self.scene, statics))
        if edited:
            self.info.emit(f"Applied edits of {', '.join(sorted(edited))} to the running simulation")

