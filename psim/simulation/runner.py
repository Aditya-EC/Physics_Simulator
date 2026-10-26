"""Background execution of a SimulationEngine with its own timing scheduler.

The runner owns a worker thread. The GUI never touches the engine directly:
it sends commands through a queue (applied between steps, so a step never
sees a half-applied change) and receives copies of the state (snapshots)
through Qt signals, at most once per render interval. Rendering therefore
never determines Δt or how many steps are executed.

Real-time mode targets `update_rate_hz` steps per wall-clock second with a
monotonic clock. If the engine falls behind, at most `max_catch_up_steps`
extra steps run per tick; the remaining backlog is dropped from the schedule
and reported as an overrun. Δt is never changed to compensate.
As-fast-as-possible mode runs steps back to back (Δt and duration still apply).
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QObject, Signal

from .clock import UpdateRateController
from .settings import PacingMode
from .simulation_engine import SimulationEngine


class RunState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class RunStats:
    state: str
    time: float
    steps: int
    delta_time: float
    duration: float
    pacing: str
    requested_hz: float
    achieved_hz: float
    sim_speed: float            # simulated seconds per wall-clock second (recent window)
    wall_time: float            # wall-clock time spent running (pauses excluded)
    dropped_steps: int          # steps removed from the real-time schedule because the engine fell behind
    overrun: bool               # requested rate currently not achieved
    step_ms: float              # mean compute time per step (recent)


class _WindowsTimerResolution:
    """Request 1 ms scheduler granularity on Windows while a real-time run is active."""

    def __enter__(self):
        self._ok = False
        if sys.platform == "win32":
            try:
                import ctypes
                self._ok = ctypes.windll.winmm.timeBeginPeriod(1) == 0
            except (AttributeError, OSError):
                pass
        return self

    def __exit__(self, *exc):
        if self._ok:
            import ctypes
            ctypes.windll.winmm.timeEndPeriod(1)


class SimulationRunner(QObject):
    snapshotReady = Signal(object)    # Snapshot
    statsUpdated = Signal(object)     # RunStats
    stateChanged = Signal(str)        # RunState value
    errorOccurred = Signal(str)
    message = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._commands: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._state = RunState.IDLE
        self.engine: SimulationEngine | None = None
        self.render_fps = 60.0

    # ---- public API (GUI thread) -----------------------------------------------------------------
    @property
    def state(self) -> RunState:
        return self._state

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, engine: SimulationEngine, paused: bool = False) -> None:
        if self.active:
            raise RuntimeError("A simulation is already running")
        self.engine = engine
        self.render_fps = engine.settings.render_fps
        self._commands = queue.Queue()
        self._thread = threading.Thread(target=self._run, args=(paused,), name="psim-simulation", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._commands.put(("pause",))

    def resume(self) -> None:
        self._commands.put(("resume",))

    def step(self, n: int = 1) -> None:
        self._commands.put(("step", int(n)))

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the worker and wait for it (the engine keeps its final state)."""
        if self.active:
            self._commands.put(("stop",))
            self._thread.join(timeout)
        self._thread = None
        self._set_state(RunState.IDLE)

    def set_delta_time(self, dt: float) -> None:
        self._commands.put(("dt", float(dt)))

    def set_update_rate(self, hz: float) -> None:
        self._commands.put(("rate", float(hz)))

    def set_duration(self, duration: float) -> None:
        self._commands.put(("duration", float(duration)))

    def set_pacing(self, pacing: PacingMode) -> None:
        self._commands.put(("pacing", PacingMode(pacing)))

    def set_render_fps(self, fps: float) -> None:
        self._commands.put(("fps", float(fps)))

    def replace_plan(self, plan, bodies, state, keep: set[str], statics=None, globals_=None, frames=None) -> None:
        self._commands.put(("plan", plan, bodies, state, keep, statics, globals_, frames))

    def wait_until(self, predicate, timeout: float = 10.0) -> bool:
        """Test helper: poll `predicate()` (no Qt event processing)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            time.sleep(0.002)
        return predicate()

    # ---- worker thread ---------------------------------------------------------------------------------
    def _set_state(self, st: RunState) -> None:
        if st != self._state:
            self._state = st
            self.stateChanged.emit(st.value)

    def _run(self, start_paused: bool) -> None:
        eng = self.engine
        s = eng.settings
        ctl = UpdateRateController(s.update_rate_hz, s.max_catch_up_steps)
        pending_steps = 0
        last_publish = 0.0
        wall_running = 0.0
        last_tick = time.perf_counter()
        speed_window: deque[tuple[float, float]] = deque()
        step_costs: deque[float] = deque(maxlen=200)
        drops_window: deque[tuple[float, int]] = deque()
        self._set_state(RunState.PAUSED if start_paused else RunState.RUNNING)
        stop = False
        last_overrun = False
        dropped_base = 0  # drops from earlier scheduling segments (the controller restarts on resume / rate change)

        def restart_schedule() -> None:
            nonlocal dropped_base
            dropped_base += ctl.dropped_steps
            ctl.restart()

        def dropped_total() -> int:
            return dropped_base + ctl.dropped_steps

        def publish(force: bool = False) -> None:
            nonlocal last_publish, last_overrun
            now = time.perf_counter()
            if not force and now - last_publish < 1.0 / max(self.render_fps, 1.0):
                return
            last_publish = now
            self.snapshotReady.emit(eng.snapshot())
            speed_window.append((wall_running, eng.clock.time))
            while len(speed_window) > 2 and speed_window[-1][0] - speed_window[0][0] > 1.0:
                speed_window.popleft()
            drops_window.append((now, dropped_total()))
            while len(drops_window) > 2 and now - drops_window[0][0] > 1.0:
                drops_window.popleft()
            dw = speed_window[-1][0] - speed_window[0][0]
            speed = (speed_window[-1][1] - speed_window[0][1]) / dw if dw > 0 else 0.0
            achieved = ctl.achieved_rate()
            recent_drops = drops_window[-1][1] - drops_window[0][1]
            realtime = s.pacing == PacingMode.REAL_TIME
            if self._state == RunState.RUNNING:  # while paused, keep reporting the last running assessment
                last_overrun = realtime and wall_running > 0.5 and (
                    recent_drops > 0 or achieved < 0.9 * s.update_rate_hz)
            overrun = last_overrun
            self.statsUpdated.emit(RunStats(
                self._state.value, eng.clock.time, eng.clock.steps, eng.clock.delta_time, s.duration, s.pacing.value,
                s.update_rate_hz, achieved, speed, wall_running, dropped_total(), overrun,
                1000 * sum(step_costs) / len(step_costs) if step_costs else 0.0))

        def do_step() -> bool:
            t0 = eng.clock.time
            c0 = time.perf_counter()
            try:
                advanced = eng.step()
            except Exception as exc:  # numerical failure or process error: stop cleanly with a report
                self._set_state(RunState.ERROR)
                self.errorOccurred.emit(str(exc))
                publish(force=True)
                return False
            step_costs.append(time.perf_counter() - c0)
            if not advanced:
                return False
            ctl.record_step()
            for bp in s.breakpoints:
                if t0 < bp <= eng.clock.time + 1e-12:
                    self._set_state(RunState.PAUSED)
                    self.message.emit(f"Breakpoint reached at t = {eng.clock.time:.6g} s")
                    publish(force=True)
                    return False
            if eng.finished:
                self._set_state(RunState.FINISHED)
                self.message.emit(f"Simulation finished at t = {eng.clock.time:.6g} s ({eng.clock.steps} steps)")
                publish(force=True)
                return False
            return True

        def handle(cmd) -> None:
            nonlocal stop, pending_steps
            kind = cmd[0]
            try:
                if kind == "stop":
                    stop = True
                elif kind == "pause" and self._state == RunState.RUNNING:
                    self._set_state(RunState.PAUSED)
                    publish(force=True)
                elif kind == "resume" and self._state in (RunState.PAUSED, RunState.FINISHED) and not eng.finished:
                    restart_schedule()  # no catch-up burst for the time spent paused
                    self._set_state(RunState.RUNNING)
                elif kind == "step" and self._state in (RunState.PAUSED, RunState.FINISHED):
                    pending_steps += cmd[1]
                elif kind == "dt":
                    eng.set_delta_time(cmd[1])
                elif kind == "duration":
                    eng.set_duration(cmd[1])
                    if not eng.finished and self._state == RunState.FINISHED:
                        self._set_state(RunState.PAUSED)
                elif kind == "rate":
                    s.update_rate_hz = cmd[1]
                    s.check()
                    ctl.set_rate(cmd[1])
                    restart_schedule()
                elif kind == "pacing":
                    s.pacing = cmd[1]
                    restart_schedule()
                elif kind == "fps":
                    self.render_fps = cmd[1]
                elif kind == "plan":
                    eng.replace_plan(*cmd[1:])
                    publish(force=True)
            except Exception as exc:  # invalid live change: report and keep the previous configuration
                self.errorOccurred.emit(f"Change not applied: {exc}")

        with _WindowsTimerResolution():
            publish(force=True)
            while not stop:
                running = self._state == RunState.RUNNING
                if running and s.pacing == PacingMode.REAL_TIME:
                    timeout = min(ctl.seconds_until_next(), 0.02)
                elif running or pending_steps:
                    timeout = 0.0
                else:
                    timeout = 0.05
                try:
                    cmd = self._commands.get(timeout=timeout) if timeout > 0 else self._commands.get_nowait()
                    handle(cmd)
                    while True:
                        handle(self._commands.get_nowait())
                except queue.Empty:
                    pass
                if stop:
                    break
                now = time.perf_counter()
                if self._state == RunState.RUNNING:
                    wall_running += now - last_tick
                last_tick = now

                if self._state == RunState.RUNNING:
                    if s.pacing == PacingMode.AS_FAST_AS_POSSIBLE:
                        budget_end = time.perf_counter() + 0.01
                        while self._state == RunState.RUNNING and time.perf_counter() < budget_end:
                            if not do_step():
                                break
                    else:
                        for _ in range(ctl.steps_due()):
                            if not do_step():
                                break
                    publish()
                elif pending_steps:
                    while pending_steps > 0:
                        pending_steps -= 1
                        if not do_step():
                            pending_steps = 0
                    publish(force=True)
            publish(force=True)
