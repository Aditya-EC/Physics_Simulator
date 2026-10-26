"""Simulation clock and real-time update-rate controller (Phase 3 building blocks).

These are deterministic, GUI-independent components with an injectable
time source so their behaviour is unit tested. The Phase 3 engine will run
them on a worker thread and evaluate processes inside `step` callbacks.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .settings import DurationPolicy, SimulationSettings


class SimulationClock:
    """Simulated time advanced in fixed Δt increments.

    Time is computed from the step count within each constant-Δt segment
    (t = t_segment_start + k·Δt) rather than by repeated addition, so no
    floating-point drift accumulates over long runs. Changing Δt starts a
    new segment; subsequent steps use the new value.
    """

    def __init__(self, delta_time: float) -> None:
        self._dt = float(delta_time)
        self._segment_start = 0.0
        self._segment_steps = 0
        self.steps = 0
        self._partial_time = 0.0

    @property
    def delta_time(self) -> float:
        return self._dt

    @property
    def time(self) -> float:
        return self._segment_start + self._segment_steps * self._dt + self._partial_time

    def set_delta_time(self, dt: float) -> None:
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError("Δt must be positive and finite")
        self._segment_start = self.time
        self._segment_steps = 0
        self._partial_time = 0.0
        self._dt = float(dt)

    def next_step_dt(self, duration: float | None, policy: DurationPolicy) -> float | None:
        """Δt to use for the next step, or None if the duration boundary has been reached."""
        if duration is None:
            return self._dt
        remaining = duration - self.time
        tol = 1e-9 * max(1.0, abs(duration))
        if remaining <= tol:
            return None
        if remaining >= self._dt - tol:
            return self._dt
        if policy == DurationPolicy.FINAL_PARTIAL_STEP:
            return remaining
        if policy == DurationPolicy.NEXT_FULL_STEP:
            return self._dt
        return None  # LAST_FULL_STEP

    def advance(self, dt: float | None = None) -> None:
        if dt is None or abs(dt - self._dt) <= 1e-15 * self._dt:
            self._segment_steps += 1
        else:
            self._partial_time += dt  # only used for a final partial step
        self.steps += 1

    def export(self) -> dict:
        """Exact clock state (segment start, steps in the segment, partial time, total steps, Δt)."""
        return {"segment_start": self._segment_start, "segment_steps": self._segment_steps,
                "partial_time": self._partial_time, "steps": self.steps, "delta_time": self._dt}

    def restore(self, d: dict) -> None:
        dt = float(d["delta_time"])
        values = (float(d["segment_start"]), int(d["segment_steps"]), float(d["partial_time"]), int(d["steps"]))
        if not (math.isfinite(dt) and dt > 0 and all(math.isfinite(v) for v in values) and values[1] >= 0
                and values[3] >= 0):
            raise ValueError("Invalid clock state")
        self._segment_start, self._segment_steps, self._partial_time, self.steps = values
        self._dt = dt

    def reset(self) -> None:
        self._segment_start = 0.0
        self._segment_steps = 0
        self._partial_time = 0.0
        self.steps = 0


@dataclass
class RateStats:
    requested_hz: float
    achieved_hz: float
    behind_steps: int
    dropped_steps: int
    overrun: bool


class UpdateRateController:
    """Decides how many steps are due at each scheduler tick for real-time pacing.

    Uses a monotonic clock. When execution falls behind, up to
    `max_catch_up_steps` extra steps are run per tick; any further backlog is
    dropped from the *schedule* (never by enlarging Δt) and reported as an
    overrun so the user knows the requested rate is not being achieved.
    """

    def __init__(self, rate_hz: float, max_catch_up_steps: int = 10,
                 now: Callable[[], float] = time.perf_counter, window_s: float = 1.0) -> None:
        self._now = now
        self.max_catch_up_steps = max(1, int(max_catch_up_steps))
        self._window = window_s
        self._history: deque[float] = deque()
        self.dropped_steps = 0
        self.behind_steps = 0
        self.set_rate(rate_hz)
        self.restart()

    def set_rate(self, rate_hz: float) -> None:
        if not (math.isfinite(rate_hz) and rate_hz > 0):
            raise ValueError("Update rate must be positive")
        self.rate_hz = float(rate_hz)
        self._period = 1.0 / self.rate_hz
        if hasattr(self, "_next_due"):
            self._next_due = self._now() + self._period

    def restart(self) -> None:
        self._next_due = self._now() + self._period
        self._history.clear()
        self.dropped_steps = 0
        self.behind_steps = 0

    def steps_due(self) -> int:
        now = self._now()
        if now < self._next_due:
            return 0
        due = int((now - self._next_due) / self._period) + 1
        allowed = min(due, self.max_catch_up_steps)
        self.behind_steps = due - allowed
        if self.behind_steps:
            self.dropped_steps += self.behind_steps
        self._next_due += due * self._period
        return allowed

    def seconds_until_next(self) -> float:
        return max(0.0, self._next_due - self._now())

    def record_step(self) -> None:
        t = self._now()
        self._history.append(t)
        while self._history and t - self._history[0] > self._window:
            self._history.popleft()

    def achieved_rate(self) -> float:
        if len(self._history) < 2:
            return 0.0
        span = self._history[-1] - self._history[0]
        return (len(self._history) - 1) / span if span > 0 else 0.0

    def stats(self) -> RateStats:
        achieved = self.achieved_rate()
        return RateStats(self.rate_hz, achieved, self.behind_steps, self.dropped_steps,
                         overrun=self.dropped_steps > 0 or (achieved > 0 and achieved < 0.95 * self.rate_hz))


def run_steps(settings: SimulationSettings, step: Callable[[float, float], None]) -> SimulationClock:
    """Run a complete simulation as fast as possible (no pacing).

    `step(t, dt)` is called once per simulation step with the time at the start
    of the step and the Δt to apply. Returns the final clock.
    """
    settings.check()
    clock = SimulationClock(settings.delta_time)
    while True:
        dt = clock.next_step_dt(settings.duration, settings.duration_policy)
        if dt is None:
            break
        step(clock.time, dt)
        clock.advance(dt)
    return clock
