"""Simulation timing configuration (interface for the Phase 3 engine).

Definitions (see requirements §9):

* ``delta_time`` (Δt) – simulated-time increment applied by every step and
  used by process formulas. Independent of rendering and of the update rate.
* ``update_rate_hz`` – number of complete simulation steps the real-time
  scheduler targets per second of wall-clock time. Δt is NEVER derived from it.
* ``duration`` – total simulated time to compute.
* ``render_fps`` – viewport redraw rate; it never influences Δt or step count.

Simulated-time progression in real-time mode is ``update_rate_hz * delta_time``
simulated seconds per wall-clock second (when the hardware keeps up).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum


class PacingMode(str, Enum):
    REAL_TIME = "real_time"          # paced at update_rate_hz using a monotonic clock
    AS_FAST_AS_POSSIBLE = "afap"     # no pacing; Δt and duration still apply


class DurationPolicy(str, Enum):
    """How to stop when duration is not an integer multiple of Δt."""

    FINAL_PARTIAL_STEP = "final_partial_step"  # last step uses the remaining time (< Δt) so t_end == duration
    NEXT_FULL_STEP = "next_full_step"          # ceil(duration/Δt) full steps; may overshoot by < Δt
    LAST_FULL_STEP = "last_full_step"          # floor(duration/Δt) full steps; may stop short by < Δt


class SettingsError(ValueError):
    pass


@dataclass
class SimulationSettings:
    delta_time: float = 0.01
    update_rate_hz: float = 100.0
    duration: float = 10.0
    render_fps: float = 60.0
    pacing: PacingMode = PacingMode.REAL_TIME
    duration_policy: DurationPolicy = DurationPolicy.FINAL_PARTIAL_STEP
    max_catch_up_steps: int = 10      # per scheduler tick when behind schedule
    record_every_n_steps: int = 1     # data recording interval, independent of Δt and the render rate
    record_enabled: bool = True
    record_max_mb: float = 256.0      # memory limit of the recording
    record_overflow: str = "decimate"  # decimate | window | stop (see psim.recording.recording)
    record_process_outputs: bool = True
    integrator: str = "semi_implicit_euler"
    breakpoints: list = field(default_factory=list)  # simulated times at which the run pauses
    min_delta_time: float = 1e-9
    max_delta_time: float = 10.0
    min_update_rate_hz: float = 0.1
    max_update_rate_hz: float = 100_000.0

    def validate(self) -> list[str]:
        errors = []
        if not (math.isfinite(self.delta_time) and self.min_delta_time <= self.delta_time <= self.max_delta_time):
            errors.append(f"Δt must be between {self.min_delta_time:g} s and {self.max_delta_time:g} s")
        if not (math.isfinite(self.update_rate_hz) and
                self.min_update_rate_hz <= self.update_rate_hz <= self.max_update_rate_hz):
            errors.append(f"Update rate must be between {self.min_update_rate_hz:g} Hz and {self.max_update_rate_hz:g} Hz")
        if not (math.isfinite(self.duration) and self.duration > 0):
            errors.append("Duration must be a positive number of simulated seconds")
        if not (math.isfinite(self.render_fps) and 1 <= self.render_fps <= 480):
            errors.append("Render rate must be between 1 and 480 FPS")
        if self.max_catch_up_steps < 1:
            errors.append("Maximum catch-up steps must be at least 1")
        if self.record_every_n_steps < 1:
            errors.append("Recording interval must be at least 1 step")
        if not (math.isfinite(self.record_max_mb) and 1 <= self.record_max_mb <= 65536):
            errors.append("Recording memory limit must be between 1 MB and 64 GB")
        if self.record_overflow not in ("decimate", "window", "stop"):
            errors.append("Recording overflow policy must be decimate, window or stop")
        from .integrators import INTEGRATORS
        if self.integrator not in INTEGRATORS:
            errors.append(f"Unknown integrator '{self.integrator}'")
        if any(not (isinstance(b, (int, float)) and math.isfinite(b) and b > 0) for b in self.breakpoints):
            errors.append("Breakpoints must be positive simulated times")
        return errors

    def check(self) -> None:
        errors = self.validate()
        if errors:
            raise SettingsError("; ".join(errors))

    # ---- derived quantities -----------------------------------------------------------
    @property
    def time_ratio(self) -> float:
        """Simulated seconds per wall-clock second in real-time mode (if the rate is achieved)."""
        return self.update_rate_hz * self.delta_time

    def expected_steps(self) -> int:
        """Number of steps needed to reach `duration` under the duration policy."""
        exact = self.duration / self.delta_time
        nearest = round(exact)
        if abs(exact - nearest) <= 1e-9 * max(1.0, exact):
            return int(nearest)
        if self.duration_policy == DurationPolicy.LAST_FULL_STEP:
            return int(math.floor(exact))
        return int(math.ceil(exact))  # FINAL_PARTIAL_STEP and NEXT_FULL_STEP

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pacing"] = self.pacing.value
        d["duration_policy"] = self.duration_policy.value
        return d

    @staticmethod
    def from_dict(data: dict) -> "SimulationSettings":
        s = SimulationSettings()
        for k, v in data.items():
            if not hasattr(s, k):
                continue
            try:
                if k == "pacing":
                    v = PacingMode(v)
                elif k == "duration_policy":
                    v = DurationPolicy(v)
                elif k == "integrator":
                    v = str(v)
                elif k == "breakpoints":
                    v = sorted(float(b) for b in v)
                elif isinstance(getattr(s, k), bool):
                    v = bool(v)
                elif isinstance(getattr(s, k), str):
                    v = str(v)
                elif isinstance(getattr(s, k), int):
                    v = int(v)
                else:
                    v = float(v)
                setattr(s, k, v)
            except (TypeError, ValueError):
                continue
        return s
