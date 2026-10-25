"""Deterministic fixed-step simulation engine (no threading, no Qt).

One step (requirements §9.4):
  1. read the current state (shared previous state for every process);
  2. use the enabled processes of the validated execution plan;
  3. references were resolved to body indices when the plan was built;
  4-5. the integrator evaluates contribution processes (summed) and prescribed
     processes at each solver stage;
  6. the new state replaces the old one atomically (no partial updates);
  7. the clock advances by Δt (or the final partial step, per policy);
  8-9. the step counter increments; the caller records / publishes.

Identical inputs (scene, assignments, settings) give bit-identical results:
evaluation order is fixed by the assignment list, and wall-clock timing never
enters the numerics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..processes.process_base import StepContext
from ..processes.process_scheduler import ExecutionPlan
from .clock import SimulationClock
from .integrators import integrate
from .settings import SimulationSettings
from .state import BodySet, SimState, origins


class SimulationError(Exception):
    pass


@dataclass
class Snapshot:
    """Copy of the engine state handed to the GUI thread."""

    ids: list[str]
    origin: np.ndarray        # (N, 3) object origins (global)
    q: np.ndarray             # (N, 4)
    v: np.ndarray             # (N, 3) COM velocity
    w: np.ndarray             # (N, 3)
    acc: np.ndarray           # (N, 3) total acceleration at the start of the last step
    alpha: np.ndarray         # (N, 3)
    time: float
    steps: int


class SimulationEngine:
    def __init__(self, settings: SimulationSettings, plan: ExecutionPlan, bodies: BodySet, state: SimState) -> None:
        settings.check()
        if not plan.ok:
            raise SimulationError("; ".join(d.message for d in plan.errors))
        self.settings = settings
        self.plan = plan
        self.bodies = bodies
        self.clock = SimulationClock(settings.delta_time)
        n = len(bodies.ids)
        self.acc = np.zeros((n, 3))
        self.alpha = np.zeros((n, 3))
        self.state = self._apply_prescribed(0.0, state)
        self.finished = False

    # ---- process evaluation -----------------------------------------------------------------
    def _apply_prescribed(self, t: float, s: SimState) -> SimState:
        if not self.plan.prescribed:
            return s
        s = s.copy()
        b = self.bodies
        outputs = [(p, p.pdef.evaluate(t, p.idx, p.params, b)) for p in self.plan.prescribed]
        for p, out in outputs:  # orientations first: the COM offset depends on them
            if "orientation" in out:
                s.q[p.idx] = out["orientation"]
                s.w[p.idx] = out["angular_velocity"]
        com = b.com_offset_world(s.q)
        for p, out in outputs:
            if "position" in out:
                # prescribed values describe the object origin; the state stores the COM
                s.x[p.idx] = out["position"] + com[p.idx]
                s.v[p.idx] = out["velocity"] + np.cross(s.w[p.idx], com[p.idx])
        return s

    def _deriv(self, t: float, s: SimState) -> tuple[np.ndarray, np.ndarray]:
        stage = self._apply_prescribed(t, s)
        # write prescribed values back so the integrator stage uses them
        s.x[:], s.v[:], s.q[:], s.w[:] = stage.x, stage.v, stage.q, stage.w
        n = len(self.bodies.ids)
        acc = np.zeros((n, 3))
        alpha = np.zeros((n, 3))
        ctx = StepContext(t, self.clock.delta_time, self.bodies, stage.x, stage.q, stage.v, stage.w, acc, alpha)
        for p in self.plan.contributions:
            p.pdef.evaluate(ctx, p.idx, p.params)
        return acc, alpha

    # ---- stepping ---------------------------------------------------------------------------------
    def step(self) -> bool:
        """Advance one step. Returns False (and does nothing) once the duration is reached."""
        if self.finished:
            return False
        dt = self.clock.next_step_dt(self.settings.duration, self.settings.duration_policy)
        if dt is None:
            self.finished = True
            return False
        t = self.clock.time
        new, acc, alpha = integrate(self.settings.integrator, self.state.copy(), t, dt, self._deriv,
                                    self.plan.integrate_translation, self.plan.integrate_rotation)
        new = self._apply_prescribed(t + dt, new)
        self._check_finite(new, t + dt)
        self.state, self.acc, self.alpha = new, acc, alpha
        self.clock.advance(dt)
        if self.clock.next_step_dt(self.settings.duration, self.settings.duration_policy) is None:
            self.finished = True
        return True

    def _check_finite(self, s: SimState, t: float) -> None:
        for name, arr in (("position", s.x), ("velocity", s.v), ("orientation", s.q), ("angular velocity", s.w)):
            bad = ~np.all(np.isfinite(arr), axis=1) | (np.abs(arr).max(axis=1) > 1e12)
            if bad.any():
                oid = self.bodies.ids[int(np.argmax(bad))]
                raise SimulationError(f"Solver instability at t = {t:.6g} s: {name} of {oid} became non-finite or "
                                      "exceeded 1e12. Reduce Δt, change the integrator or check the process "
                                      "parameters.")

    # ---- live changes (applied between steps) ---------------------------------------------------
    def set_delta_time(self, dt: float) -> None:
        self.settings.delta_time = float(dt)
        self.settings.check()
        self.clock.set_delta_time(dt)
        self.finished = self.clock.next_step_dt(self.settings.duration, self.settings.duration_policy) is None

    def set_duration(self, duration: float) -> None:
        self.settings.duration = float(duration)
        self.settings.check()
        self.finished = self.clock.next_step_dt(self.settings.duration, self.settings.duration_policy) is None

    def replace_plan(self, plan: ExecutionPlan, bodies: BodySet, state: SimState, keep: set[str]) -> None:
        """Swap in a rebuilt plan; bodies listed in `keep` continue from the engine's own state."""
        if not plan.ok:
            raise SimulationError("; ".join(d.message for d in plan.errors))
        old = {oid: i for i, oid in enumerate(self.bodies.ids)}
        for k, oid in enumerate(bodies.ids):
            if oid in keep and oid in old:
                i = old[oid]
                state.x[k], state.q[k], state.v[k], state.w[k] = (self.state.x[i], self.state.q[i],
                                                                  self.state.v[i], self.state.w[i])
                bodies.origin0[k], bodies.q0[k] = self.bodies.origin0[i], self.bodies.q0[i]
        self.plan, self.bodies = plan, bodies
        n = len(bodies.ids)
        self.acc, self.alpha = np.zeros((n, 3)), np.zeros((n, 3))
        self.state = self._apply_prescribed(self.clock.time, state)

    # ---- output ------------------------------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        s = self.state
        return Snapshot(list(self.bodies.ids), origins(self.bodies, s), s.q.copy(), s.v.copy(), s.w.copy(),
                        self.acc.copy(), self.alpha.copy(), self.clock.time, self.clock.steps)
