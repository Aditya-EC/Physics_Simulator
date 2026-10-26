"""Deterministic fixed-step simulation engine (no threading, no Qt).

One step (requirements §9.4):
  1. read the current state (shared previous state for every process);
  2. use the enabled processes of the validated execution plan;
  3. references were resolved to body indices when the plan was built;
  4-5. discrete update processes run first, in dependency order, against the
     state at the start of the step; then the integrator evaluates contribution
     processes (summed) and prescribed processes at each solver stage;
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
from .. import numerics
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
    custom: dict = None       # object id -> {key: value} custom properties written by update processes
    warnings: list = None     # skipped evaluations (error policy "skip") since the previous snapshot


class SimulationEngine:
    def __init__(self, settings: SimulationSettings, plan: ExecutionPlan, bodies: BodySet, state: SimState,
                 world=None) -> None:
        settings.check()
        if not plan.ok:
            raise SimulationError("; ".join(d.message for d in plan.errors))
        needs_world = plan.updates or any(p.pdef.is_formula for p in plan.contributions + plan.prescribed)
        if needs_world and world is None:
            raise SimulationError("Formula processes need the scene world view (capture_static)")
        self.settings = settings
        self.plan = plan
        self.bodies = bodies
        self.world = world
        if world is not None:
            world.set_bodies(plan.body_ids)
            world.dt = settings.delta_time
            bodies.world = world
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
        if self.world is not None:
            self.world.set_stage(s.x, s.q, s.v, s.w)
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
        ctx.world = self.world
        if self.world is not None:
            self.world.set_stage(stage.x, stage.q, stage.v, stage.w)
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
        updates = self._run_updates(t, dt)
        new, acc, alpha = integrate(self.settings.integrator, self.state.copy(), t, dt, self._deriv,
                                    self.plan.integrate_translation, self.plan.integrate_rotation)
        self._apply_updates(updates, new)
        new = self._apply_prescribed(t + dt, new)
        self._check_finite(new, t + dt)
        self.state, self.acc, self.alpha = new, acc, alpha
        self.clock.advance(dt)
        if self.clock.next_step_dt(self.settings.duration, self.settings.duration_policy) is None:
            self.finished = True
        return True

    def _run_updates(self, t: float, dt: float) -> dict:
        """Evaluate update processes (dependency order) against the state at the start of the step."""
        if not self.plan.updates:
            return {}
        w = self.world
        s = self.state
        w.set_stage(s.x, s.q, s.v, s.w)
        w.updated_values = {}
        for p in self.plan.updates:
            for ref, q, value in p.pdef.evaluate(w, p.idx, t, dt, p.params):
                w.updated_values[(ref, q)] = value
        return dict(w.updated_values)

    def _apply_updates(self, updates: dict, s: SimState) -> None:
        w = self.world
        b = self.bodies
        for (ref, q), value in updates.items():  # orientations first (the COM offset depends on them)
            if q == "orientation":
                s.q[w.body_of_ref(ref)] = value.q
        for (ref, q), value in updates.items():
            if q == "orientation":
                continue
            if q.startswith("custom."):
                w.custom_state[ref][q[7:]] = value
                w.custom_dirty.add((ref, q[7:]))
                w.written.add((ref, q[7:]))
                continue
            i = w.body_of_ref(ref)
            if q == "position":
                s.x[i] = np.asarray(value, float) + np.asarray(numerics.quat_rotate(s.q[i], b.com_local[i]), float)
            elif q == "velocity":
                s.v[i] = value
            elif q == "angular_velocity":
                s.w[i] = value

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
        if self.world is not None:
            self.world.dt = float(dt)
        self.finished = self.clock.next_step_dt(self.settings.duration, self.settings.duration_policy) is None

    def set_duration(self, duration: float) -> None:
        self.settings.duration = float(duration)
        self.settings.check()
        self.finished = self.clock.next_step_dt(self.settings.duration, self.settings.duration_policy) is None

    def replace_plan(self, plan: ExecutionPlan, bodies: BodySet, state: SimState, keep: set[str],
                     statics=None, globals_: dict | None = None) -> None:
        """Swap in a rebuilt plan; bodies listed in `keep` continue from the engine's own state.

        `statics` / `globals_` refresh the formula world's view of the scene (e.g. after a mass edit);
        custom-property values written by update processes are kept.
        """
        if not plan.ok:
            raise SimulationError("; ".join(d.message for d in plan.errors))
        if self.world is not None and statics is not None:
            if [st.id for st in statics] != self.world.ids:
                raise SimulationError("The scene structure changed during the simulation")
            w = self.world
            for ref, st in enumerate(statics):
                fresh = dict(st.custom)   # user edits win, except values the simulation itself writes
                for r, k in w.written:
                    if r == ref:
                        fresh[k] = w.custom_state[ref][k]
                w.custom_state[ref] = fresh
            w.statics = statics
        if self.world is not None and globals_ is not None:
            self.world.globals = globals_
        old = {oid: i for i, oid in enumerate(self.bodies.ids)}
        for k, oid in enumerate(bodies.ids):
            if oid in keep and oid in old:
                i = old[oid]
                state.x[k], state.q[k], state.v[k], state.w[k] = (self.state.x[i], self.state.q[i],
                                                                  self.state.v[i], self.state.w[i])
                bodies.origin0[k], bodies.q0[k] = self.bodies.origin0[i], self.bodies.q0[i]
        self.plan, self.bodies = plan, bodies
        if self.world is not None:
            self.world.set_bodies(plan.body_ids)
            bodies.world = self.world
        n = len(bodies.ids)
        self.acc, self.alpha = np.zeros((n, 3)), np.zeros((n, 3))
        self.state = self._apply_prescribed(self.clock.time, state)

    # ---- output ------------------------------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        s = self.state
        custom, warnings = {}, []
        if self.world is not None:
            w = self.world
            for ref, key in w.custom_dirty:
                v = w.custom_state[ref][key]
                custom.setdefault(w.ids[ref], {})[key] = v.copy() if isinstance(v, np.ndarray) else v
            w.custom_dirty = set()
            warnings, w.skips = w.skips, []
        return Snapshot(list(self.bodies.ids), origins(self.bodies, s), s.q.copy(), s.v.copy(), s.w.copy(),
                        self.acc.copy(), self.alpha.copy(), self.clock.time, self.clock.steps, custom, warnings)
