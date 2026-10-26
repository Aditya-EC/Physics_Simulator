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
from time import perf_counter

import numpy as np

from ..processes.process_base import StepContext
from ..processes.process_scheduler import ExecutionPlan
from .. import numerics
from .clock import SimulationClock
from .conditions import evaluate as evaluate_conditions
from .integrators import integrate
from .settings import SimulationSettings
from .state import BodySet, SimState, origins


STATE_FORMAT = "psim-simulation-state"
STATE_VERSION = 1
PROFILE_STAGES = ("updates", "processes_and_integration", "prescribed", "constraints", "recording", "conditions")


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
    contacts: list = None     # contact points of the last step: (point, normal from A to B, normal force N)


class SimulationEngine:
    def __init__(self, settings: SimulationSettings, plan: ExecutionPlan, bodies: BodySet, state: SimState,
                 world=None) -> None:
        settings.check()
        if not plan.ok:
            raise SimulationError("; ".join(d.message for d in plan.errors))
        needs_world = plan.updates or any(p.pdef.is_formula for p in plan.contributions + plan.prescribed)
        if (needs_world or plan.conditions) and world is None:
            raise SimulationError("Formula processes and stop conditions need the scene world view (capture_static)")
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
        self._gyro_setup()
        self._pinned_setup()
        self._physics_setup()
        self.constraint_acc = None   # (Δv/Δt, Δω/Δt) of the last step's contacts and joints
        self.state = self._apply_prescribed(0.0, state)
        self.finished = False
        self.sampler = None     # EngineSampler when data recording is attached
        self.conditions = list(plan.conditions)
        self.fired: list = []   # stop conditions that became true (taken by the runner)
        self.profile = {k: 0.0 for k in PROFILE_STAGES}   # wall seconds per stage, accumulated
        self.profile_steps = 0

    def attach_recording(self, recording, process_outputs: bool = True) -> None:
        """Record samples into `recording` every recording.interval_steps steps (see psim.recording.sampler)."""
        from ..recording.sampler import EngineSampler
        self.sampler = EngineSampler(self, recording, process_outputs)

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
        pinned = self._pinned
        if pinned is not None:
            # translation not driven by anything: the object origin stays where it is while the body turns
            s.x[pinned] = b.origin0[pinned] + com[pinned]
            s.v[pinned] = np.cross(s.w[pinned], com[pinned])
        for p, out in outputs:
            if "position" in out:
                # prescribed values describe the object origin; the state stores the COM
                s.x[p.idx] = out["position"] + com[p.idx]
                s.v[p.idx] = out["velocity"] + np.cross(s.w[p.idx], com[p.idx])
        return s

    def _deriv(self, t: float, s: SimState, quiet: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if quiet and self.world is not None:
            self.world.quiet = True
            try:
                return self._deriv(t, s)
            finally:
                self.world.quiet = False
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
        if self.settings.gyroscopic and self._gyro_rows is not None:
            alpha[self._gyro_rows] += self._gyroscopic(stage.q[self._gyro_rows], stage.w[self._gyro_rows])
        return acc, alpha

    def _pinned_setup(self) -> None:
        """Rows whose orientation is prescribed but whose position is neither integrated, prescribed nor updated."""
        n = len(self.bodies.ids)
        driven = self.plan.integrate_translation.copy()
        turned = np.zeros(n, bool)
        for p in self.plan.prescribed:
            if "position" in p.pdef.writes:
                driven[p.idx] = True
            if "orientation" in p.pdef.writes:
                turned[p.idx] = True
        for p in self.plan.updates:
            if {"position", "velocity"} & set(p.pdef.writes):
                driven[p.idx] = True
        rows = np.flatnonzero(turned & ~driven)
        self._pinned = rows if len(rows) and self.bodies.has_com_offset else None

    def _physics_setup(self) -> None:
        """Collisions and joints (the native rigid-body constraint stage), if the plan uses them."""
        setup = getattr(self.plan, "physics", None)
        self.physics = None
        if self.settings.physics_enabled and setup is not None and setup.has_work:
            from ..physics.system import PhysicsSettings, PhysicsSystem
            s = self.settings
            self.physics = PhysicsSystem(setup, self.bodies, self.plan.integrate_translation,
                                         self.plan.integrate_rotation,
                                         PhysicsSettings(s.solver_iterations, s.position_iterations, s.baumgarte,
                                                         s.contact_slop, s.restitution_threshold))

    def _gyro_setup(self) -> None:
        """Rows that get the gyroscopic term: integrated rotation and a non-zero inertia tensor."""
        b = self.bodies
        rows = np.flatnonzero(self.plan.integrate_rotation & np.array([np.any(i) for i in b.inertia_local], bool)) \
            if b.inertia_local is not None and len(b.ids) else np.zeros(0, int)
        self._gyro_rows = rows if len(rows) else None

    def _gyroscopic(self, q: np.ndarray, w: np.ndarray) -> np.ndarray:
        """−I_w⁻¹ (ω × I_w ω) for the selected rows (world frame)."""
        b, rows = self.bodies, self._gyro_rows
        i_w = numerics.rotate_tensors(np.ascontiguousarray(q), np.ascontiguousarray(b.inertia_local[rows]))
        inv_w = numerics.rotate_tensors(np.ascontiguousarray(q), np.ascontiguousarray(b.inv_inertia_local[rows]))
        lw = np.einsum("nij,nj->ni", i_w, w)
        return -np.einsum("nij,nj->ni", inv_w, np.cross(w, lw))

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
        step = self.clock.steps
        old = self.state
        prof = self.profile
        c0 = perf_counter()
        updates = self._run_updates(t, dt)
        c1 = perf_counter()
        new, acc, alpha = integrate(self.settings.integrator, self.state.copy(), t, dt, self._deriv,
                                    self.plan.integrate_translation, self.plan.integrate_rotation,
                                    self.settings.adaptive_tolerance)
        c2 = perf_counter()
        self._apply_updates(updates, new)
        new = self._apply_prescribed(t + dt, new)
        c3 = perf_counter()
        if self.physics is not None:
            dv, dw = self.physics.step(new, dt, old)
            if dt > 0:     # contact and joint forces are part of the total acceleration of this step
                self.constraint_acc = (dv / dt, dw / dt)
                acc, alpha = acc + dv / dt, alpha + dw / dt
            if self.sampler is not None:
                for a, b in self.physics.take_new_contacts():
                    self.sampler.rec.log(t + dt, step + 1, "contact", f"Contact {a} ↔ {b}")
            else:
                self.physics.take_new_contacts()
        c4 = perf_counter()
        self._check_finite(new, t + dt)
        if self.sampler is not None:
            self.sampler.after_step(t, step, old, acc, alpha, new, dt)
        c5 = perf_counter()
        self.state, self.acc, self.alpha = new, acc, alpha
        self.clock.advance(dt)
        if self.conditions:
            self._check_conditions(dt, step + 1)
        c6 = perf_counter()
        if not self.finished and \
                self.clock.next_step_dt(self.settings.duration, self.settings.duration_policy) is None:
            self.finished = True
            if self.sampler is not None:
                self.sampler.final()
        c7 = perf_counter()
        prof["updates"] += c1 - c0
        prof["processes_and_integration"] += c2 - c1
        prof["prescribed"] += c3 - c2
        prof["constraints"] += c4 - c3
        prof["recording"] += (c5 - c4) + (c7 - c6)
        prof["conditions"] += c6 - c5
        self.profile_steps += 1
        return True

    def _check_conditions(self, dt: float, step: int) -> None:
        s = self.state
        self.world.set_stage(s.x, s.q, s.v, s.w)
        for c in evaluate_conditions(self.conditions, self.world, self.clock.time, dt):
            self.fired.append((c.name, c.action, self.clock.time))
            if self.sampler is not None:
                self.sampler.rec.log(self.clock.time, step, "condition",
                                     f"Stop condition '{c.name}' became true ({c.action})")
            if c.action == "stop" and not self.finished:
                self.finished = True
                if self.sampler is not None:
                    self.sampler.final()

    def take_fired(self) -> list[tuple[str, str, float]]:
        """(name, action, time) of the stop conditions that fired since the last call."""
        out, self.fired = self.fired, []
        return out

    def profile_report(self) -> dict:
        """Average wall time per step (ms) of each stage, and their total."""
        n = max(self.profile_steps, 1)
        out = {k: 1000 * v / n for k, v in self.profile.items()}
        out["total"] = sum(out.values())
        out["steps"] = self.profile_steps
        return out

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
                     statics=None, globals_: dict | None = None, frames=None) -> None:
        """Swap in a rebuilt plan; bodies listed in `keep` continue from the engine's own state.

        `statics` / `globals_` / `frames` refresh the formula world's view of the scene (e.g. after a mass edit);
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
        if self.world is not None and frames is not None:
            self.world.frames = list(frames)
            self.world.frame_index = {f.id: i for i, f in enumerate(self.world.frames)}
        old = {oid: i for i, oid in enumerate(self.bodies.ids)}
        for k, oid in enumerate(bodies.ids):
            if oid in keep and oid in old:
                i = old[oid]
                state.x[k], state.q[k], state.v[k], state.w[k] = (self.state.x[i], self.state.q[i],
                                                                  self.state.v[i], self.state.w[i])
                bodies.origin0[k], bodies.q0[k] = self.bodies.origin0[i], self.bodies.q0[i]
        was_true = {c.name: c.was_true for c in self.conditions}
        self.conditions = list(plan.conditions)
        for c in self.conditions:      # a condition that is already true does not fire again after a live edit
            c.was_true = was_true.get(c.name, c.was_true)
        self.plan, self.bodies = plan, bodies
        if self.world is not None:
            self.world.set_bodies(plan.body_ids)
            bodies.world = self.world
        n = len(bodies.ids)
        self.acc, self.alpha = np.zeros((n, 3)), np.zeros((n, 3))
        self._gyro_setup()
        self._pinned_setup()
        self._physics_setup()
        self.state = self._apply_prescribed(self.clock.time, state)
        if self.sampler is not None:
            self.sampler.refresh_schema()

    # ---- saved simulation states ------------------------------------------------------------------------
    def export_state(self) -> dict:
        """Everything needed to continue this run exactly from its current step (JSON-serialisable)."""
        s = self.state
        bodies = {oid: {"x": s.x[i].tolist(), "q": s.q[i].tolist(), "v": s.v[i].tolist(), "w": s.w[i].tolist(),
                        "acc": self.acc[i].tolist(), "alpha": self.alpha[i].tolist()}
                  for i, oid in enumerate(self.bodies.ids)}
        custom = {}
        if self.world is not None:
            for ref, key in sorted(self.world.written):
                v = self.world.custom_state[ref][key]
                custom.setdefault(self.world.ids[ref], {})[key] = v.tolist() if isinstance(v, np.ndarray) else v
        physics = None
        if self.physics is not None:
            physics = {"touching": sorted(sorted(p) for p in self.physics.touching),
                       "last_dv": self.physics._last_dv.tolist(), "last_dw": self.physics._last_dw.tolist(),
                       "keys": list(self.physics.keys)}
        return {"format": STATE_FORMAT, "version": STATE_VERSION, "time": self.clock.time,
                "clock": self.clock.export(), "integrator": self.settings.integrator, "bodies": bodies,
                "custom": custom, "physics": physics, "conditions": {c.name: c.was_true for c in self.conditions}}

    def import_state(self, d: dict) -> None:
        """Continue from a state saved by export_state for the same simulated objects. Recorded samples later
        than or at the restored time are discarded (the next step records it again)."""
        if not isinstance(d, dict) or d.get("format") != STATE_FORMAT:
            raise SimulationError("Not a PSim simulation state")
        if int(d.get("version", 0)) > STATE_VERSION:
            raise SimulationError(f"Simulation state version {d.get('version')} is newer than this application "
                                  f"supports ({STATE_VERSION})")
        saved = d.get("bodies", {})
        ids = list(self.bodies.ids)
        if set(saved) != set(ids):
            extra, missing = sorted(set(saved) - set(ids)), sorted(set(ids) - set(saved))
            raise SimulationError("The saved state does not match the simulated objects"
                                  + (f" (not simulated now: {', '.join(extra[:5])})" if extra else "")
                                  + (f" (not in the state: {', '.join(missing[:5])})" if missing else ""))
        try:
            new = self.state.copy()
            acc, alpha = self.acc.copy(), self.alpha.copy()
            for i, oid in enumerate(ids):
                b = saved[oid]
                new.x[i], new.v[i], new.w[i] = (np.asarray(b[k], float).reshape(3) for k in ("x", "v", "w"))
                q = np.asarray(b["q"], float).reshape(4)
                new.q[i] = q / np.linalg.norm(q)
                acc[i] = np.asarray(b.get("acc", [0, 0, 0]), float).reshape(3)
                alpha[i] = np.asarray(b.get("alpha", [0, 0, 0]), float).reshape(3)
            values = np.concatenate([new.x.ravel(), new.q.ravel(), new.v.ravel(), new.w.ravel()])
            if not np.all(np.isfinite(values)):
                raise ValueError("non-finite values")
            clock = SimulationClock(self.clock.delta_time)
            clock.restore(d["clock"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise SimulationError(f"Invalid simulation state: {exc}") from None
        if self.world is not None:
            for oid, values in dict(d.get("custom", {})).items():
                ref = self.world.ref_index.get(oid)
                if ref is None:
                    continue
                for key, v in values.items():
                    self.world.custom_state[ref][key] = np.asarray(v, float) if isinstance(v, list) else v
                    self.world.written.add((ref, key))
                    self.world.custom_dirty.add((ref, key))
        ph = d.get("physics")
        if self.physics is not None:
            self.physics.touching = set()
            self.physics._last_dv[:] = 0.0
            self.physics._last_dw[:] = 0.0
            if ph and list(ph.get("keys", [])) == list(self.physics.keys):
                self.physics.touching = {frozenset(p) for p in ph.get("touching", [])}
                self.physics._last_dv[:] = np.asarray(ph["last_dv"], float)
                self.physics._last_dw[:] = np.asarray(ph["last_dw"], float)
        for c in self.conditions:
            c.was_true = bool(dict(d.get("conditions", {})).get(c.name, False))
        self.state, self.acc, self.alpha, self.clock = new, acc, alpha, clock
        self.finished = self.clock.next_step_dt(self.settings.duration, self.settings.duration_policy) is None
        if self.sampler is not None:
            removed = self.sampler.rec.truncate_from(self.clock.time)
            self.sampler.rec.log(self.clock.time, self.clock.steps, "restore",
                                 f"Simulation state restored at t = {self.clock.time:.6g} s"
                                 + (f" ({removed} later samples discarded)" if removed else ""))

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
        contacts = []
        if self.physics is not None:
            dt = max(self.clock.delta_time, 1e-12)
            contacts = [(np.array(c[2], float), np.array(c[3], float), float(c[5]) / dt)
                        for c in getattr(self.physics, "last_contacts", [])]
        return Snapshot(list(self.bodies.ids), origins(self.bodies, s), s.q.copy(), s.v.copy(), s.w.copy(),
                        self.acc.copy(), self.alpha.copy(), self.clock.time, self.clock.steps, custom, warnings,
                        contacts)
