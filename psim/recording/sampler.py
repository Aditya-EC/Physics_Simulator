"""Turns the engine state into recorded samples.

Every sample is consistent in time: the values at simulated time t_k are the
state *at the start* of step k together with the accelerations the processes
computed for that state (the integrator's first stage). This costs no extra
evaluation. Exceptions:

* bodies whose motion is prescribed: accelerations are the central difference
  of the prescribed velocity, evaluated at t ± h;
* bodies moved by discrete update processes: accelerations are the change of
  velocity over the step divided by Δt;
* the final sample of a run (after the last step) evaluates the processes once.

Objects that are not simulated but move with a simulated ancestor are recorded
too (rigid-body motion of the ancestor). Per-assignment contributions
("process outputs") need one extra, silent evaluation per sample and can be
switched off.
"""
from __future__ import annotations

import numpy as np

from ..processes.process_base import StepContext
from ..simulation.state import SimState, origins
from .recording import Recording

OBJECT_CHANNELS = (("origin", 3), ("q", 4), ("com", 3), ("v", 3), ("w", 3), ("acc", 3), ("alpha", 3))


class EngineSampler:
    def __init__(self, engine, recording: Recording, process_outputs: bool = True) -> None:
        self.engine, self.rec, self.process_outputs = engine, recording, process_outputs
        self.refresh_schema()

    # ---- schema -------------------------------------------------------------------------------------------
    def refresh_schema(self) -> None:
        eng = self.engine
        world = eng.world
        self.body_ids = list(eng.bodies.ids)
        self.followers: list[tuple[str, int, int]] = []   # (object id, world ref, body index)
        if world is not None:
            for ref, d in world.driver.items():
                if d != ref:
                    self.followers.append((world.ids[ref], ref, world.ref_to_body[d]))
        specs = [((oid, ch), w) for oid in self.body_ids + [f[0] for f in self.followers] for ch, w in OBJECT_CHANNELS]
        self.proc_targets: list[tuple[object, str, int]] = []
        if self.process_outputs:
            for p in eng.plan.contributions:
                for i in np.atleast_1d(p.idx):
                    oid = self.body_ids[int(i)]
                    self.proc_targets.append((p, oid, int(i)))
                    specs += [(("proc", p.assignment.uid, oid, "acc"), 3), (("proc", p.assignment.uid, oid, "alpha"), 3)]
        self.custom_keys: list[tuple[int, str, str, int]] = []   # (world ref, object id, key, width)
        if world is not None:
            for p in eng.plan.updates:
                spec = p.pdef.formula_spec
                for c in (spec.custom if spec else []):
                    for i in np.atleast_1d(p.idx):
                        ref = world.ref_of_body(int(i))
                        w = 3 if c.get("type") == "vec3" else 1
                        item = (ref, world.ids[ref], c["key"], w)
                        if item not in self.custom_keys:
                            self.custom_keys.append(item)
                            specs.append(((world.ids[ref], f"custom.{c['key']}"), w))
        self.rec.ensure_columns(specs)
        self._prescribed_trans = np.zeros(len(self.body_ids), bool)
        self._prescribed_rot = np.zeros(len(self.body_ids), bool)
        for p in eng.plan.prescribed:
            writes = set(p.pdef.writes)
            if "position" in writes:
                self._prescribed_trans[np.atleast_1d(p.idx)] = True
            if "orientation" in writes:
                self._prescribed_rot[np.atleast_1d(p.idx)] = True
        meta = self.rec.meta
        b = eng.bodies
        for k, oid in enumerate(self.body_ids):
            meta["objects"].setdefault(oid, {})
            meta["objects"][oid].update(mass=float(b.mass[k]), com_local=b.com_local[k].tolist(), body=True,
                                        inv_inertia_local=b.inv_inertia_local[k].tolist())
        for oid, ref, _ in self.followers:
            st = world.statics[ref]
            meta["objects"].setdefault(oid, {})
            meta["objects"][oid].update(mass=float(st.mass), com_local=st.com_local.tolist(), body=False,
                                        inertia_local=st.inertia_local.tolist())
        for p in eng.plan.contributions + eng.plan.prescribed + eng.plan.updates:
            meta["processes"][p.assignment.uid] = {"process_id": p.pdef.process_id, "name": p.pdef.name,
                                                   "kind": p.pdef.kind}

    # ---- sampling -----------------------------------------------------------------------------------------
    def after_step(self, t: float, step: int, old: SimState, acc: np.ndarray, alpha: np.ndarray, new: SimState,
                   dt: float) -> None:
        if not self.rec.wants(step):
            return
        acc, alpha = acc.copy(), alpha.copy()
        eng = self.engine
        pinned = np.zeros(len(acc), bool)
        if getattr(eng, "_pinned", None) is not None:
            pinned[eng._pinned] = True
        discrete_t = ~eng.plan.integrate_translation & ~self._prescribed_trans & ~pinned
        discrete_r = ~eng.plan.integrate_rotation & ~self._prescribed_rot
        if dt > 0:
            acc[discrete_t] = (new.v[discrete_t] - old.v[discrete_t]) / dt
            alpha[discrete_r] = (new.w[discrete_r] - old.w[discrete_r]) / dt
        self._fix_prescribed(t, old, acc, alpha, max(dt, 1e-6))
        self._fix_pinned(old, acc, alpha)
        self._append(t, step, old, acc, alpha)

    def final(self) -> None:
        """Record the end state (evaluates the processes once at the final time)."""
        eng = self.engine
        t, step = eng.clock.time, eng.clock.steps
        if self.rec.n and self.rec.steps()[-1] == step:
            return
        s = eng.state.copy()
        acc, alpha = eng._deriv(t, s, quiet=True)
        if getattr(eng, "constraint_acc", None) is not None:
            # contact / joint forces at the end: those of the last step (the best available estimate)
            acc, alpha = acc + eng.constraint_acc[0], alpha + eng.constraint_acc[1]
        self._fix_prescribed(t, eng.state, acc, alpha, max(eng.clock.delta_time, 1e-6))
        self._fix_pinned(eng.state, acc, alpha)
        self._append(t, step, eng.state, acc, alpha)

    def _fix_pinned(self, s: SimState, acc: np.ndarray, alpha: np.ndarray) -> None:
        """Bodies turning about their fixed origin: the COM moves on a rigid circle, a = α×c + ω×(ω×c)."""
        rows = getattr(self.engine, "_pinned", None)
        if rows is None:
            return
        c = self.engine.bodies.com_offset_world(s.q)[rows]
        w = s.w[rows]
        acc[rows] = np.cross(alpha[rows], c) + np.cross(w, np.cross(w, c))

    def _fix_prescribed(self, t: float, s: SimState, acc: np.ndarray, alpha: np.ndarray, dt: float) -> None:
        eng = self.engine
        if not eng.plan.prescribed:
            return
        h = 1e-3 * dt
        world = eng.world
        for p in eng.plan.prescribed:
            if world is not None:
                world.set_stage(s.x, s.q, s.v, s.w)
            hi = p.pdef.evaluate(t + h, p.idx, p.params, eng.bodies)
            lo = p.pdef.evaluate(t - h, p.idx, p.params, eng.bodies)
            if "velocity" in hi:
                acc[p.idx] = (np.asarray(hi["velocity"]) - np.asarray(lo["velocity"])) / (2 * h)
            if "angular_velocity" in hi:
                alpha[p.idx] = (np.asarray(hi["angular_velocity"]) - np.asarray(lo["angular_velocity"])) / (2 * h)

    def _append(self, t: float, step: int, s: SimState, acc, alpha) -> None:
        eng = self.engine
        b = eng.bodies
        org = origins(b, s)
        values: dict = {}
        for k, oid in enumerate(self.body_ids):
            values[(oid, "origin")] = org[k]
            values[(oid, "q")] = s.q[k]
            values[(oid, "com")] = s.x[k]
            values[(oid, "v")] = s.v[k]
            values[(oid, "w")] = s.w[k]
            values[(oid, "acc")] = acc[k]
            values[(oid, "alpha")] = alpha[k]
        world = eng.world
        if world is not None and (self.followers or self.custom_keys):
            world.set_stage(s.x, s.q, s.v, s.w)
            for oid, ref, k in self.followers:
                origin, q, v, w, com = world._pose(ref)
                r = com - s.x[k]
                values[(oid, "origin")] = origin
                values[(oid, "q")] = q
                values[(oid, "com")] = com
                values[(oid, "v")] = v
                values[(oid, "w")] = w
                values[(oid, "acc")] = acc[k] + np.cross(alpha[k], r) + np.cross(s.w[k], np.cross(s.w[k], r))
                values[(oid, "alpha")] = alpha[k]
            for ref, oid, key, _w in self.custom_keys:
                v = world.custom_state.get(ref, {}).get(key)
                if isinstance(v, (int, float, np.ndarray, list)) and not isinstance(v, bool):
                    values[(oid, f"custom.{key}")] = np.asarray(v, float)
        if self.proc_targets:
            self._process_outputs(t, s, values)
        self.rec.append(t, step, values)

    def _process_outputs(self, t: float, s: SimState, values: dict) -> None:
        eng = self.engine
        n = len(self.body_ids)
        world = eng.world
        if world is not None:
            world.set_stage(s.x, s.q, s.v, s.w)
            world.quiet = True
        try:
            done = {}
            for p, oid, i in self.proc_targets:
                key = p.assignment.uid
                if key not in done:
                    acc, alpha = np.zeros((n, 3)), np.zeros((n, 3))
                    ctx = StepContext(t, eng.clock.delta_time, eng.bodies, s.x, s.q, s.v, s.w, acc, alpha)
                    ctx.world = world
                    try:
                        p.pdef.evaluate(ctx, p.idx, p.params)
                    except Exception:  # noqa: BLE001 — the step itself reports process errors
                        acc[:], alpha[:] = np.nan, np.nan
                    done[key] = (acc, alpha)
                acc, alpha = done[key]
                values[("proc", key, oid, "acc")] = acc[i]
                values[("proc", key, oid, "alpha")] = alpha[i]
        finally:
            if world is not None:
                world.quiet = False
