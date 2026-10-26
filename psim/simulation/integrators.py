"""Fixed-step integrators.

Each integrator advances the state from t to t + dt given a derivative
function ``deriv(t, state) -> (acc, alpha)`` that (1) applies prescribed
processes at time t to a stage copy of the state and (2) evaluates every
contribution process on that stage state. Derivative-based solvers (RK4)
therefore evaluate processes at their intermediate stages.

Masks select which bodies have integrated (rather than prescribed or fixed)
translation and rotation; other rows are left unchanged here and set by the
prescribed processes after the step.

Order of accuracy: explicit Euler 1, semi-implicit (symplectic) Euler 1
(energy-stable for oscillators), midpoint (RK2) 2, velocity Verlet 2
(symplectic for position-dependent forces), RK4 4, Dormand–Prince 5(4) with
adaptive sub-steps inside each fixed Δt (the step itself never changes).
Rotation uses the stage-weighted angular velocity with the exact exponential map.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .. import numerics
from .state import SimState

Deriv = Callable[[float, SimState], tuple[np.ndarray, np.ndarray]]

INTEGRATORS = {
    "semi_implicit_euler": "Semi-implicit (symplectic) Euler — 1st order, default",
    "explicit_euler": "Explicit Euler — 1st order",
    "midpoint": "Midpoint (RK2) — 2nd order",
    "velocity_verlet": "Velocity Verlet — 2nd order, symplectic",
    "rk4": "Runge–Kutta 4 — 4th order",
    "dopri5": "Dormand–Prince 5(4) — adaptive sub-steps within each Δt",
}


def _masked(new: np.ndarray, old: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.where(mask[:, None], new, old))


def step_euler(method: str, s: SimState, t: float, dt: float, deriv: Deriv, trans: np.ndarray,
               rot: np.ndarray) -> tuple[SimState, np.ndarray, np.ndarray]:
    acc, alpha = deriv(t, s)
    x, v = s.x.copy(), s.v.copy()
    numerics.integrate_linear(method, x, v, acc, dt)
    w_rot = s.w if method == "explicit_euler" else s.w + alpha * dt
    q = s.q.copy()
    numerics.quat_integrate(q, np.ascontiguousarray(w_rot), dt)
    new = SimState(x=_masked(x, s.x, trans), q=_masked(q, s.q, rot), v=_masked(v, s.v, trans),
                   w=_masked(s.w + alpha * dt, s.w, rot))
    return new, acc, alpha


def step_rk4(s: SimState, t: float, dt: float, deriv: Deriv, trans: np.ndarray,
             rot: np.ndarray) -> tuple[SimState, np.ndarray, np.ndarray]:
    def stage(base: SimState, kv, ka, kw, kal, h) -> SimState:
        q = base.q.copy()
        numerics.quat_integrate(q, np.ascontiguousarray(kw), h)
        return SimState(base.x + kv * h, q, base.v + ka * h, base.w + kal * h)

    a1, al1 = deriv(t, s)
    v1, w1 = s.v, s.w
    s2 = stage(s, v1, a1, w1, al1, dt / 2)
    a2, al2 = deriv(t + dt / 2, s2)
    s3 = stage(s, s2.v, a2, s2.w, al2, dt / 2)
    a3, al3 = deriv(t + dt / 2, s3)
    s4 = stage(s, s3.v, a3, s3.w, al3, dt)
    a4, al4 = deriv(t + dt, s4)
    x = s.x + dt / 6 * (v1 + 2 * s2.v + 2 * s3.v + s4.v)
    v = s.v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
    w_avg = (w1 + 2 * s2.w + 2 * s3.w + s4.w) / 6
    q = s.q.copy()
    numerics.quat_integrate(q, np.ascontiguousarray(w_avg), dt)
    w = s.w + dt / 6 * (al1 + 2 * al2 + 2 * al3 + al4)
    new = SimState(_masked(x, s.x, trans), _masked(q, s.q, rot), _masked(v, s.v, trans), _masked(w, s.w, rot))
    return new, a1, al1


def step_midpoint(s: SimState, t: float, dt: float, deriv: Deriv, trans: np.ndarray,
                  rot: np.ndarray) -> tuple[SimState, np.ndarray, np.ndarray]:
    a1, al1 = deriv(t, s)
    q_mid = s.q.copy()
    numerics.quat_integrate(q_mid, np.ascontiguousarray(s.w), dt / 2)
    mid = SimState(s.x + s.v * dt / 2, q_mid, s.v + a1 * dt / 2, s.w + al1 * dt / 2)
    a2, al2 = deriv(t + dt / 2, mid)
    q = s.q.copy()
    numerics.quat_integrate(q, np.ascontiguousarray(mid.w), dt)
    new = SimState(_masked(s.x + mid.v * dt, s.x, trans), _masked(q, s.q, rot), _masked(s.v + a2 * dt, s.v, trans),
                   _masked(s.w + al2 * dt, s.w, rot))
    return new, a1, al1


def step_verlet(s: SimState, t: float, dt: float, deriv: Deriv, trans: np.ndarray,
                rot: np.ndarray) -> tuple[SimState, np.ndarray, np.ndarray]:
    """Velocity Verlet (kick-drift-kick). Velocity-dependent forces use the predicted velocity."""
    a1, al1 = deriv(t, s)
    v_half, w_half = s.v + a1 * dt / 2, s.w + al1 * dt / 2
    q = s.q.copy()
    numerics.quat_integrate(q, np.ascontiguousarray(w_half), dt)
    pred = SimState(s.x + v_half * dt, q, s.v + a1 * dt, s.w + al1 * dt)
    a2, al2 = deriv(t + dt, pred)
    new = SimState(_masked(pred.x, s.x, trans), _masked(q, s.q, rot), _masked(v_half + a2 * dt / 2, s.v, trans),
                   _masked(w_half + al2 * dt / 2, s.w, rot))
    return new, a1, al1


# Dormand–Prince 5(4) tableau
_DP_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_DP_A = ((), (1 / 5,), (3 / 40, 9 / 40), (44 / 45, -56 / 15, 32 / 9),
         (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
         (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
         (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84))
_DP_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_DP_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)


class AdaptiveStepError(RuntimeError):
    pass


def _dp_attempt(s: SimState, t: float, h: float, deriv: Deriv, first):
    kx, kv, kw, ka = [], [], [], []   # dx = v, dv = a, dq ~ w, dw = alpha
    for i in range(7):
        if i == 0 and first is not None:
            a, al = first
            st = s
        else:
            ax = sum((c * k for c, k in zip(_DP_A[i], kx)), np.zeros_like(s.x))
            av = sum((c * k for c, k in zip(_DP_A[i], kv)), np.zeros_like(s.v))
            aw = sum((c * k for c, k in zip(_DP_A[i], kw)), np.zeros_like(s.w))
            aal = sum((c * k for c, k in zip(_DP_A[i], ka)), np.zeros_like(s.w))
            q = s.q.copy()
            numerics.quat_integrate(q, np.ascontiguousarray(aw), h)
            st = SimState(s.x + ax * h, q, s.v + av * h, s.w + aal * h)
            a, al = deriv(t + _DP_C[i] * h, st)
        kx.append(st.v.copy())
        kv.append(a)
        kw.append(st.w.copy())
        ka.append(al)
    comb = [lambda b, ks=ks: sum(c * k for c, k in zip(b, ks)) for ks in (kx, kv, kw, ka)]
    dx5, dv5, dw5, da5 = (f(_DP_B5) for f in comb)
    dx4, dv4, _dw4, da4 = (f(_DP_B4) for f in comb)
    q = s.q.copy()
    numerics.quat_integrate(q, np.ascontiguousarray(dw5), h)
    new = SimState(s.x + dx5 * h, q, s.v + dv5 * h, s.w + da5 * h)
    err = np.concatenate([((dx5 - dx4) * h).ravel(), ((dv5 - dv4) * h).ravel(), ((da5 - da4) * h).ravel()])
    scale = np.concatenate([np.abs(new.x).ravel(), np.abs(new.v).ravel(), np.abs(new.w).ravel()])
    return new, err, scale, (kv[6], ka[6])


def step_dopri5(s: SimState, t: float, dt: float, deriv: Deriv, trans: np.ndarray, rot: np.ndarray,
                tol: float = 1e-8, max_substeps: int = 10_000) -> tuple[SimState, np.ndarray, np.ndarray]:
    """Adaptive Dormand–Prince sub-steps that end exactly at t + dt (error per sub-step ≤ tol·(1 + |y|))."""
    a1, al1 = deriv(t, s)
    cur, tc, end = s, t, t + dt
    h = dt
    first = (a1, al1)
    for _ in range(max_substeps):
        h = min(h, end - tc)
        new, err, scale, last = _dp_attempt(cur, tc, h, deriv, first)
        norm = float(np.max(np.abs(err) / (tol * (1.0 + scale)))) if err.size else 0.0
        if not np.isfinite(norm):
            norm = 1e9
        if norm <= 1.0:
            # keep non-integrated rows unchanged (prescribed / fixed bodies)
            cur = SimState(_masked(new.x, s.x, trans), _masked(new.q, s.q, rot), _masked(new.v, s.v, trans),
                           _masked(new.w, s.w, rot))
            tc += h
            first = last   # FSAL: the last stage is the first stage of the next sub-step
            if end - tc <= 1e-12 * max(1.0, abs(end)):
                return cur, a1, al1
        h *= min(5.0, max(0.2, 0.9 * norm ** -0.2)) if norm > 0 else 5.0
        if h < 1e-14 * max(1.0, abs(tc)):
            raise AdaptiveStepError(f"Adaptive integrator could not meet the tolerance {tol:g} at t = {tc:.6g} s")
    raise AdaptiveStepError(f"Adaptive integrator needed more than {max_substeps} sub-steps in one Δt")


def integrate(method: str, s: SimState, t: float, dt: float, deriv: Deriv, trans: np.ndarray,
              rot: np.ndarray, tol: float = 1e-8) -> tuple[SimState, np.ndarray, np.ndarray]:
    if method == "rk4":
        return step_rk4(s, t, dt, deriv, trans, rot)
    if method == "midpoint":
        return step_midpoint(s, t, dt, deriv, trans, rot)
    if method == "velocity_verlet":
        return step_verlet(s, t, dt, deriv, trans, rot)
    if method == "dopri5":
        return step_dopri5(s, t, dt, deriv, trans, rot, tol)
    if method in ("semi_implicit_euler", "explicit_euler"):
        return step_euler(method, s, t, dt, deriv, trans, rot)
    raise ValueError(f"Unknown integrator '{method}'")
