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
(energy-stable for oscillators), RK4 4 for translation. Rotation in RK4 uses
the RK4-weighted angular velocity with the exact exponential map.
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
    "rk4": "Runge–Kutta 4 — 4th order",
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


def integrate(method: str, s: SimState, t: float, dt: float, deriv: Deriv, trans: np.ndarray,
              rot: np.ndarray) -> tuple[SimState, np.ndarray, np.ndarray]:
    if method == "rk4":
        return step_rk4(s, t, dt, deriv, trans, rot)
    if method in ("semi_implicit_euler", "explicit_euler"):
        return step_euler(method, s, t, dt, deriv, trans, rot)
    raise ValueError(f"Unknown integrator '{method}'")
