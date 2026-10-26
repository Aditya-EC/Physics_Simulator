"""Additional integrators (orders of convergence, adaptive accuracy) and Euler's equations for rigid rotation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim import numerics
from psim.processes.formula_process import FormulaProcessSpec, build_process_def, compile_spec
from psim.processes.process_base import TargetFilter
from psim.processes.process_registry import process_registry

from test_recording import make_engine, run


@pytest.fixture
def oscillator(scene):
    """Unit mass on a linear spring to the origin: x(t) = cos(ω t), ω = 2."""
    reg = process_registry()
    spec = FormulaProcessSpec.from_dict({"process_id": "t.spring_origin", "name": "Spring to origin",
                                         "kind": "contribution", "requires_mass": True,
                                         "script": "force = -4[N/m] * self.position"})
    cp, issues = compile_spec(spec)
    assert cp is not None, issues
    reg._defs[spec.process_id] = build_process_def(cp)
    s, e = scene
    o = e.add_new("body.point_mass", position=[1, 0, 0])
    o.mass_input = 1.0
    e.assign_process("t.spring_origin", TargetFilter(ids=[o.id]))
    yield scene
    reg._defs.pop(spec.process_id, None)


def _error(scene, integrator, dt, duration=2.0, **extra):
    eng = run(make_engine(scene, delta_time=dt, duration=duration, integrator=integrator, **extra))
    x = eng.snapshot().origin[0][0]
    return abs(x - math.cos(2 * duration)), eng.clock.steps


@pytest.mark.parametrize("integrator,order", [("explicit_euler", 1), ("semi_implicit_euler", 1), ("midpoint", 2),
                                              ("velocity_verlet", 2), ("rk4", 4)])
def test_orders_of_convergence(oscillator, integrator, order):
    e1, _ = _error(oscillator, integrator, 0.01)
    e2, _ = _error(oscillator, integrator, 0.005)
    assert math.log2(e1 / e2) == pytest.approx(order, abs=0.3)


def test_adaptive_dormand_prince_keeps_the_fixed_step(oscillator):
    err, steps = _error(oscillator, "dopri5", 0.25, adaptive_tolerance=1e-11)
    assert steps == 8 and err < 1e-8          # Δt and the step count are unchanged; sub-steps give the accuracy
    coarse, _ = _error(oscillator, "dopri5", 0.25, adaptive_tolerance=1e-4)
    assert err < coarse


def test_velocity_verlet_conserves_energy_long_term(oscillator):
    eng = run(make_engine(oscillator, delta_time=0.05, duration=200.0, integrator="velocity_verlet"))
    sn = eng.snapshot()
    energy = 0.5 * sn.v[0] @ sn.v[0] + 0.5 * 4 * sn.origin[0] @ sn.origin[0]
    # bounded oscillation of order (ωΔt)²/4 = 0.25 %, no drift over 640 periods
    assert energy == pytest.approx(2.0, rel=3e-3)
    euler = run(make_engine(oscillator, delta_time=0.05, duration=200.0, integrator="explicit_euler")).snapshot()
    assert 0.5 * euler.v[0] @ euler.v[0] + 2 * euler.origin[0] @ euler.origin[0] > 100   # explicit Euler blows up


def _spinning_box(scene, gyroscopic=True, duration=20.0):
    s, e = scene
    box = e.add_new("body.rigid_box", params={"length": 0.3, "width": 0.6, "height": 0.9})
    # spin mostly about the intermediate principal axis (local y) with a small perturbation
    box.angular_velocity = [0.01, 2.0, 0.01]
    e.assign_process("kin.free_motion", TargetFilter(ids=[box.id]))
    from psim.recording.recording import Recording
    rec = Recording(interval_steps=10)
    eng = run(make_engine(scene, rec, delta_time=0.002, duration=duration, integrator="rk4", gyroscopic=gyroscopic))
    return box, rec, eng


def test_torque_free_rotation_conserves_momentum_and_energy(scene):
    box, rec, eng = _spinning_box(scene)
    inertia = np.asarray(box.mass_properties()["inertia"], float)
    q, w = rec.column((box.id, "q")), rec.column((box.id, "w"))
    L, E, w_local = [], [], []
    for qk, wk in zip(q, w):
        r = np.asarray(numerics.quat_to_matrix(qk))
        i_w = r @ inertia @ r.T
        L.append(i_w @ wk)
        E.append(0.5 * wk @ i_w @ wk)
        w_local.append(r.T @ wk)
    L, E, w_local = np.array(L), np.array(E), np.array(w_local)
    np.testing.assert_allclose(L, np.tile(L[0], (len(L), 1)), atol=1e-6 * np.linalg.norm(L[0]))
    np.testing.assert_allclose(E, E[0], rtol=1e-6)
    # Dzhanibekov (tennis-racket) effect: rotation about the intermediate axis is unstable and flips
    assert w_local[:, 1].max() > 1.9 and w_local[:, 1].min() < -1.9


def test_without_gyroscopic_term_the_spin_does_not_flip(scene):
    box, rec, _eng = _spinning_box(scene, gyroscopic=False, duration=5.0)
    w = rec.column((box.id, "w"))
    np.testing.assert_allclose(w, np.tile(w[0], (len(w), 1)), atol=1e-12)   # constant ω in the world frame
