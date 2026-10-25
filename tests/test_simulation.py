"""Simulation engine: step counts, time progression, rate scheduling, determinism, integrators (§9.9)."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from psim.processes.process_base import TargetFilter
from psim.processes.process_scheduler import build_plan
from psim.simulation.runner import RunState, SimulationRunner
from psim.simulation.settings import DurationPolicy, PacingMode, SimulationSettings
from psim.simulation.simulation_engine import SimulationEngine, SimulationError
from psim.simulation.state import capture_bodies

G = np.array([0.0, 0.0, -9.80665])


def make_engine(scene, **settings) -> SimulationEngine:
    s, _ = scene
    plan = build_plan(s, s.assignments)
    assert plan.ok, [d.message for d in plan.errors]
    bodies, state = capture_bodies(s, plan.body_ids)
    return SimulationEngine(SimulationSettings(**settings), plan, bodies, state)


def run_to_end(engine) -> SimulationEngine:
    while engine.step():
        pass
    return engine


def projectile(scene, v0=(3.0, 0.0, 5.0)):
    s, e = scene
    ball = e.add_new("basic.sphere", position=[0, 0, 10], params={"radius": 0.1})
    ball.linear_velocity = list(v0)
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    return ball


# ---- step counts, clock and duration -----------------------------------------------------------------

def test_step_count_and_simulated_time(scene):
    projectile(scene)
    eng = run_to_end(make_engine(scene, delta_time=0.01, duration=10.0, pacing=PacingMode.AS_FAST_AS_POSSIBLE))
    assert eng.clock.steps == 1000
    assert eng.clock.time == pytest.approx(10.0, abs=1e-12)


@pytest.mark.parametrize("policy,steps,end", [
    (DurationPolicy.FINAL_PARTIAL_STEP, 4, 1.0),
    (DurationPolicy.NEXT_FULL_STEP, 4, 1.2),
    (DurationPolicy.LAST_FULL_STEP, 3, 0.9),
])
def test_duration_boundary_policies(scene, policy, steps, end):
    projectile(scene)
    eng = run_to_end(make_engine(scene, delta_time=0.3, duration=1.0, duration_policy=policy))
    assert eng.clock.steps == steps
    assert eng.clock.time == pytest.approx(end)


def test_each_step_advances_clock_by_delta_time(scene):
    projectile(scene)
    eng = make_engine(scene, delta_time=0.004, duration=1.0)
    times = []
    while eng.step():
        times.append(eng.clock.time)
    np.testing.assert_allclose(np.diff(times), 0.004, atol=1e-12)
    assert len(times) == 250


def test_delta_time_change_applies_to_subsequent_steps(scene):
    projectile(scene)
    eng = make_engine(scene, delta_time=0.01, duration=100.0)
    for _ in range(100):
        eng.step()
    eng.set_delta_time(0.001)
    for _ in range(100):
        eng.step()
    assert eng.clock.time == pytest.approx(1.1)
    assert eng.clock.steps == 200


# ---- determinism and independence from rendering / pacing ---------------------------------------------

def _complex_scene(scene):
    s, e = scene
    a = e.add_new("body.point_mass", position=[0, 0, 2])
    b = e.add_new("body.rigid_box", position=[1.3, 0.2, 1], params={"length": 0.3, "width": 0.3, "height": 0.3})
    c = e.add_new("basic.cube", position=[-2, 0, 1])
    c.torque = [0.0, 0.0, 2.0]
    c.force = [1.0, 0.5, 0.0]
    e.assign_process("dyn.spring", TargetFilter(ids=[a.id, b.id]), {"stiffness": 30, "rest_length": 1.0,
                                                                    "damping": 0.2})
    e.assign_process("dyn.gravity", TargetFilter(ids=[b.id]))
    e.assign_process("dyn.applied_force", TargetFilter(ids=[c.id]))
    e.assign_process("dyn.damping", TargetFilter(ids=[c.id]), {"linear": 0.3, "angular": 0.1})
    return a, b, c


@pytest.mark.parametrize("integrator", ["explicit_euler", "semi_implicit_euler", "rk4"])
def test_repeated_runs_are_bitwise_identical(scene, integrator):
    _complex_scene(scene)
    r1 = run_to_end(make_engine(scene, delta_time=0.002, duration=1.0, integrator=integrator)).snapshot()
    r2 = run_to_end(make_engine(scene, delta_time=0.002, duration=1.0, integrator=integrator)).snapshot()
    for f in ("origin", "q", "v", "w"):
        assert np.array_equal(getattr(r1, f), getattr(r2, f))


def _runner_result(scene, **settings):
    runner = SimulationRunner()
    runner.start(make_engine(scene, **settings))
    assert runner.wait_until(lambda: runner.state == RunState.FINISHED, 30)
    snap = runner.engine.snapshot()
    runner.stop()
    return snap


def test_results_independent_of_render_rate_and_pacing(qapp, scene):
    _complex_scene(scene)
    base = dict(delta_time=0.002, duration=0.6, integrator="rk4")
    ref = run_to_end(make_engine(scene, **base)).snapshot()
    slow_render = _runner_result(scene, pacing=PacingMode.AS_FAST_AS_POSSIBLE, render_fps=2, **base)
    fast_render = _runner_result(scene, pacing=PacingMode.AS_FAST_AS_POSSIBLE, render_fps=240, **base)
    realtime = _runner_result(scene, pacing=PacingMode.REAL_TIME, update_rate_hz=2000, **base)
    for snap in (slow_render, fast_render, realtime):
        assert snap.steps == ref.steps == 300
        for f in ("origin", "q", "v", "w"):
            assert np.array_equal(getattr(snap, f), getattr(ref, f))


# ---- real-time scheduling --------------------------------------------------------------------------------

def test_real_time_rate_and_time_progression(qapp, scene):
    """100 Hz with Δt = 0.001 s: ≈100 steps and ≈0.1 simulated s per real second."""
    projectile(scene)
    runner = SimulationRunner()
    runner.start(make_engine(scene, delta_time=0.001, update_rate_hz=100, duration=1000))
    t0 = time.perf_counter()
    time.sleep(1.0)
    runner.pause()
    assert runner.wait_until(lambda: runner.state == RunState.PAUSED, 2)
    wall = time.perf_counter() - t0
    eng = runner.engine
    runner.stop()
    assert eng.clock.steps == pytest.approx(100 * wall, rel=0.1)
    assert eng.clock.time == pytest.approx(eng.clock.steps * 0.001, abs=1e-12)
    assert eng.clock.time == pytest.approx(0.1 * wall, rel=0.1)


def test_unachievable_rate_is_reported_and_dt_unchanged(qapp, scene):
    projectile(scene)
    stats = []
    runner = SimulationRunner()
    runner.statsUpdated.connect(stats.append)
    runner.start(make_engine(scene, delta_time=0.001, update_rate_hz=100000, duration=1000, max_catch_up_steps=3))
    time.sleep(1.2)
    runner.pause()
    runner.wait_until(lambda: runner.state == RunState.PAUSED, 2)
    eng = runner.engine
    runner.stop()
    qapp.processEvents()
    last = stats[-1]
    assert last.overrun and last.dropped_steps > 0
    assert last.achieved_hz < 0.9 * 100000
    assert eng.clock.delta_time == 0.001  # never enlarged to compensate
    assert eng.clock.time == pytest.approx(eng.clock.steps * 0.001)


def test_pause_resume_step_and_breakpoint(qapp, scene):
    projectile(scene)
    runner = SimulationRunner()
    eng = make_engine(scene, delta_time=0.01, duration=10, breakpoints=[0.25],
                      pacing=PacingMode.AS_FAST_AS_POSSIBLE)
    runner.start(eng)
    assert runner.wait_until(lambda: runner.state == RunState.PAUSED, 5)
    assert eng.clock.time == pytest.approx(0.25)  # stopped at the breakpoint
    n = eng.clock.steps
    runner.step(3)
    assert runner.wait_until(lambda: eng.clock.steps == n + 3, 5)
    assert runner.state == RunState.PAUSED
    runner.resume()
    assert runner.wait_until(lambda: runner.state == RunState.FINISHED, 10)
    assert eng.clock.time == pytest.approx(10.0)
    runner.stop()


def test_live_rate_and_dt_changes(qapp, scene):
    projectile(scene)
    runner = SimulationRunner()
    eng = make_engine(scene, delta_time=0.01, update_rate_hz=200, duration=1000)
    runner.start(eng)
    time.sleep(0.2)
    runner.set_delta_time(0.002)
    runner.set_update_rate(400)
    time.sleep(0.3)
    runner.pause()
    runner.wait_until(lambda: runner.state == RunState.PAUSED)
    runner.stop()
    assert eng.clock.delta_time == 0.002 and eng.settings.update_rate_hz == 400


def test_numerical_failure_is_detected(qapp, scene):
    s, e = scene
    b = e.add_new("body.point_mass")
    b.linear_velocity = [1e11, 0, 0]
    e.assign_process("dyn.gravity", TargetFilter(ids=[b.id]))
    eng = make_engine(scene, delta_time=100.0 / 10, duration=1e6)
    with pytest.raises(SimulationError, match="instability"):
        run_to_end(eng)
    errors = []
    runner = SimulationRunner()
    runner.errorOccurred.connect(errors.append)
    runner.start(make_engine(scene, delta_time=10.0, duration=1e6, pacing=PacingMode.AS_FAST_AS_POSSIBLE))
    assert runner.wait_until(lambda: runner.state == RunState.ERROR, 5)
    runner.stop()
    qapp.processEvents()
    assert errors and "instability" in errors[0]


# ---- integrator regression ---------------------------------------------------------------------------------

def test_free_fall_matches_closed_forms(scene):
    ball = projectile(scene, v0=(2.0, 0.0, 4.0))
    s, _ = scene
    p0 = np.array([0, 0, 10.0])
    dt, T = 0.01, 1.0
    exact = p0 + np.array([2.0, 0, 4.0]) * T + 0.5 * G * T * T
    res = {m: run_to_end(make_engine(scene, delta_time=dt, duration=T, integrator=m)).snapshot().origin[0]
           for m in ("explicit_euler", "semi_implicit_euler", "rk4")}
    np.testing.assert_allclose(res["rk4"], exact, atol=1e-12)
    # first-order schemes: z error is exactly ∓ g dt T / 2
    assert res["explicit_euler"][2] - exact[2] == pytest.approx(9.80665 * dt * T / 2, rel=1e-9)
    assert res["semi_implicit_euler"][2] - exact[2] == pytest.approx(-9.80665 * dt * T / 2, rel=1e-9)
    assert ball.id == s.assignments[0].targets.ids[0]


def _oscillator(scene, k=40.0):
    s, e = scene
    a = e.add_new("body.point_mass", position=[0, 0, 1])
    b = e.add_new("body.point_mass", position=[1.5, 0, 1])
    e.assign_process("dyn.spring", TargetFilter(ids=[a.id, b.id]), {"stiffness": k, "rest_length": 1.0})
    return a, b


def _separation_error(scene, dt, integrator, T=1.0, k=40.0):
    eng = run_to_end(make_engine(scene, delta_time=dt, duration=T, integrator=integrator))
    o = eng.snapshot().origin
    omega = math.sqrt(k / 0.5)  # reduced mass of two 1 kg bodies
    return abs((o[1, 0] - o[0, 0]) - (1.0 + 0.5 * math.cos(omega * T)))


def test_rk4_is_fourth_order(scene):
    _oscillator(scene)
    e1 = _separation_error(scene, 0.02, "rk4")
    e2 = _separation_error(scene, 0.01, "rk4")
    assert e1 / e2 == pytest.approx(16, rel=0.25)
    assert e2 < 5e-6  # ~ (ω Δt)^4 · amplitude for ω ≈ 8.9 rad/s, Δt = 0.01 s


def test_first_order_convergence(scene):
    _oscillator(scene)
    for m in ("explicit_euler", "semi_implicit_euler"):
        e1 = _separation_error(scene, 0.002, m)
        e2 = _separation_error(scene, 0.001, m)
        assert e1 / e2 == pytest.approx(2, rel=0.3), m


def test_symplectic_euler_keeps_oscillator_energy_bounded(scene):
    a, b = _oscillator(scene)

    def energy(eng):
        sn = eng.snapshot()
        d = sn.origin[1, 0] - sn.origin[0, 0]
        return 0.5 * 40 * (d - 1.0) ** 2 + 0.5 * (sn.v ** 2).sum()

    e0 = 0.5 * 40 * 0.25
    semi = run_to_end(make_engine(scene, delta_time=0.01, duration=50.0, integrator="semi_implicit_euler"))
    expl = run_to_end(make_engine(scene, delta_time=0.01, duration=50.0, integrator="explicit_euler"))
    assert abs(energy(semi) - e0) / e0 < 0.1
    assert energy(expl) > 2 * e0  # explicit Euler gains energy on oscillators


def test_constant_spin_integration(scene):
    s, e = scene
    c = e.add_new("basic.cube")
    c.angular_velocity = [0.0, 0.0, math.pi / 2]
    e.assign_process("kin.free_motion", TargetFilter(ids=[c.id]))
    snap = run_to_end(make_engine(scene, delta_time=0.01, duration=1.0)).snapshot()
    np.testing.assert_allclose(snap.q[0], [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)], atol=1e-12)
    np.testing.assert_allclose(snap.origin[0], [0, 0, 0], atol=1e-12)
