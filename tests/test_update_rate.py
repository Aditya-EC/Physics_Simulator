"""Simulation timing building blocks (Phase 3 interfaces): fixed Δt, update rate, duration policy.

The full engine arrives in Phase 3; these tests pin down the timing semantics it must follow:
Δt and update rate are independent, the clock advances by exactly Δt per step, duration is
measured in simulated time, and slow hardware never changes Δt.
"""
from __future__ import annotations

import pytest

from psim.simulation.clock import SimulationClock, UpdateRateController, run_steps
from psim.simulation.settings import DurationPolicy, PacingMode, SimulationSettings


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_delta_time_is_independent_of_update_rate():
    s = SimulationSettings(delta_time=0.001, update_rate_hz=100.0)
    assert s.delta_time == 0.001 and s.update_rate_hz == 100.0
    assert s.time_ratio == pytest.approx(0.1)  # 0.1 simulated s per real s
    s2 = SimulationSettings(delta_time=0.01, update_rate_hz=100.0)
    assert s2.time_ratio == pytest.approx(1.0)


def test_expected_steps_and_duration_policies():
    assert SimulationSettings(delta_time=0.01, duration=10.0).expected_steps() == 1000
    for policy, steps in ((DurationPolicy.FINAL_PARTIAL_STEP, 4), (DurationPolicy.NEXT_FULL_STEP, 4),
                          (DurationPolicy.LAST_FULL_STEP, 3)):
        s = SimulationSettings(delta_time=0.3, duration=1.0, duration_policy=policy)
        assert s.expected_steps() == steps
        dts = []
        clock = run_steps(s, lambda t, dt: dts.append(dt))
        assert clock.steps == steps == len(dts)
        if policy == DurationPolicy.FINAL_PARTIAL_STEP:
            assert clock.time == pytest.approx(1.0) and dts[-1] == pytest.approx(0.1)
        elif policy == DurationPolicy.NEXT_FULL_STEP:
            assert clock.time == pytest.approx(1.2)
        else:
            assert clock.time == pytest.approx(0.9)


def test_clock_advances_exactly_by_delta_time_without_drift():
    clock = SimulationClock(0.001)
    for _ in range(1_000_000):
        clock.advance()
    assert clock.steps == 1_000_000
    assert clock.time == 1000.0  # computed as steps * Δt, so no accumulated rounding error


def test_changing_delta_time_applies_to_subsequent_steps():
    clock = SimulationClock(0.01)
    for _ in range(100):
        clock.advance()
    clock.set_delta_time(0.001)
    for _ in range(500):
        clock.advance()
    assert clock.time == pytest.approx(1.0 + 0.5)
    assert clock.steps == 600


def test_run_steps_calls_every_step_with_fixed_dt():
    s = SimulationSettings(delta_time=0.01, duration=10.0, pacing=PacingMode.AS_FAST_AS_POSSIBLE)
    times = []
    clock = run_steps(s, lambda t, dt: times.append((t, dt)))
    assert len(times) == 1000 and clock.time == pytest.approx(10.0)
    assert all(dt == 0.01 for _, dt in times)
    assert times[500][0] == pytest.approx(5.0)


def test_update_rate_controller_paces_steps():
    fake = FakeClock()
    ctl = UpdateRateController(100.0, max_catch_up_steps=10, now=fake)
    total = 0
    for _ in range(1000):  # one real second in 1 ms ticks
        fake.t += 0.001
        n = ctl.steps_due()
        for _ in range(n):
            ctl.record_step()
        total += n
    assert total in (99, 100, 101)
    assert ctl.achieved_rate() == pytest.approx(100.0, rel=0.03)
    assert not ctl.stats().overrun


def test_update_rate_controller_reports_unachievable_rate_and_limits_catch_up():
    fake = FakeClock()
    ctl = UpdateRateController(1000.0, max_catch_up_steps=5, now=fake)
    fake.t += 0.1  # the GUI/engine stalled for 100 ms (100 steps due)
    n = ctl.steps_due()
    assert n == 5  # bounded catch-up keeps the application responsive
    assert ctl.dropped_steps == 95
    assert ctl.stats().overrun


def test_settings_validation_and_serialization():
    s = SimulationSettings(delta_time=-1, update_rate_hz=0, duration=0)
    errors = s.validate()
    assert len(errors) == 3
    good = SimulationSettings(delta_time=0.002, update_rate_hz=500, duration=3, pacing=PacingMode.AS_FAST_AS_POSSIBLE)
    assert SimulationSettings.from_dict(good.to_dict()) == good


def test_platform_timing_helpers_are_safe_everywhere():
    """macOS: a run declares a latency-critical activity (no App Nap throttling) and ends it again; Windows: 1 ms
    timer resolution. Elsewhere both are no-ops."""
    import sys
    import threading

    from psim.simulation.runner import _MacOSActivity, _WindowsTimerResolution

    seen = {}

    def worker():                                   # the runner enters them on its worker thread
        with _WindowsTimerResolution() as win, _MacOSActivity() as mac:
            seen["win"], seen["mac"] = win._ok, mac._token
        seen["after"] = mac._token

    t = threading.Thread(target=worker)
    t.start()
    t.join(10)
    assert seen["win"] == (sys.platform == "win32")
    assert bool(seen["mac"]) == (sys.platform == "darwin")
    assert seen["after"] is None
