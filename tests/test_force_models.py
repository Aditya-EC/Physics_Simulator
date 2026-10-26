"""Phase 7 force models: central gravity, region force field, anchored spring-damper."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim.processes.process_base import TargetFilter
from psim.recording.recording import Recording

from test_recording import make_engine, run


def test_central_gravity_circular_orbit(scene):
    s, e = scene
    sat = e.add_new("body.point_mass", position=[2.0, 0, 0])
    sat.linear_velocity = [0, math.sqrt(4.0 / 2.0), 0]          # v = √(μ/r)
    e.assign_process("dyn.central_gravity", TargetFilter(ids=[sat.id]), {"mu": 4.0})
    rec = Recording()
    period = 2 * math.pi * math.sqrt(2.0 ** 3 / 4.0)
    run(make_engine(scene, rec, delta_time=0.002, duration=period, integrator="rk4"))
    x = rec.column((sat.id, "com"))
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 2.0, rtol=1e-6)
    np.testing.assert_allclose(x[-1], [2.0, 0, 0], atol=1e-4)    # back after one period
    acc = rec.column((sat.id, "acc"))
    np.testing.assert_allclose(np.linalg.norm(acc, axis=1), 1.0, rtol=1e-6)   # μ/r²


def test_central_gravity_softening_keeps_centre_finite(scene):
    s, e = scene
    p = e.add_new("body.point_mass", position=[0, 0, 0])
    e.assign_process("dyn.central_gravity", TargetFilter(ids=[p.id]), {"mu": 1.0, "softening": 0.1})
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.01, duration=0.1))
    assert np.all(np.isfinite(rec.column((p.id, "acc"))))
    np.testing.assert_allclose(rec.column((p.id, "com"))[-1], 0, atol=1e-12)


def test_force_field_acts_only_inside_its_region(scene):
    s, e = scene
    m = e.add_new("body.point_mass", position=[-2.0, 0, 0])
    e.set_property([m.id], "mass_input", 2.0)
    m.linear_velocity = [1.0, 0, 0]
    e.assign_process("dyn.force_field", TargetFilter(ids=[m.id]),
                     {"force": [0, 0, 4.0], "region_min": [-1, -1, -1], "region_max": [1, 1, 5]})
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.001, duration=4.0, integrator="rk4"))
    t = rec.times()
    x = rec.column((m.id, "com"))
    vz = rec.column((m.id, "v"))[:, 2]
    assert np.all(x[t < 0.99, 2] == 0)                 # before it enters the box
    # 2 s inside at F/m = 2 m/s²: vz ≈ 4 m/s, then constant after leaving
    assert vz[-1] == pytest.approx(4.0, abs=0.01)
    late = t > 3.05
    np.testing.assert_allclose(vz[late], vz[-1], atol=1e-12)


def test_force_field_without_region_applies_everywhere(scene):
    s, e = scene
    m = e.add_new("body.point_mass", position=[50.0, 0, 0])
    e.assign_process("dyn.force_field", TargetFilter(ids=[m.id]),
                     {"acceleration": [1.0, 0, 0], "use_region": False})
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.01, duration=1.0))
    assert rec.column((m.id, "com"))[-1, 0] == pytest.approx(50.5, abs=0.006)   # explicit Euler: ½a t² + O(Δt)


def test_anchor_spring_vertical_oscillation_period(scene):
    s, e = scene
    m = e.add_new("body.point_mass", position=[0, 0, 0.5])
    e.set_property([m.id], "mass_input", 0.5)
    k, length = 20.0, 1.0
    e.assign_process("dyn.anchor_spring", TargetFilter(ids=[m.id]),
                     {"anchor": [0, 0, 2.0], "stiffness": k, "rest_length": length})
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.001, duration=3.0, integrator="rk4"))
    z = rec.column((m.id, "com"))[:, 2]
    t = rec.times()
    # equilibrium at z = 1, amplitude 0.5 (starts at the bottom): z = 1 − ½cos(ωt)
    omega = math.sqrt(k / 0.5)
    np.testing.assert_allclose(z, 1 - 0.5 * np.cos(omega * t), atol=1e-6)


def test_anchor_spring_off_centre_attachment_produces_torque(scene):
    s, e = scene
    box = e.add_new("body.rigid_box", position=[0, 0, 0], params={"length": 1, "width": 0.2, "height": 0.2})
    e.assign_process("dyn.anchor_spring", TargetFilter(ids=[box.id]),
                     {"anchor": [0.5, 0, 1.0], "attachment": [0.5, 0, 0], "stiffness": 10.0, "rest_length": 0.0,
                      "damping": 0.0})
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.001, duration=0.05, integrator="rk4"))
    w = rec.column((box.id, "w"))
    assert w[-1, 1] < -1e-3                    # pulled up at +X end: rotation about −Y
    assert abs(w[-1, 0]) < 1e-12 and abs(w[-1, 2]) < 1e-12
