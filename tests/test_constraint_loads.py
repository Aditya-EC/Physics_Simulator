"""Contact and joint forces / torques on simulated bodies (recorded loads and their plot quantities)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim.processes.process_base import TargetFilter
from psim.recording.recording import Recording
from psim.recording.variables import Evaluator, Variable, VariableError, object_quantities

from test_physics import _box, _gravity, _ground
from test_recording import make_engine, run

G = 9.80665


def test_resting_box_carries_its_weight_on_contacts(scene):
    s, e = scene
    _ground(e)
    box = _box(e, [0.3, -0.2, 0.2])
    _gravity(e, box)
    rec = Recording(interval_steps=1)
    run(make_engine(scene, rec, delta_time=0.005, duration=1.5))
    m = box.mass_properties()["mass"]
    late = rec.times() > 1.0
    fc = rec.column((box.id, "f_contact"))[late]
    np.testing.assert_allclose(fc.mean(axis=0), [0, 0, m * G], atol=1e-3 * m * G)
    np.testing.assert_allclose(fc, np.tile([0, 0, m * G], (len(fc), 1)), atol=0.02 * m * G)
    np.testing.assert_allclose(rec.column((box.id, "t_contact"))[late], 0.0, atol=1e-3 * m * G * 0.2)
    assert not np.any(rec.column((box.id, "f_joint"))) and not np.any(rec.column((box.id, "t_joint")))
    ev = Evaluator(rec, s)
    z = ev.values(Variable(box.id, "contact_force", "z"))
    assert np.mean(z[late]) == pytest.approx(m * G, rel=1e-3)
    assert {"contact_force", "contact_torque", "joint_force", "joint_torque"} <= set(object_quantities(rec, box.id))


def test_pendulum_rod_tension_balances_the_motion(scene):
    """A point mass released horizontally on a rod: m a = m g + F_joint at every sample, 3 m g at the bottom."""
    s, e = scene
    L = 1.0
    ball = e.add_new("body.point_mass", position=[L, 0, 2.0])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    e.add_joint("distance", ball.id, None, [L, 0, 2.0], anchor_world_b=[0, 0, 2.0])
    rec = Recording(interval_steps=1)
    run(make_engine(scene, rec, delta_time=0.0005, duration=0.6, integrator="velocity_verlet"))
    m = ball.mass_properties()["mass"]
    fj = rec.column((ball.id, "f_joint"))
    acc = rec.column((ball.id, "acc"))
    np.testing.assert_allclose(m * acc, fj + [0, 0, -m * G], atol=1e-9 * m * G)
    x = rec.column((ball.id, "com"))
    r = x - [0, 0, 2.0]
    along = np.einsum("ij,ij->i", fj, r) / np.linalg.norm(r, axis=1)
    assert np.all(along[1:] < 0)                                       # the rod pulls towards the pivot
    bottom = np.argmin(x[:, 2])
    assert 0 < bottom < len(x) - 1                                     # passed the lowest point
    assert np.linalg.norm(fj[bottom]) == pytest.approx(3 * m * G, rel=0.01)
    assert not np.any(rec.column((ball.id, "f_contact")))


def test_joint_loads_on_two_bodies_are_equal_and_opposite(scene):
    s, e = scene
    a = e.add_new("body.rigid_box", position=[0, 0, 1.0], params={"length": 0.4, "width": 0.4, "height": 0.4})
    b = e.add_new("body.rigid_box", position=[0.6, 0, 1.0], params={"length": 0.4, "width": 0.4, "height": 0.4})
    e.assign_process("dyn.gravity", TargetFilter(ids=[b.id]))
    e.assign_process("dyn.gravity", TargetFilter(ids=[a.id]), {"g": [0.0, 0.0, 0.0]})
    e.add_joint("ball", a.id, b.id, [0.3, 0, 1.0])
    rec = Recording(interval_steps=1)
    run(make_engine(scene, rec, delta_time=0.002, duration=0.3))
    fa, fb = rec.column((a.id, "f_joint")), rec.column((b.id, "f_joint"))
    assert np.max(np.linalg.norm(fa, axis=1)) > 1.0
    np.testing.assert_allclose(fa, -fb, atol=1e-9)
    # torques about each COM: r × F with the anchor offset of each body
    ta, tb = rec.column((a.id, "t_joint")), rec.column((b.id, "t_joint"))
    assert np.max(np.linalg.norm(ta, axis=1)) > 0.1 and np.max(np.linalg.norm(tb, axis=1)) > 0.1


def test_spring_load_counts_as_joint_force(scene):
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 1.0])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    k = 400.0
    m = ball.mass_properties()["mass"]
    e.add_joint("distance", ball.id, None, [0, 0, 1.0], anchor_world_b=[0, 0, 2.0], spring_stiffness=k,
                spring_damping=2 * math.sqrt(k * m))
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.002, duration=3.0))
    np.testing.assert_allclose(rec.column((ball.id, "f_joint"))[-1], [0, 0, m * G], rtol=1e-4, atol=1e-9)


def test_runs_without_contacts_or_joints_have_no_loads(scene):
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 1.0])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.01, duration=0.1))
    assert "contact_force" not in object_quantities(rec, ball.id)
    with pytest.raises(VariableError, match="Contact force"):
        Evaluator(rec, s).values(Variable(ball.id, "contact_force", "z"))


@pytest.mark.parametrize("integrator", ["semi_implicit_euler", "velocity_verlet", "rk4"])
@pytest.mark.parametrize("kind", ["rope", "limit"])
def test_one_sided_joint_rows_hold_steadily_with_every_integrator(scene, integrator, kind):
    """A box hanging on a taut rope or resting on a slider limit: constant load m g, no chatter (higher-order
    integrators used to slacken the row every other step, the load alternating between 0 and 2 m g)."""
    s, e = scene
    box = e.add_new("body.rigid_box", position=[0, 0, 1.0], params={"length": 0.4, "width": 0.4, "height": 0.4})
    e.assign_process("dyn.gravity", TargetFilter(ids=[box.id]))
    if kind == "rope":
        e.add_joint("distance", box.id, None, [0, 0, 1.2], anchor_world_b=[0, 0, 2.2], rope=True)
    else:
        e.add_joint("prismatic", box.id, None, [0, 0, 1.0], [0, 0, 1], limit_enabled=True, lower=0.0, upper=1.0)
    rec = Recording(interval_steps=1)
    run(make_engine(scene, rec, delta_time=0.004, duration=1.0, integrator=integrator))
    m = box.mass_properties()["mass"]
    np.testing.assert_allclose(rec.column((box.id, "f_joint"))[50:], np.tile([0, 0, m * G], (201, 1)),
                               rtol=1e-6, atol=1e-6 * m * G)
    assert np.ptp(rec.column((box.id, "com"))[50:, 2]) < 1e-9
