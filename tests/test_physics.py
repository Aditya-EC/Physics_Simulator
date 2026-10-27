"""Rigid-body solver: collisions (restitution, friction, resting, stacking), joints, ownership and the PGS core."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim import numerics
from psim.processes.process_base import TargetFilter
from psim.processes.process_scheduler import build_plan
from psim.recording.recording import Recording

from test_recording import make_engine, run

G = 9.80665


def _ground(e, tilt: float = 0.0, mu: float = 0.5):
    g = e.add_new("basic.plane", params={"length": 20, "width": 20})
    e.set_property([g.id], "physics.collider", "auto")
    e.set_property([g.id], "physics.friction", mu)
    if tilt:
        e.scene.get(g.id).set_orientation(numerics.quat_from_axis_angle([0, 1, 0], tilt))
    return g


def _box(e, pos, size=0.4, mu=0.5):
    b = e.add_new("body.rigid_box", position=pos, params={"length": size, "width": size, "height": size})
    e.set_property([b.id], "physics.friction", mu)
    return b


def _gravity(e, *objs):
    e.assign_process("dyn.gravity", TargetFilter(ids=[o.id for o in objs]))


def test_ball_bounces_with_restitution(scene):
    s, e = scene
    _ground(e)
    ball = e.add_new("basic.sphere", position=[0, 0, 1.0], params={"radius": 0.1})
    e.set_property([ball.id], "physics.collider", "auto")
    e.set_property([ball.id], "physics.restitution", 0.5)
    _gravity(e, ball)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.001, duration=1.5))
    z = rec.column((ball.id, "com"))[:, 2]
    peaks = [z[i] for i in range(1, len(z) - 1) if z[i] > z[i - 1] and z[i] >= z[i + 1]]
    assert peaks[0] == pytest.approx(0.1 + 0.25 * 0.9, abs=0.01)       # e² of the drop height
    assert z.min() > 0.1 - 0.005                                       # no tunnelling through the ground
    kinds = [ev.kind for ev in rec.event_list()]
    assert "contact" in kinds


def test_box_rests_without_drift_and_net_force_balances(scene):
    s, e = scene
    _ground(e)
    box = _box(e, [0, 0, 0.21])
    _gravity(e, box)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.005, duration=2.0))
    z = rec.column((box.id, "com"))[:, 2]
    assert z[-1] == pytest.approx(0.2, abs=0.003)
    assert np.ptp(z[len(z) // 2:]) < 1e-6 and np.abs(rec.column((box.id, "w"))).max() < 1e-5
    # gravity and the contact force cancel: the recorded total acceleration of the resting box is zero
    np.testing.assert_allclose(rec.column((box.id, "acc"))[-50:], 0, atol=1e-4)


def test_sliding_friction_stops_at_v2_over_2_mu_g(scene):
    s, e = scene
    _ground(e, mu=0.5)
    box = _box(e, [0, 0, 0.2], mu=0.5)
    e.set_property([box.id], "linear_velocity", [3.0, 0, 0])
    _gravity(e, box)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.002, duration=1.5))
    x = rec.column((box.id, "com"))[:, 0]
    assert x[-1] == pytest.approx(9 / (2 * 0.5 * G), rel=0.01)
    assert abs(rec.column((box.id, "v"))[-1, 0]) < 1e-6 and np.abs(rec.column((box.id, "w"))).max() < 1e-6


@pytest.mark.parametrize("tan_theta,slides", [(0.3, False), (0.7, True)])
def test_block_on_an_incline(scene, tan_theta, slides):
    s, e = scene
    theta = math.atan(tan_theta)
    _ground(e, tilt=theta, mu=0.5)
    n = np.array([math.sin(theta), 0, math.cos(theta)])
    box = _box(e, list(n * 0.2))
    s.get(box.id).set_orientation(numerics.quat_from_axis_angle([0, 1, 0], theta))
    _gravity(e, box)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.002, duration=1.0))
    p = rec.column((box.id, "com"))
    moved = float(np.linalg.norm(p[-1] - p[0]))
    expected = max(G * (math.sin(theta) - 0.5 * math.cos(theta)), 0.0) / 2   # a·t²/2 at t = 1 s
    assert moved == pytest.approx(expected, abs=0.01) and (moved > 0.1) == slides


@pytest.mark.parametrize("pushed", [False, True])
def test_static_friction_holds_until_the_block_slides(scene, pushed):
    """tan θ = 0.5 lies between μk = 0.4 and μs = 0.6: at rest the block sticks; pushed, it slides with
    a = g (sin θ − μk cos θ)."""
    s, e = scene
    theta = math.atan(0.5)
    g = _ground(e, tilt=theta, mu=0.4)
    n = np.array([math.sin(theta), 0, math.cos(theta)])
    t = np.array([math.cos(theta), 0, -math.sin(theta)])                    # down the slope
    box = _box(e, list(n * 0.2), mu=0.4)
    for o in (g, box):
        e.set_property([o.id], "physics.static_friction", 0.6)
    s.get(box.id).set_orientation(numerics.quat_from_axis_angle([0, 1, 0], theta))
    if pushed:
        e.set_property([box.id], "linear_velocity", list(t * 1.0))
    _gravity(e, box)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.002, duration=1.0))
    p = rec.column((box.id, "com"))
    moved = float((p[-1] - p[0]) @ t)
    if pushed:
        a = G * (math.sin(theta) - 0.4 * math.cos(theta))
        assert moved == pytest.approx(1.0 + a / 2, abs=0.01)
    else:
        assert abs(moved) < 1e-3                                             # μk alone (0.4) would let it slide


@pytest.mark.parametrize("b", [0.0, 0.002])
def test_rolling_resistance_decelerates_a_rolling_ball(scene, b):
    """A rolling sphere with rolling resistance length b slows at a = (5/7) b g / r and keeps rolling."""
    s, e = scene
    r = 0.1
    g = _ground(e)
    ball = e.add_new("basic.sphere", position=[0, 0, r], params={"radius": r})
    e.set_property([ball.id], "physics.collider", "auto")
    e.set_property([ball.id], "linear_velocity", [1.0, 0, 0])
    e.set_property([ball.id], "angular_velocity", [0, 1.0 / r, 0])          # rolling without slipping
    e.set_property([g.id], "physics.rolling_resistance", b)                  # on either body
    _gravity(e, ball)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.002, duration=2.0))
    v = rec.column((ball.id, "v"))[-1, 0]
    w = rec.column((ball.id, "w"))[-1, 1]
    assert v == pytest.approx(1.0 - 5 / 7 * b * G / r * 2.0, abs=2e-3)
    assert w * r == pytest.approx(v, abs=1e-3)                                # still rolling


def test_spinning_resistance_slows_a_spinning_ball(scene):
    """A ball spinning in place about the vertical: torque b·m·g, so α = b m g / I."""
    s, e = scene
    r, b = 0.1, 0.002
    g = _ground(e)
    ball = e.add_new("basic.sphere", position=[0, 0, r], params={"radius": r})
    e.set_property([ball.id], "physics.collider", "auto")
    e.set_property([ball.id], "angular_velocity", [0, 0, 10.0])
    e.set_property([ball.id], "physics.spinning_resistance", b)
    _gravity(e, ball)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.002, duration=1.0))
    w = rec.column((ball.id, "w"))[-1]
    mp = ball.mass_properties()                                               # (a faceted sphere: I ≈ 0.4 m r²)
    assert w[2] == pytest.approx(10.0 - b * mp["mass"] * G / mp["inertia"][2, 2] * 1.0, abs=0.005)
    assert np.linalg.norm(rec.column((ball.id, "com"))[-1, :2]) < 1e-6          # spins in place
    assert g


@pytest.mark.parametrize("integrator", ["semi_implicit_euler", "explicit_euler", "velocity_verlet", "midpoint",
                                        "rk4", "dopri5"])
def test_resting_contact_holds_with_every_integrator(scene, integrator):
    """A box resting flat, and one held by static friction on a slope, stay in contact at every step and do not
    move, whatever the integrator (higher-order integrators used to let resting bodies chatter)."""
    s, e = scene
    flat = _box(e, [0, 0, 0.2])
    theta = math.atan(0.5)
    ramp = e.add_new("basic.cuboid", position=[3, 0, 1.0], params={"length": 3, "width": 1, "height": 0.1})
    e.set_property([ramp.id], "physics.collider", "auto")
    n = np.array([math.sin(theta), 0, math.cos(theta)])
    slope = _box(e, list(np.array([3, 0, 1.0]) + n * 0.25), mu=0.3)
    for o in (ramp, slope):
        e.set_property([o.id], "orientation", list(numerics.quat_from_axis_angle([0, 1, 0], theta)))
        e.set_property([o.id], "physics.static_friction", 0.6)
    e.set_property([ramp.id], "physics.friction", 0.3)
    _ground(e)
    _gravity(e, flat, slope)
    rec = Recording()
    eng = make_engine(scene, rec, delta_time=0.004, duration=1.0, integrator=integrator)
    ph = eng.physics
    boxes = {ph.index[flat.id], ph.index[slope.id]}
    missing = []
    orig = ph._contacts

    def counting(*args, **kwargs):
        d = orig(*args, **kwargs)
        touching = set() if d is None else set(np.r_[d["ia"], d["ib"]].tolist())
        missing.append(bool(boxes - touching))
        return d
    ph._contacts = counting
    run(eng)
    assert not any(missing), [k for k, x in enumerate(missing) if x][:10]      # in contact at every step
    tol = 1e-3 if integrator == "explicit_euler" else 1e-6     # explicit Euler: a slow creep (0.2 mm/s)
    for b in (flat, slope):
        p = rec.column((b.id, "com"))
        assert np.linalg.norm(p[-1] - p[0]) < tol


def test_stack_of_boxes_is_stable(scene):
    s, e = scene
    _ground(e)
    boxes = [_box(e, [0, 0, 0.2 + 0.401 * k]) for k in range(3)]
    _gravity(e, *boxes)
    rec = Recording(interval_steps=10)
    run(make_engine(scene, rec, delta_time=0.005, duration=2.0))
    for k, b in enumerate(boxes):
        p = rec.column((b.id, "com"))
        assert p[-1, 2] == pytest.approx(0.2 + 0.4 * k, abs=0.005)
        assert np.abs(p[:, :2]).max() < 0.01


def test_collisions_conserve_momentum_and_elastic_energy(scene):
    s, e = scene
    a = e.add_new("basic.sphere", position=[-1, 0.15, 0], params={"radius": 0.2})
    b = e.add_new("basic.sphere", position=[1, 0, 0], params={"radius": 0.3})
    for o, m in ((a, 1.0), (b, 3.0)):
        for path, value in (("physics.collider", "auto"), ("physics.restitution", 1.0), ("physics.friction", 0.0),
                            ("mass_mode", "mass"), ("mass_input", m)):
            e.set_property([o.id], path, value)
    e.set_property([a.id], "linear_velocity", [2.0, 0, 0])
    e.assign_process("kin.free_motion", TargetFilter(ids=[a.id, b.id]))
    sn = run(make_engine(scene, Recording(), delta_time=0.001, duration=1.5)).snapshot()
    np.testing.assert_allclose(1.0 * sn.v[0] + 3.0 * sn.v[1], [2.0, 0, 0], atol=1e-9)
    energy = 0.5 * sn.v[0] @ sn.v[0] + 1.5 * sn.v[1] @ sn.v[1]
    assert energy == pytest.approx(2.0, rel=1e-6)
    assert abs(sn.v[1][1]) > 0.05   # oblique impact deflects the target


def test_kinematic_platform_carries_a_box(scene):
    s, e = scene
    plat = e.add_new("body.rigid_box", params={"length": 4, "width": 4, "height": 0.2})
    e.assign_process("kin.oscillation", TargetFilter(ids=[plat.id]), {"direction": [0, 0, 1], "amplitude": 0.1,
                                                                    "frequency": 0.5})
    box = _box(e, [0, 0, 0.3])
    _gravity(e, box)
    plan = build_plan(s, s.assignments)
    assert any("treats it as kinematic" in d.message for d in plan.diagnostics)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.002, duration=2.0))
    gap = rec.column((box.id, "com"))[:, 2] - rec.column((plat.id, "com"))[:, 2]
    assert np.abs(gap - 0.3).max() < 0.003


def _period(rec, oid, pivot):
    x, t = rec.column((oid, "com"))[:, 0] - pivot[0], rec.times()
    crossings = [t[i] for i in range(1, len(x)) if x[i - 1] > 0 >= x[i]]
    return crossings[1] - crossings[0]


def test_physical_pendulum_on_a_revolute_joint(scene):
    s, e = scene
    r = 0.05
    bob = e.add_new("basic.sphere", position=[math.sin(0.1), 0, 2.0 - math.cos(0.1)], params={"radius": r})
    _gravity(e, bob)
    e.add_joint("revolute", bob.id, None, [0, 0, 2.0], [0, 1, 0])
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.001, duration=4.0))
    p = rec.column((bob.id, "com"))
    np.testing.assert_allclose(np.linalg.norm(p - [0, 0, 2.0], axis=1), 1.0, atol=1e-4)
    i_pivot = 0.4 * r * r + 1.0     # (I_cm + m L²) / m for a solid sphere on a 1 m arm
    expected = 2 * math.pi * math.sqrt(i_pivot / G) * (1 + 0.1 ** 2 / 16)
    assert _period(rec, bob.id, [0, 0, 2.0]) == pytest.approx(expected, rel=0.01)


def test_point_mass_on_a_rod(scene):
    s, e = scene
    bob = e.add_new("body.point_mass", position=[math.sin(0.1), 0, 2.0 - math.cos(0.1)])
    _gravity(e, bob)
    e.add_joint("distance", bob.id, None, list(bob.position), anchor_world_b=[0, 0, 2.0])
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.001, duration=4.0))
    np.testing.assert_allclose(np.linalg.norm(rec.column((bob.id, "com")) - [0, 0, 2.0], axis=1), 1.0, atol=1e-4)
    expected = 2 * math.pi * math.sqrt(1.0 / G) * (1 + 0.1 ** 2 / 16)
    assert _period(rec, bob.id, [0, 0, 2.0]) == pytest.approx(expected, rel=0.01)
    # a hinge on a point mass cannot turn it: the scheduler explains why it will not move
    e.add_joint("revolute", bob.id, None, [0, 0, 2.0], [0, 1, 0])
    assert any("no rotational inertia" in d.message for d in build_plan(s, s.assignments).diagnostics)


@pytest.mark.parametrize("body", ["body.point_mass", "body.rigid_box"])
def test_prismatic_slider(scene, body):
    """Anchored at the body's centre, so a point mass can slide too."""
    s, e = scene
    b = e.add_new(body, position=[0, 0, 1])
    _gravity(e, b)
    axis = np.array([1, 0, -1]) / math.sqrt(2)
    e.add_joint("prismatic", b.id, None, [0, 0, 1], axis, limit_enabled=True, lower=-1.0, upper=2.0)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.002, duration=1.0))
    d = rec.column((b.id, "com")) - rec.column((b.id, "com"))[0]
    along = d @ axis
    assert np.linalg.norm(d - np.outer(along, axis), axis=1).max() < 1e-6
    assert along.max() == pytest.approx(2.0, abs=0.01)                  # stopped by the upper limit
    t_hit = math.sqrt(2 * 2.0 / (G / math.sqrt(2)))
    assert along[np.searchsorted(rec.times(), 0.5 * t_hit)] == pytest.approx(G / math.sqrt(2) / 2 * (0.5 * t_hit) ** 2,
                                                                          rel=0.02)


def test_fixed_joint_and_motor_and_limits(scene):
    s, e = scene
    a1, a2 = _box(e, [0, 0, 0]), _box(e, [0.5, 0, 0])
    e.set_property([a1.id], "angular_velocity", [0.3, 1.0, 2.0])
    e.set_property([a1.id], "linear_velocity", [1, 0, 0])
    e.assign_process("kin.free_motion", TargetFilter(ids=[a1.id, a2.id]))
    e.add_joint("fixed", a1.id, a2.id, [0.25, 0, 0])
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.002, duration=2.0, integrator="rk4"))
    q1 = rec.column((a1.id, "q"))
    rel = np.array([np.asarray(numerics.quat_to_matrix(q)).T @ (pb - pa) for q, pa, pb in
                    zip(q1, rec.column((a1.id, "origin")), rec.column((a2.id, "origin")))])
    np.testing.assert_allclose(rel, np.tile([0.5, 0, 0], (len(rel), 1)), atol=1e-4)
    # a motorised hinge reaches its target speed; a limited hinge stays inside its range
    s2, e2 = scene[0].__class__(), None
    from psim.scene.editing import SceneEditor
    e2 = SceneEditor(s2)
    arm = _box(e2, [0.5, 0, 1])
    e2.assign_process("kin.free_motion", TargetFilter(ids=[arm.id]))
    # the 502 kg steel cube has I ≈ 139 kg·m² about the hinge: 1e5 N·m reaches 2 rad/s in a few steps
    e2.add_joint("revolute", arm.id, None, [0, 0, 1], [0, 1, 0], motor_enabled=True, motor_speed=2.0,
                 motor_max=1e5)
    sn = run(make_engine((s2, e2), Recording(), delta_time=0.002, duration=0.5)).snapshot()
    np.testing.assert_allclose(sn.w[0], [0, 2.0, 0], atol=1e-6)
    s3 = scene[0].__class__()
    e3 = SceneEditor(s3)
    arm = _box(e3, [0.5, 0, 1])
    _gravity(e3, arm)
    e3.add_joint("revolute", arm.id, None, [0, 0, 1], [0, 1, 0], limit_enabled=True, lower=-0.5, upper=0.5)
    rec = Recording()
    run(make_engine((s3, e3), rec, delta_time=0.002, duration=2.0))
    p = rec.column((arm.id, "com"))
    angle = np.arctan2(-(p[:, 2] - 1), p[:, 0])
    assert angle.max() == pytest.approx(0.5, abs=0.01) and angle.min() > -0.01


def test_rope_is_slack_until_taut(scene):
    s, e = scene
    bob = e.add_new("body.point_mass", position=[0, 0, 1.5])
    _gravity(e, bob)
    e.add_joint("distance", bob.id, None, [0, 0, 1.5], anchor_world_b=[0, 0, 2.0], length=1.0, rope=True)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.001, duration=1.0))
    z = rec.column((bob.id, "com"))[:, 2]
    t_taut = math.sqrt(2 * 0.5 / G)
    # free fall while slack (semi-implicit Euler is O(Δt) accurate)
    assert z[np.searchsorted(rec.times(), 0.5 * t_taut)] == pytest.approx(1.5 - G / 2 * (0.5 * t_taut) ** 2, abs=3e-3)
    assert z.min() > 1.0 - 0.002 and z[-1] == pytest.approx(1.0, abs=0.002)


def test_compound_assembly_collides_as_one_body(scene):
    s, e = scene
    _ground(e)
    asm = e.add_new("group.assembly", position=[0, 0, 1])
    c1 = e.add_new("basic.cube", position=[-0.4, 0, 1], params={"size": 0.2})
    c2 = e.add_new("basic.cube", position=[0.4, 0, 1], params={"size": 0.2})
    e.reparent([c1.id, c2.id], asm.id)
    for c in (c1, c2):
        e.set_property([c.id], "physics.collider", "auto")
    _gravity(e, asm)
    sn = run(make_engine(scene, Recording(), delta_time=0.005, duration=2.0)).snapshot()
    assert sn.origin[0][2] == pytest.approx(0.1, abs=0.004)              # cubes resting on the ground


def test_ownership_rules(scene):
    s, e = scene
    ground = _ground(e)
    _gravity(e, ground)          # a plane collider may not move
    plan = build_plan(s, s.assignments)
    assert any("must be static" in d.message for d in plan.errors)
    s.assignments = []
    box = _box(e, [0, 0, 1])
    e.assign_process("lib.euler_velocity_update", TargetFilter(ids=[box.id]))
    e.assign_process("lib.euler_position_update", TargetFilter(ids=[box.id]))
    plan = build_plan(s, s.assignments)
    assert plan.ok and plan.physics.bodies[box.id].role == "kinematic"
    s.assignments = []
    other = _box(e, [2, 0, 1])
    e.add_joint("ball", box.id, other.id, [1, 0, 1])
    plan = build_plan(s, s.assignments)
    assert any("cannot move anything" in d.message for d in plan.diagnostics)
    sphere = e.add_new("basic.sphere")
    assert sphere.id not in build_plan(s, s.assignments).physics.bodies   # mass alone does not make a collider


def test_physics_can_be_disabled(scene):
    s, e = scene
    _ground(e)
    ball = e.add_new("basic.sphere", position=[0, 0, 1.0], params={"radius": 0.1})
    e.set_property([ball.id], "physics.collider", "auto")
    _gravity(e, ball)
    sn = run(make_engine(scene, None, delta_time=0.01, duration=1.0, physics_enabled=False)).snapshot()
    assert sn.origin[0][2] < 0     # falls through: no collision stage


def test_native_and_fallback_pgs_agree():
    rng = np.random.default_rng(3)
    n, m = 4, 14
    inv_m = rng.uniform(0.2, 2.0, n)
    inv_i = np.array([np.diag(rng.uniform(0.5, 5.0, 3)) for _ in range(n)])
    ia = rng.integers(-1, n, m)
    ib = np.array([(a + 1) % n for a in ia])
    J = rng.normal(size=(m, 12))
    rhs = rng.normal(size=m)
    lo = np.where(rng.random(m) < 0.5, 0.0, -1e30)
    hi = np.full(m, 1e30)
    parent = np.full(m, -1)
    parent[5], mu = 4, np.where(np.arange(m) == 5, 0.4, 0.0)
    block = np.ones(m, np.int64)
    block[8:11] = [3, 0, 0]
    ia[8:11], ib[8:11] = ia[8], ib[8]
    lo[8:11] = -1e30
    v0, w0 = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    out = []
    for impl in (numerics.pgs_solve, numerics.FALLBACK.pgs_solve):
        v, w = v0.copy(), w0.copy()
        lam = impl(ia, ib, J, rhs, lo, hi, parent, mu, block, inv_m, inv_i, v, w, 30)
        out.append((lam, v, w))
    for x, y in zip(out[0], out[1]):
        np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-9)
    # with separate impulse directions B (midpoint Jacobian of joints)
    B = J + 0.05 * rng.normal(size=(m, 12))
    out = []
    for impl in (numerics.pgs_solve, numerics.FALLBACK.pgs_solve):
        v, w = v0.copy(), w0.copy()
        lam = impl(ia, ib, J, rhs, lo, hi, parent, mu, block, inv_m, inv_i, v, w, 30, B)
        out.append((lam, v, w))
    for x, y in zip(out[0], out[1]):
        np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-9)
    # an exactly solved bilateral block satisfies J·v = rhs although the impulses act along B
    r = slice(8, 11)
    a, b = ia[8], ib[8]
    cdot = J[r, :3] @ out[1][1][a] + J[r, 3:6] @ out[1][2][a] + J[r, 6:9] @ out[1][1][b] + J[r, 9:] @ out[1][2][b]
    assert np.abs(cdot - rhs[r]).max() < 0.5      # PGS: approximately, sharing bodies with other rows


@pytest.mark.parametrize("integrator", ["velocity_verlet", "rk4"])
def test_large_swing_pendulum_keeps_its_energy(scene, integrator):
    """Joint impulses act along the midpoint Jacobian, so a hinge turns the velocity instead of projecting it
    (a plain projection loses ≈ KE·ω²Δt per second: 20 % of the swing energy in 6 s at Δt = 5 ms)."""
    s, e = scene
    arm = e.add_new("body.rigid_box", position=[2.3, 0, 3.0], params={"length": 1.6, "width": 0.12, "height": 0.12})
    e.add_joint("revolute", arm.id, None, [1.5, 0, 3.0], [0, 1, 0])
    _gravity(e, arm)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.005, duration=10.0, integrator=integrator))
    m = arm.mass()
    i_yy = m * (1.6 ** 2 + 0.12 ** 2) / 12
    x, v, w = rec.column((arm.id, "com")), rec.column((arm.id, "v")), rec.column((arm.id, "w"))
    energy = 0.5 * m * np.einsum("ij,ij->i", v, v) + 0.5 * i_yy * w[:, 1] ** 2 + m * G * (x[:, 2] - 3.0)
    peak = m * G * 0.8
    assert np.abs(energy).max() < 0.01 * peak          # released horizontal: E = 0 throughout
    np.testing.assert_allclose(np.linalg.norm(x - [1.5, 0, 3.0], axis=1), 0.8, atol=1e-3)
