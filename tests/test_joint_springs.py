"""Joint springs (revolute, prismatic, distance) and breakable joints."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim.physics.joints import JointError, validate_joints
from psim.processes.process_base import TargetFilter
from psim.recording.recording import Recording

from test_recording import make_engine, run

G = 9.80665


def _box(e, pos=(0, 0, 1.0)):
    return e.add_new("body.rigid_box", position=list(pos), params={"length": 0.4, "width": 0.4, "height": 0.4})


def _free(e, *objs):
    """Dynamic bodies without forces (a body without any process would be static)."""
    e.assign_process("dyn.gravity", TargetFilter(ids=[o.id for o in objs]), {"g": [0.0, 0.0, 0.0]})


def _crossings(t, x):
    """Times at which x changes sign (linear interpolation)."""
    k = np.flatnonzero(np.sign(x[:-1]) * np.sign(x[1:]) < 0)
    return t[k] - x[k] * (t[k + 1] - t[k]) / (x[k + 1] - x[k])


def test_torsion_spring_on_a_hinge_oscillates_at_sqrt_k_over_i(scene):
    s, e = scene
    box = _box(e)
    _free(e, box)
    e.set_property([box.id], "angular_velocity", [0, 0, 1.0])
    k = 50.0
    e.add_joint("revolute", box.id, None, [0, 0, 1.0], [0, 0, 1], spring_stiffness=k)
    rec = Recording(interval_steps=1)
    run(make_engine(scene, rec, delta_time=0.001, duration=3.0, integrator="velocity_verlet"))
    i_zz = box.mass_properties()["inertia"][2, 2]
    omega = math.sqrt(k / i_zz)
    tc = _crossings(rec.times(), rec.column((box.id, "w"))[:, 2])
    assert np.mean(np.diff(tc)) == pytest.approx(math.pi / omega, rel=0.01)       # half periods
    np.testing.assert_allclose(rec.column((box.id, "com"))[-1], [0, 0, 1.0], atol=1e-6)


def test_slider_spring_and_damping(scene):
    s, e = scene
    box = _box(e)
    _free(e, box)
    e.set_property([box.id], "linear_velocity", [0.5, 0, 0])
    k = 2000.0
    jid = e.add_joint("prismatic", box.id, None, [0, 0, 1.0], [1, 0, 0], spring_stiffness=k)
    rec = Recording(interval_steps=1)
    run(make_engine(scene, rec, delta_time=0.001, duration=5.0))
    m = box.mass_properties()["mass"]
    omega = math.sqrt(k / m)
    x = rec.column((box.id, "com"))[:, 0]
    assert np.max(np.abs(x)) == pytest.approx(0.5 / omega, rel=0.01)            # amplitude v₀/Ω
    assert np.mean(np.diff(_crossings(rec.times(), x))) == pytest.approx(math.pi / omega, rel=0.01)
    # with damping the oscillation decays as exp(−c t / 2m)
    joints = s.joints_snapshot()
    joints[0]["spring_damping"] = c = 2 * 0.1 * math.sqrt(k * m)                 # damping ratio 0.1
    e.set_joints(joints)
    rec = Recording(interval_steps=1)
    run(make_engine(scene, rec, delta_time=0.001, duration=2.0))
    x, v = rec.column((box.id, "com"))[-1, 0], rec.column((box.id, "v"))[-1, 0]
    zeta = 0.1
    wd = omega * math.sqrt(1 - zeta ** 2)
    envelope = math.hypot(x, (v + zeta * omega * x) / wd)                       # A·exp(−ζΩt) at t = 2 s
    assert envelope == pytest.approx(0.5 / wd * math.exp(-c / (2 * m) * 2.0), rel=0.01)
    assert jid


def test_soft_distance_joint_stretches_by_mg_over_k(scene):
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 1.0])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    k = 400.0
    m = ball.mass_properties()["mass"]
    e.add_joint("distance", ball.id, None, [0, 0, 1.0], anchor_world_b=[0, 0, 2.0], spring_stiffness=k,
                spring_damping=2 * math.sqrt(k * m))                              # critically damped
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.002, duration=3.0))
    assert rec.column((ball.id, "com"))[-1, 2] == pytest.approx(2.0 - 1.0 - m * G / k, abs=1e-4)


def test_very_stiff_spring_stays_stable(scene):
    s, e = scene
    box = _box(e)
    _free(e, box)
    e.set_property([box.id], "linear_velocity", [1.0, 0, 0])
    e.add_joint("prismatic", box.id, None, [0, 0, 1.0], [1, 0, 0], spring_stiffness=1e12, spring_damping=10.0)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.005, duration=1.0))
    x = rec.column((box.id, "com"))[:, 0]
    assert np.all(np.isfinite(x)) and np.max(np.abs(x)) < 0.01                   # implicit: no blow-up


@pytest.mark.parametrize("limit,breaks", [(0.5, True), (2.0, False)])
def test_joint_breaks_above_its_force(scene, limit, breaks):
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 1.0])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    m = ball.mass_properties()["mass"]
    e.add_joint("ball", ball.id, None, [0, 0, 1.0], break_force=limit * m * G, name="Hook")
    rec = Recording()
    eng = make_engine(scene, rec, delta_time=0.002, duration=0.5)
    run(eng)
    z = rec.column((ball.id, "com"))[-1, 2]
    events = [ev.message for ev in rec.events if ev.kind == "joint"]
    if breaks:
        assert z == pytest.approx(1.0 - G * 0.5 ** 2 / 2, abs=0.01)                # free fall after the break
        assert eng.physics.broken == [s.joints[0].id]
        assert len(events) == 1 and "Hook" in events[0] and "broke" in events[0]
    else:
        assert z == pytest.approx(1.0, abs=1e-6) and not eng.physics.broken and not events


def test_torque_limit_breaks_a_cantilever_and_state_keeps_it(scene):
    s, e = scene
    box = _box(e, pos=(0.5, 0, 1.0))
    e.assign_process("dyn.gravity", TargetFilter(ids=[box.id]))
    m = box.mass_properties()["mass"]
    jid = e.add_joint("fixed", box.id, None, [0, 0, 1.0], break_torque=0.5 * m * G * 0.5)   # holds half m g L
    eng = make_engine(scene, None, delta_time=0.002, duration=0.2)
    run(eng)
    assert eng.physics.broken == [jid]
    state = eng.export_state()
    assert state["physics"]["broken"] == [jid]
    eng2 = make_engine(scene, None, delta_time=0.002, duration=0.4)
    eng2.import_state(state)
    assert eng2.physics.broken == [jid] and not eng2.physics.active_joints


def test_spring_and_break_validation():
    base = {"id": "JNT-0001", "name": "J", "type": "revolute", "body_a": "OBJ-000001"}
    j = validate_joints([dict(base, spring_stiffness=5.0, spring_rest=-0.3, break_force=10.0)])[0]
    assert j.has_spring and j.to_dict()["spring_rest"] == -0.3
    for bad, msg in ((dict(base, spring_stiffness=-1), "not negative"), (dict(base, break_torque="x"), "number"),
                     (dict(base, type="ball", spring_stiffness=1.0), "springs act on"),
                     (dict(base, spring_damping=float("inf")), "finite")):
        with pytest.raises(JointError, match=msg):
            validate_joints([bad])
    assert not validate_joints([dict(base, type="ball")])[0].has_spring


def test_mechanisms_demo_behaves_as_described(scene):
    """The 'mechanisms' template (examples/mechanisms_demo.psim): each station does what the docs say."""
    from psim.project.templates import PROJECT_TEMPLATES
    s, e = scene
    PROJECT_TEMPLATES["mechanisms"][1](s, e)
    by = {o.name: o for o in s.objects.values()}
    rec = Recording()
    eng = make_engine(scene, rec, delta_time=0.004, duration=2.5, integrator="velocity_verlet")
    run(eng)

    def last(name, key="com"):
        return rec.column((by[name].id, key))[-1]
    marble = last("Marble")
    assert marble[2] == pytest.approx(0.12, abs=0.002) and math.hypot(marble[0] + 8.0, marble[1]) < 0.35 - 0.12
    assert [j.name for j in s.joints if j.id in eng.physics.broken] == ["Breakable Rope"]
    assert last("Hanging Crate")[2] == pytest.approx(0.2, abs=0.01)
    sticky, slippery = (rec.column((by[n].id, "com")) for n in ("Sticky Block", "Slippery Block"))
    assert np.linalg.norm(sticky[-1] - sticky[0]) < 1e-5 and np.linalg.norm(slippery[-1] - slippery[0]) > 1.0
    v_resisted = np.linalg.norm(last("Rolling Ball (b = 5 mm)", "v"))
    v_free = np.linalg.norm(last("Rolling Ball (b = 0)", "v"))
    assert v_free == pytest.approx(2.5, abs=0.01) and v_resisted < 2.5 - 0.3
    assert last("Sprung Platform")[2] < 1.0 - 0.05                                  # sagged on its spring


def test_damper_alone_keeps_the_rod(scene):
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 1.0])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    e.set_property([ball.id], "linear_velocity", [1.0, 0, 0])
    e.add_joint("distance", ball.id, None, [0, 0, 1.0], anchor_world_b=[0, 0, 2.0], spring_damping=5.0)
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.002, duration=1.0))
    r = np.linalg.norm(rec.column((ball.id, "com")) - [0, 0, 2.0], axis=1)
    np.testing.assert_allclose(r, 1.0, atol=1e-3)                                  # still a rod of length 1


def test_spring_force_counts_toward_breaking(scene):
    """A soft rope has no solver rows: its spring force (≈ m g at rest) alone breaks it."""
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 1.0])
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    m = ball.mass_properties()["mass"]
    jid = e.add_joint("distance", ball.id, None, [0, 0, 1.0], anchor_world_b=[0, 0, 2.0], spring_stiffness=1000.0,
                      spring_damping=50.0, break_force=0.5 * m * G)
    eng = make_engine(scene, None, delta_time=0.002, duration=0.5)
    run(eng)
    assert eng.physics.broken == [jid]


def test_collider_shapes_are_cached(scene):
    from psim.physics.colliders import collider_for
    s, e = scene
    tor = e.add_new("basic.torus", params={"segments": 24})
    e.set_property([tor.id], "physics.collider", "auto")
    first = collider_for(tor)
    assert collider_for(tor) is first                                              # no recomputation per frame
    e.set_property([tor.id], "scale", [2.0, 1.0, 1.0])
    assert collider_for(tor) is not first and collider_for(tor).half_extents[0] == pytest.approx(
        2 * first.half_extents[0], rel=1e-6)                                       # new geometry, new shape


def _dist_to_line(p, a, b):
    d = (b - a) / np.linalg.norm(b - a)
    r = p - a
    return np.linalg.norm(r - np.outer(r @ d, d), axis=1)


def test_springs_are_drawn_as_coils_and_spirals(scene):
    from psim.rendering import physics_overlay as po
    s, e = scene
    m = 0.05
    ball = e.add_new("body.point_mass", position=[0, 0, 1.0])
    e.add_joint("distance", ball.id, None, [0, 0, 1.0], anchor_world_b=[0, 0, 2.0], spring_stiffness=100.0)
    (pts,), _ = po.joint_lines(s, m)
    a, b = np.array([0, 0, 1.0]), np.array([0, 0, 2.0])
    assert len(pts) % 2 == 0 and np.all(np.isfinite(pts))
    coil = pts[_dist_to_line(pts, a, b) > 1e-9]
    assert len(coil) > 100                                            # windings, not a straight rod
    assert np.max(_dist_to_line(pts, a, b)) <= 1.2 * m * 1.2 + 1e-12
    assert np.min(coil[:, 2]) >= 1.0 - 1e-12 and np.max(coil[:, 2]) <= 2.0 + 1e-12   # between the anchors

    # a damper alone keeps the rod: drawn straight
    joints = s.joints_snapshot()
    joints[0]["spring_stiffness"], joints[0]["spring_damping"] = 0.0, 5.0
    e.set_joints(joints)
    (pts,), _ = po.joint_lines(s, m)
    assert np.max(_dist_to_line(pts[(pts[:, 0] == 0) & (pts[:, 1] == 0)], a, b)) == 0.0
    assert len(pts) < 40

    # a prismatic spring: the coil length follows the travel
    e.set_joints([])
    box = _box(e)
    e.add_joint("prismatic", box.id, None, [0, 0, 1.0], [1, 0, 0], spring_stiffness=50.0)
    (rest,), _ = po.joint_lines(s, m)
    e.set_property([box.id], "position", [0.3, 0, 1.0])
    (moved,), _ = po.joint_lines(s, m)
    assert np.ptp(moved[:, 0]) == pytest.approx(np.ptp(rest[:, 0]) + 0.3, abs=1e-9)

    # a torsion spring: a flat spiral around the hinge axis at the anchor
    e.set_joints([])
    e.add_joint("revolute", box.id, None, [0.3, 0, 1.0], [0, 0, 1], spring_stiffness=5.0)
    (pts,), _ = po.joint_lines(s, m)
    r = np.linalg.norm(pts[:, :2] - [0.3, 0.0], axis=1)
    flat = np.abs(pts[:, 2] - 1.0) < 1e-12
    spiral = pts[flat & (r > 0.8 * m - 1e-12) & (r < 3.0 * m + 1e-12) & (r > 1.5 * m)]
    assert len(spiral) > 20


def test_mechanisms_demo_joint_overlay_is_well_formed(scene):
    import os

    from psim.project import serialization as ser
    from psim.rendering import physics_overlay as po
    from psim.units.unit_registry import UnitRegistry
    s, _ = scene
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples",
                        "mechanisms_demo.psim")
    ser.load_project(path, s, UnitRegistry())
    pts, cols = po.joint_lines(s, 0.05)
    assert len(pts) == len(s.joints) == 3
    for p, c in zip(pts, cols):
        assert len(p) % 2 == 0 and len(p) == len(c) and np.all(np.isfinite(p))
    assert max(len(p) for p in pts) > 100                             # the sprung joints carry coils / spirals
