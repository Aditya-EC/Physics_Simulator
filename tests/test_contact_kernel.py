"""The native contact pipeline (C++) against its NumPy reference: same pairs, contacts and rows."""
from __future__ import annotations

import numpy as np
import pytest

from psim import numerics
from psim.physics.collision import WorldShape, collide
from psim.physics.contact_kernel import KINDS, candidate_pairs, contact_rows as reference

native = pytest.mark.skipif(numerics.BACKEND != "native", reason="native core not built")


def _rot(rng):
    q = rng.normal(size=4)
    return np.asarray(numerics.FALLBACK.quat_to_matrix(q / np.linalg.norm(q)), float)


def _scene(seed: int, n: int = 24, spread: float = 1.6):
    """Random shapes of every kind packed so that most kinds of pairs touch; one ground plane."""
    rng = np.random.default_rng(seed)
    kind = np.r_[3, rng.integers(0, 3, n)]
    nb = n + 1
    body = np.r_[0, rng.integers(1, nb, n)]             # some bodies carry several shapes
    center = np.r_[[[0, 0, 0]], rng.uniform(-spread, spread, (n, 3)) + [0, 0, 0.6]]
    rot = np.array([np.eye(3)] + [_rot(rng) for _ in range(n)])
    radius = np.r_[0.0, rng.uniform(0.15, 0.5, n)]
    half = np.r_[[[0, 0, 0]], rng.uniform(0.15, 0.6, (n, 3))]
    half_len = np.r_[0.0, rng.uniform(0.0, 0.5, n)]
    rest = rng.uniform(0, 1, n + 1)
    fric = np.r_[0.5, np.where(rng.random(n) < 0.2, 0.0, rng.uniform(0.1, 1.0, n))]
    dynamic = (rng.random(nb) < 0.8).astype(np.uint8)
    dynamic[0] = 0
    excluded = np.array([[1, 2], [3, 5]], np.int64)
    com = rng.normal(size=(nb, 3))
    v, w = rng.normal(size=(nb, 3)), rng.normal(size=(nb, 3))
    return (kind, body, center, rot, radius, half, half_len, rest, fric, dynamic, excluded, com, v, w, 0.002, 0.2)


@native
@pytest.mark.parametrize("seed", range(12))
def test_native_contacts_and_rows_match_the_reference(seed):
    args = _scene(seed)
    ref = reference(*args)
    nat = numerics.contact_rows(*args)
    assert len(ref["row"]) > 10                                # the scenes really collide
    for key in ("pairs", "shape_a", "shape_b", "row", "ia", "ib", "parent", "mode"):
        np.testing.assert_array_equal(nat[key], ref[key], err_msg=key)
    for key in ("point", "normal", "depth", "J", "rhs", "lo", "hi", "mu", "err"):
        np.testing.assert_allclose(nat[key], ref[key], rtol=1e-9, atol=1e-11, err_msg=key)


@native
@pytest.mark.parametrize("seed", range(4))
def test_static_friction_matches_the_reference(seed):
    args = _scene(seed)
    rng = np.random.default_rng(100 + seed)
    nb = len(args[9])
    extra = dict(static_friction=args[8] + rng.uniform(-0.2, 0.6, len(args[8])),
                 v0=rng.normal(size=(nb, 3)) * 0.003, w0=rng.normal(size=(nb, 3)) * 0.003)
    ref = reference(*args, **extra)
    nat = numerics.contact_rows(*args, **extra)
    for key in ("row", "parent", "mode"):
        np.testing.assert_array_equal(nat[key], ref[key], err_msg=key)
    np.testing.assert_allclose(nat["mu"], ref["mu"], rtol=1e-12)
    plain = reference(*args)

    def contact_mu(d):
        """Friction coefficient of each contact (0 without friction rows)."""
        out = np.zeros(len(d["row"]))
        for k, r in enumerate(d["row"]):
            rows = np.flatnonzero(d["parent"] == r)
            out[k] = d["mu"][rows[0]] if len(rows) else 0.0
        return out
    stat, kin = contact_mu(ref), contact_mu(plain)
    assert np.all(stat >= kin - 1e-12) and np.any(stat > kin + 1e-9)       # sticking contacts get more, never less
    with pytest.raises(ValueError, match="v0"):
        numerics.contact_rows(*args, static_friction=extra["static_friction"], v0=extra["v0"])


@native
@pytest.mark.parametrize("seed", range(3))
def test_rolling_resistance_matches_the_reference(seed):
    args = _scene(seed)
    rolling = np.random.default_rng(seed).uniform(0.0, 0.01, len(args[0]))
    rolling[::3] = 0.0
    ref = reference(*args, rolling=rolling)
    nat = numerics.contact_rows(*args, rolling=rolling)
    spin = dict(rolling=rolling, spinning=rolling[::-1].copy())
    ref2, nat2 = reference(*args, **spin), numerics.contact_rows(*args, **spin)
    for key in ("row", "parent", "mode"):
        np.testing.assert_array_equal(nat2[key], ref2[key], err_msg=key)
    np.testing.assert_allclose(nat2["J"], ref2["J"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(nat2["mu"], ref2["mu"], rtol=1e-12)
    assert len(ref2["ia"]) > len(ref["ia"])                                    # the spinning rows
    for key in ("row", "parent", "mode", "ia", "ib"):
        np.testing.assert_array_equal(nat[key], ref[key], err_msg=key)
    for key in ("J", "mu", "lo", "hi"):
        np.testing.assert_allclose(nat[key], ref[key], rtol=1e-12, atol=1e-14, err_msg=key)
    angular = (np.abs(ref["J"][:, :3]).sum(axis=1) == 0) & (ref["mode"] == 2)
    assert angular.sum() == 2 * len(ref["row"]) - 2 * sum(                     # two rows per contact unless b = 0
        1 for p, q in zip(ref["shape_a"], ref["shape_b"]) if max(rolling[p], rolling[q]) == 0)


@native
def test_every_pair_kind_is_exercised():
    seen = set()
    for seed in range(12):
        args = _scene(seed)
        d = numerics.contact_rows(*args)
        for p, q in d["pairs"]:
            seen.add(tuple(sorted((KINDS[args[0][p]], KINDS[args[0][q]]))))
    basic = KINDS[:4]                                          # hull pairs: tests/test_hulls.py
    expected = {tuple(sorted((a, b))) for a in basic for b in basic if not (a == b == "plane")}
    assert seen == expected


def test_sweep_broadphase_equals_all_pairs():
    """The C++ sort-and-sweep keeps exactly the pairs of the all-pairs bounding-sphere test (same order)."""
    for seed in range(5):
        kind, body, center, rot, radius, half, half_len, *_ = _scene(seed, n=60, spread=4.0)
        ref = candidate_pairs(kind, center, radius, half, half_len, 0.002)
        if numerics.BACKEND == "native":
            d = numerics.contact_rows(*_scene(seed, n=60, spread=4.0))
            touching = {tuple(p) for p in d["pairs"]}
            assert touching <= set(ref)
        assert ref == sorted(ref) and all(i < j for i, j in ref)


@native
@pytest.mark.parametrize("stretch", [(6, 1, 1), (1, 6, 1), (1, 1, 6), (1, 1, 1)])
def test_sweep_axis_follows_the_spread_without_changing_pairs(stretch):
    """Scenes stretched along x, y or z (the sweep runs along the axis of largest spread) keep exactly the
    reference's pairs and contacts."""
    args = list(_scene(7, n=80, spread=2.0))
    args[2] = args[2] * np.asarray(stretch, float)
    ref = reference(*args)
    nat = numerics.contact_rows(*args)
    assert len(ref["pairs"]) > 5
    np.testing.assert_array_equal(nat["pairs"], ref["pairs"])
    np.testing.assert_allclose(nat["point"], ref["point"], rtol=1e-9, atol=1e-11)


def test_reference_kernel_reproduces_the_pairwise_narrowphase():
    args = _scene(3)
    kind, body, center, rot, radius, half, half_len = args[:7]
    d = reference(*args)
    for k, (p, q) in enumerate(d["pairs"]):
        a = WorldShape(KINDS[kind[p]], center[p], rot[p], radius[p], half[p], half_len[p])
        b = WorldShape(KINDS[kind[q]], center[q], rot[q], radius[q], half[q], half_len[q])
        found = collide(a, b)
        sel = (d["shape_a"] == p) & (d["shape_b"] == q)
        np.testing.assert_allclose(d["depth"][sel], [c.depth for c in found])


@native
def test_invalid_input_is_rejected():
    args = list(_scene(0))
    args[1] = args[1].copy()
    args[1][3] = 999                                            # a shape on a missing body
    with pytest.raises(ValueError, match="missing body"):
        numerics.contact_rows(*args)
    args = list(_scene(0))
    args[2] = args[2][:-1]
    with pytest.raises(ValueError, match="same number of shapes"):
        numerics.contact_rows(*args)


@native
@pytest.mark.parametrize("seed", range(6))
def test_native_speculative_contacts_match_the_reference(seed):
    args = list(_scene(seed, spread=2.5))
    rng = np.random.default_rng(100 + seed)
    sweep = np.where(rng.random(len(args[0])) < 0.5, rng.uniform(0.2, 1.0, len(args[0])), 0.0)
    sweep[0] = 0.0
    shift = rng.normal(scale=0.1, size=args[2].shape)
    shift[0] = 0.0
    start = dict(center0=args[2] - shift, rot0=args[3], com0=args[11] - rng.normal(scale=0.1, size=args[11].shape))
    ref = reference(*args, sweep=sweep, dt=0.01, **start)
    nat = numerics.contact_rows(*args, sweep=sweep, dt=0.01, **start)
    assert (ref["depth"] < 0).any()                             # some speculative contacts with a gap
    for key in ("pairs", "shape_a", "shape_b", "row", "ia", "ib", "parent", "mode"):
        np.testing.assert_array_equal(nat[key], ref[key], err_msg=key)
    for key in ("point", "normal", "depth", "J", "rhs", "lo", "hi", "mu", "err"):
        np.testing.assert_allclose(nat[key], ref[key], rtol=1e-9, atol=1e-11, err_msg=key)


def _shoot(scene, ccd: bool, speed: float, restitution: float = 0.0, thin: float = 0.02):
    from psim.processes.process_base import TargetFilter
    from psim.recording.recording import Recording
    from test_recording import make_engine, run
    s, e = scene
    wall = e.add_new("basic.cuboid", position=[1.0, 0, 0], params={"length": thin, "width": 2, "height": 2})
    e.set_property([wall.id], "physics.collider", "box")
    e.set_property([wall.id], "physics.restitution", restitution)
    ball = e.add_new("basic.sphere", position=[-1.0, 0, 0], params={"radius": 0.05})
    e.set_property([ball.id], "physics.collider", "sphere")
    e.set_property([ball.id], "physics.restitution", restitution)
    e.set_property([ball.id], "linear_velocity", [speed, 0, 0])
    e.assign_process("kin.free_motion", TargetFilter(ids=[ball.id]))
    rec = Recording()
    eng = make_engine(scene, rec, delta_time=0.005, duration=0.1, continuous_collision=ccd)
    run(eng)
    return rec.column((ball.id, "com"))[:, 0], rec.column((ball.id, "v"))[:, 0]


def test_without_ccd_a_fast_ball_tunnels_through_a_thin_wall(scene):
    x, v = _shoot(scene, ccd=False, speed=100.0)              # 0.5 m per step through a 2 cm wall
    assert x[-1] > 1.5 and v[-1] == pytest.approx(100.0)


def test_ccd_stops_a_fast_ball_at_a_thin_wall(scene):
    x, v = _shoot(scene, ccd=True, speed=100.0)
    assert x.max() <= 1.0 - 0.01 - 0.05 + 0.003               # never past the wall's front face
    assert abs(v[-1]) < 1e-6                                  # inelastic: comes to rest against it


def test_ccd_keeps_restitution_of_fast_impacts(scene):
    x, v = _shoot(scene, ccd=True, speed=100.0, restitution=0.8)
    assert x.max() <= 1.0 - 0.06 + 0.003
    assert v[-1] == pytest.approx(-80.0, rel=1e-6)            # e·v back, despite meeting mid-step


def test_ccd_fast_box_against_a_thin_plate(scene):
    from psim.processes.process_base import TargetFilter
    from psim.recording.recording import Recording
    from test_recording import make_engine, run
    s, e = scene
    plate = e.add_new("basic.cuboid", position=[0, 0, 0], params={"length": 3, "width": 3, "height": 0.01})
    e.set_property([plate.id], "physics.collider", "box")
    box = e.add_new("body.rigid_box", position=[0, 0, 2.0], params={"length": 0.2, "width": 0.2, "height": 0.2})
    e.set_property([box.id], "linear_velocity", [0, 0, -60.0])
    for o in (plate, box):                                    # inelastic: it must stop on the plate, not in it
        e.set_property([o.id], "physics.restitution", 0.0)
    e.assign_process("dyn.gravity", TargetFilter(ids=[box.id]))
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.005, duration=0.5))
    z = rec.column((box.id, "com"))[:, 2]
    assert z.min() > 0.005 + 0.1 - 0.01 and z[-1] == pytest.approx(0.105, abs=0.005)   # resting on the plate
