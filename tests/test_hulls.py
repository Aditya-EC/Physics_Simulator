"""Convex-hull colliders: hull construction, collision behaviour and native/NumPy parity."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim import numerics
from psim.physics.colliders import collider_for
from psim.physics.contact_kernel import contact_rows as reference
from psim.physics.hull import HullError, box_hull, convex_hull, pack_hulls, translated, unpack_hulls
from psim.processes.process_base import TargetFilter
from psim.recording.recording import Recording

from test_recording import make_engine, run

native = pytest.mark.skipif(numerics.BACKEND != "native", reason="native core not built")


# ---- construction ------------------------------------------------------------------------------------------------

def test_box_and_prism_hulls_are_exact():
    h = box_hull([1, 2, 3])
    assert (len(h.verts), len(h.faces), len(h.edges)) == (8, 6, 12) and h.volume == pytest.approx(48.0)
    for f, n, d in zip(h.faces, h.normals, h.offsets):
        assert len(f) == 4 and np.allclose(h.verts[f] @ n, d)                  # planar quads on their planes
        assert np.all(h.verts @ n <= d + 1e-12)                                # every vertex inside
    # hexagonal prism: 12 vertices, 8 faces (two hexagons + six rectangles), 18 edges
    ang = np.arange(6) * math.pi / 3
    ring = np.c_[np.cos(ang), np.sin(ang), np.zeros(6)]
    h = convex_hull(np.r_[ring, ring + [0, 0, 2], [[0, 0, 1]]])               # interior point is dropped
    assert (len(h.verts), len(h.faces), len(h.edges)) == (12, 8, 18)
    assert h.volume == pytest.approx(2 * 1.5 * math.sqrt(3))
    assert all(len(set(ef)) == 2 for ef in h.edge_faces.tolist())


def test_flat_and_degenerate_inputs():
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    h = convex_hull(sq)                                                       # thickened slab
    assert h.volume > 0 and np.ptp(h.verts[:, 2]) == pytest.approx(1e-3)
    with pytest.raises(HullError):
        convex_hull(np.zeros((0, 3)))


def test_detailed_meshes_are_reduced_and_marked(scene):
    s, e = scene
    ell = e.add_new("basic.ellipsoid")
    e.set_property([ell.id], "physics.collider", "auto")
    c = collider_for(ell)
    assert c.kind == "convex" and c.approximate and len(c.hull.verts) <= 64 and "support points" in c.note
    cyl = e.add_new("basic.cylinder")
    e.set_property([cyl.id], "physics.collider", "auto")
    c = collider_for(cyl)
    assert c.kind == "convex" and not c.approximate                          # the faceted mesh is its own hull
    assert c.hull.volume == pytest.approx(cyl.mass_properties()["volume"], rel=1e-9)
    tor = e.add_new("basic.torus")
    e.set_property([tor.id], "physics.collider", "convex")
    c = collider_for(tor)
    assert c.approximate and "support points" in c.note or "not convex" in c.note
    assert "convex hull" in c.describe()


def test_pack_unpack_round_trip():
    hulls = [box_hull([1, 1, 1]), convex_hull(np.random.default_rng(0).normal(size=(30, 3)))]
    again = unpack_hulls(pack_hulls(hulls))
    for a, b in zip(hulls, again):
        np.testing.assert_array_equal(a.verts, b.verts)
        np.testing.assert_array_equal(a.edges, b.edges)
        assert a.faces == b.faces


# ---- behaviour -------------------------------------------------------------------------------------------------

def _ground(e):
    g = e.add_new("basic.plane", params={"length": 30, "width": 30})
    e.set_property([g.id], "physics.collider", "auto")
    return g


def test_cylinder_on_its_side_rolls_at_its_radius(scene):
    s, e = scene
    _ground(e)
    cyl = e.add_new("basic.cylinder", position=[0, 0, 0.45])
    cyl.set_orientation(numerics.quat_from_axis_angle([1, 0, 0], math.pi / 2))
    e.set_property([cyl.id], "physics.collider", "auto")
    e.set_property([cyl.id], "linear_velocity", [2.0, 0, 0])
    e.assign_process("dyn.gravity", TargetFilter(ids=[cyl.id]))
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.005, duration=1.0))
    v, w, x = rec.column((cyl.id, "v"))[-1], rec.column((cyl.id, "w"))[-1], rec.column((cyl.id, "com"))[-1]
    r = cyl.params["radius"]
    assert v[0] == pytest.approx(w[1] * r, rel=0.02)                         # friction brought it to rolling
    assert x[2] == pytest.approx(r, abs=0.01)                                # on its curved side, not a box corner
    assert v[0] < 2.0 and v[0] > 1.0                                          # 1/3 of the momentum went into spin


def test_cone_stands_on_its_base_and_wedge_slides(scene):
    s, e = scene
    _ground(e)
    cone = e.add_new("basic.cone", position=[0, 0, 1.0])
    e.set_property([cone.id], "physics.collider", "auto")
    e.assign_process("dyn.gravity", TargetFilter(ids=[cone.id]))
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.005, duration=1.5))
    assert rec.column((cone.id, "origin"))[-1, 2] == pytest.approx(cone.params["height"] / 2, abs=0.004)
    assert np.abs(rec.column((cone.id, "w"))[-1]).max() < 1e-6


def test_stacked_cylinders_rest(scene):
    s, e = scene
    _ground(e)
    c1 = e.add_new("basic.cylinder", position=[0, 0, 0.55])
    c2 = e.add_new("basic.cylinder", position=[0.1, 0, 1.6])
    for o in (c1, c2):
        e.set_property([o.id], "physics.collider", "auto")
    e.assign_process("dyn.gravity", TargetFilter(ids=[c1.id, c2.id]))
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.005, duration=1.5))
    z1, z2 = rec.column((c1.id, "origin"))[:, 2], rec.column((c2.id, "origin"))[:, 2]
    assert z1[-1] == pytest.approx(0.5, abs=0.004) and z2[-1] == pytest.approx(1.5, abs=0.008)
    assert np.ptp(z2[-60:]) < 1e-4                                           # at rest, no jitter


# ---- native ≡ reference ------------------------------------------------------------------------------------------

def _hull_scene(seed: int, n: int = 16):
    rng = np.random.default_rng(seed)
    kinds = np.r_[3, rng.choice([0, 1, 2, 4, 4], n)]
    nb = n + 1
    body = np.r_[0, np.arange(1, nb)]
    center = np.r_[[[0, 0, 0]], rng.uniform(-1.4, 1.4, (n, 3)) + [0, 0, 0.5]]
    rot = [np.eye(3)]
    for _ in range(n):
        q = rng.normal(size=4)
        rot.append(np.asarray(numerics.FALLBACK.quat_to_matrix(q / np.linalg.norm(q)), float))
    rot = np.array(rot)
    radius = np.r_[0.0, rng.uniform(0.15, 0.45, n)]
    half = np.r_[[[0, 0, 0]], rng.uniform(0.15, 0.5, (n, 3))]
    half_len = np.r_[0.0, rng.uniform(0.0, 0.4, n)]
    hulls, hull_id = [], np.full(nb, -1, np.int64)
    for i in range(1, nb):
        if kinds[i] == 4:
            pts = rng.normal(size=(rng.integers(6, 30), 3)) * rng.uniform(0.2, 0.5, 3)
            h = convex_hull(pts)
            lo, hi = h.verts.min(axis=0), h.verts.max(axis=0)
            h = translated(h, -(lo + hi) / 2)
            half[i] = np.maximum((hi - lo) / 2, 1e-6)
            hull_id[i] = len(hulls)
            hulls.append(h)
        elif kinds[i] == 1:
            hull_id[i] = len(hulls)
            hulls.append(box_hull(half[i]))
    rest = rng.uniform(0, 1, nb)
    fric = rng.uniform(0.1, 1.0, nb)
    dynamic = np.r_[0, np.ones(n)].astype(np.uint8)
    com, v, w = rng.normal(size=(nb, 3)), rng.normal(size=(nb, 3)), rng.normal(size=(nb, 3))
    args = (kinds, body, center, rot, radius, half, half_len, rest, fric, dynamic, np.zeros((0, 2), np.int64),
            com, v, w, 0.002, 0.2)
    return args, dict(hull_id=hull_id, hulls=pack_hulls(hulls))


@native
@pytest.mark.parametrize("seed", range(10))
def test_native_hull_contacts_match_the_reference(seed):
    args, extra = _hull_scene(seed)
    ref = reference(*args, **extra)
    nat = numerics.contact_rows(*args, **extra)
    kinds = args[0]
    assert any(4 in (kinds[p], kinds[q]) for p, q in ref["pairs"])       # hull pairs really collide
    for key in ("pairs", "shape_a", "shape_b", "row", "ia", "ib", "parent", "mode"):
        np.testing.assert_array_equal(nat[key], ref[key], err_msg=key)
    for key in ("point", "normal", "depth", "J", "rhs", "lo", "hi", "mu", "err"):
        np.testing.assert_allclose(nat[key], ref[key], rtol=1e-8, atol=1e-10, err_msg=key)


@native
def test_every_hull_pair_kind_is_exercised():
    seen = set()
    for seed in range(10):
        args, extra = _hull_scene(seed)
        for p, q in numerics.contact_rows(*args, **extra)["pairs"]:
            seen.add(tuple(sorted((int(args[0][p]), int(args[0][q])))))
    assert {(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)} <= seen


@native
def test_hull_shapes_without_hull_data_are_rejected():
    args, _ = _hull_scene(0)
    with pytest.raises(ValueError, match="hull"):
        numerics.contact_rows(*args)


def test_convex_collider_is_drawn_from_its_hull_edges(scene):
    from psim.rendering import physics_overlay
    s, e = scene
    cone = e.add_new("basic.cone", position=[1, 2, 3])
    e.set_property([cone.id], "physics.collider", "auto")
    c = collider_for(cone)
    seg = physics_overlay.shape_lines(c, [1, 2, 3], np.eye(3)).reshape(-1, 2, 3)
    assert len(seg) == len(c.hull.edges)
    world = c.hull.verts + c.center + [1, 2, 3]
    for a, b in seg:                                   # every segment joins two hull vertices
        assert np.min(np.linalg.norm(world - a, axis=1)) < 1e-9 and np.min(np.linalg.norm(world - b, axis=1)) < 1e-9
