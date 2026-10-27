"""Convex decomposition of concave meshes: exact cuts, robust hull volumes, compound colliders, native ≡ NumPy."""
from __future__ import annotations

import math

import numpy as np
import pytest

from psim import numerics
from psim.physics import decompose as dc
from psim.physics.colliders import collider_for
from psim.physics.hull import convex_hull
from psim.processes.process_base import TargetFilter
from psim.recording.recording import Recording

from test_recording import make_engine, run

native = pytest.mark.skipif(numerics.BACKEND != "native", reason="native core not built")


def _mesh(scene, type_id, **params):
    s, e = scene
    o = e.add_new(type_id, params=params or None)
    m = o.mesh()
    return o, np.asarray(m.vertices, float), np.asarray(m.indices, np.int64)


def test_cuts_are_exact_and_closed(scene):
    _, v, f = _mesh(scene, "basic.torus", segments=24)
    tris = v[f]
    vol = dc.chain_volume(tris)
    assert vol == pytest.approx(2 * math.pi ** 2 * 0.5 * 0.15 ** 2, rel=0.07)   # faceted
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        d = float(rng.uniform(-0.3, 0.3))
        left, right = dc.clip(tris, n, d), dc.clip(tris, -n, -d)
        assert dc.chain_volume(left) + dc.chain_volume(right) == pytest.approx(vol, rel=1e-12)
        assert np.all(left.reshape(-1, 3) @ n <= d + 1e-12) and np.all(right.reshape(-1, 3) @ n >= d - 1e-12)
        # a closed chain: the area vectors of its triangles cancel
        area = np.cross(left[:, 1] - left[:, 0], left[:, 2] - left[:, 0]).sum(axis=0)
        np.testing.assert_allclose(area, 0, atol=1e-12)


def test_hull_volume_is_robust_with_coplanar_points():
    g = np.linspace(0, 1, 9)
    pts = np.array([[x, y, z] for x in g for y in g for z in (0.0, 1.0)]
                   + [[x, y, z] for x in (0.0, 1.0) for y in g for z in g], float)
    rng = np.random.default_rng(3)
    for _ in range(4):
        q = rng.normal(size=4)
        r = np.asarray(numerics.FALLBACK.quat_to_matrix(q / np.linalg.norm(q)), float)
        assert dc.hull_volume(pts @ r.T) == pytest.approx(1.0, rel=1e-12)
        assert numerics.hull_volume(pts @ r.T) == pytest.approx(1.0, rel=1e-12)
        h = convex_hull(pts @ r.T, max_vertices=10 ** 6)               # the collision hull merges the grid
        assert len(h.verts) == 8 and len(h.faces) == 6
    assert dc.hull_volume(pts[pts[:, 2] == 0.0]) == 0.0                   # coplanar
    p = rng.normal(size=(300, 3))
    assert numerics.hull_volume(p) == pytest.approx(dc.hull_volume(p), rel=1e-12)
    assert dc.hull_volume(p, 32) < dc.hull_volume(p)                      # reduced to support points


def test_decomposition_quality(scene):
    for type_id, params, parts_min, excess in (("basic.cylinder", {}, 1, 1e-9), ("mech.i_beam", {}, 3, 0.03),
                                                ("basic.torus", {"segments": 24}, 8, 0.25)):
        _, v, f = _mesh(scene, type_id, **params)
        vol = dc.chain_volume(v[f])
        parts = numerics.convex_decomposition(v, f, max_hull_points=128)
        hull_sum = sum(numerics.hull_volume(p) for p in parts)
        assert len(parts) >= parts_min and (type_id != "basic.cylinder" or len(parts) == 1)
        assert vol * (1 - 1e-9) <= hull_sum <= vol * (1 + excess), type_id
        lo, hi = v.min(axis=0), v.max(axis=0)
        for p in parts:                                                  # parts stay inside the mesh bounds
            assert np.all(p >= lo - 1e-12) and np.all(p <= hi + 1e-12)
        assert numerics.hull_volume(v) > hull_sum or type_id == "basic.cylinder"
    with pytest.raises(ValueError, match="closed solid"):
        numerics.convex_decomposition(v, f[:, ::-1])                     # inside out


@native
@pytest.mark.parametrize("type_id,params", [("mech.i_beam", {}), ("mech.pulley", {}),
                                            ("basic.torus", {"segments": 16})])
def test_native_decomposition_matches_the_reference(scene, type_id, params):
    _, v, f = _mesh(scene, type_id, **params)
    nat = numerics.convex_decomposition(v, f, max_hull_points=128)
    ref = dc.decompose(v, f, max_hull_points=128)
    assert len(nat) == len(ref)
    for a, b in zip(nat, ref):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_compound_colliders(scene):
    s, e = scene
    tor = e.add_new("basic.torus", params={"segments": 24})
    e.set_property([tor.id], "physics.collider", "auto")
    c = collider_for(tor)
    assert c.kind == "convex_parts" and len(c.parts) >= 8 and all(p.kind == "convex" for p in c.parts)
    assert "convex parts" in c.describe() and c.bounding_radius() >= 0.65 - 1e-9
    e.set_property([tor.id], "physics.collider", "convex")                 # explicit hull: filled, approximate
    c = collider_for(tor)
    assert c.kind == "convex" and c.approximate and "not convex" in c.note
    cyl = e.add_new("basic.cylinder")
    e.set_property([cyl.id], "physics.collider", "auto")
    assert collider_for(cyl).kind == "convex" and not collider_for(cyl).approximate
    e.set_property([cyl.id], "physics.collider", "convex_parts")           # requested on a convex mesh: one part
    c = collider_for(cyl)
    assert c.kind == "convex_parts" and len(c.parts) == 1 and not c.approximate
    beam = e.add_new("mech.i_beam")
    e.set_property([beam.id], "physics.collider", "auto")
    c = collider_for(beam)
    assert c.kind == "convex_parts" and not c.approximate                  # parts within 1 % of the volume
    e.assign_process("dyn.gravity", TargetFilter(ids=[beam.id]))
    eng = make_engine(scene, None, delta_time=0.01, duration=0.01)
    entries = [x for x in eng.physics.setup.colliders if x.source == beam.id]
    assert len(entries) == len(c.parts) and all(x.shape.kind == "convex" for x in entries)   # one entry per part


def test_ball_falls_through_the_hole_of_a_torus(scene):
    s, e = scene
    g = e.add_new("basic.plane", params={"length": 10, "width": 10})
    e.set_property([g.id], "physics.collider", "auto")
    tor = e.add_new("basic.torus", position=[0, 0, 0.15], params={"segments": 24})
    e.set_property([tor.id], "physics.collider", "auto")
    ball = e.add_new("basic.sphere", position=[0, 0, 1.0], params={"radius": 0.08})
    e.set_property([ball.id], "physics.collider", "auto")
    ring = e.add_new("basic.torus", position=[3, 0, 0.5], params={"segments": 24})
    e.set_property([ring.id], "physics.collider", "auto")
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id, ring.id]))
    rec = Recording()
    run(make_engine(scene, rec, delta_time=0.005, duration=1.2))
    assert rec.column((ball.id, "com"))[-1, 2] == pytest.approx(0.08, abs=0.003)   # on the ground, inside the ring
    assert np.linalg.norm(rec.column((ball.id, "com"))[-1, :2]) < 0.5 - 0.15
    z = rec.column((ring.id, "origin"))[:, 2]
    assert z[-1] == pytest.approx(0.15, abs=0.003) and np.ptp(z[-40:]) < 1e-4       # rests flat on its tube
