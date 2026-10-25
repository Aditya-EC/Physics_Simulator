"""The C++ core and the NumPy fallback must agree on every routine."""
from __future__ import annotations

import numpy as np
import pytest

from psim import numerics
from psim.numerics import _fallback as fb
from psim.scene import geometry as g

pytestmark = pytest.mark.skipif(numerics.BACKEND != "native", reason="native module not built")
nat = numerics._impl
rng = np.random.default_rng(1234)


def _rand_quat():
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_backend_is_native():
    assert numerics.BACKEND == "native"
    assert "C++" in numerics.backend_description()


@pytest.mark.parametrize("k", range(20))
def test_quaternion_routines_agree(k):
    a, b = _rand_quat(), _rand_quat()
    v = rng.normal(size=3)
    e = rng.uniform(-3, 3, size=3)
    np.testing.assert_allclose(nat.quat_multiply(a, b), fb.quat_multiply(a, b), atol=1e-12)
    np.testing.assert_allclose(nat.quat_normalize(a * 3), fb.quat_normalize(a * 3), atol=1e-12)
    np.testing.assert_allclose(nat.quat_to_matrix(a), fb.quat_to_matrix(a), atol=1e-12)
    np.testing.assert_allclose(nat.quat_rotate(a, v), fb.quat_rotate(a, v), atol=1e-12)
    np.testing.assert_allclose(nat.quat_from_euler(e), fb.quat_from_euler(e), atol=1e-12)
    np.testing.assert_allclose(nat.quat_to_euler(a), fb.quat_to_euler(a), atol=1e-9)
    np.testing.assert_allclose(nat.matrix_to_quat(fb.quat_to_matrix(a)), fb.quat_normalize(a), atol=1e-12)
    p1, q1 = nat.pose_compose(v, a, e, b)
    p2, q2 = fb.pose_compose(v, a, e, b)
    np.testing.assert_allclose(p1, p2, atol=1e-12)
    np.testing.assert_allclose(q1, q2, atol=1e-12)


def test_matrices_agree():
    t, q, s = rng.normal(size=3), _rand_quat(), rng.uniform(0.2, 3, 3)
    np.testing.assert_allclose(nat.trs_matrix(t, q, s), fb.trs_matrix(t, q, s), atol=1e-12)
    np.testing.assert_allclose(nat.look_at([3, -4, 2], [0, 0, 0], [0, 0, 1]), fb.look_at([3, -4, 2], [0, 0, 0], [0, 0, 1]),
                               atol=1e-12)
    np.testing.assert_allclose(nat.perspective(0.8, 1.6, 0.1, 100), fb.perspective(0.8, 1.6, 0.1, 100), atol=1e-12)
    np.testing.assert_allclose(nat.orthographic(-2, 2, -1, 1, -5, 5), fb.orthographic(-2, 2, -1, 1, -5, 5), atol=1e-12)
    m = fb.trs_matrix(t, q, s)
    pts = rng.normal(size=(50, 3))
    np.testing.assert_allclose(nat.transform_points(m, pts), fb.transform_points(m, pts), atol=1e-12)


@pytest.mark.parametrize("mesh", [g.box(1, 2, 3), g.sphere(0.7, 24, 12), g.torus(1, 0.3, 32, 12),
                                  g.extrude(g.gear_profile(10, 1, 0.8), 0.3), g.plane(2, 3, 4)])
def test_mesh_routines_agree(mesh):
    v, f = mesh.vertices, mesh.indices
    for fn in ("mass_properties_solid", "mass_properties_shell"):
        a, b = getattr(nat, fn)(v, f, 2.5), getattr(fb, fn)(v, f, 2.5)
        assert a["measure"] == pytest.approx(b["measure"], rel=1e-9, abs=1e-12)
        np.testing.assert_allclose(a["centroid"], b["centroid"], atol=1e-7)
        np.testing.assert_allclose(a["inertia"], b["inertia"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(nat.compute_vertex_normals(v, f), fb.compute_vertex_normals(v, f), atol=1e-5)
    assert len(nat.feature_edges(v, f, 30.0, 1e-5)) == len(fb.feature_edges(v, f, 30.0, 1e-5))
    for _ in range(10):
        o = rng.normal(size=3) * 4
        d = -o + rng.normal(size=3) * 0.3
        d /= np.linalg.norm(d)
        h1, t1, i1 = nat.ray_mesh_intersect(o, d, v, f)
        h2, t2, i2 = fb.ray_mesh_intersect(o, d, v, f)
        assert h1 == h2
        if h1:
            assert t1 == pytest.approx(t2, rel=1e-6)


def test_input_validation():
    with pytest.raises((ValueError, IndexError)):
        nat.quat_normalize([1, 2, 3])
    with pytest.raises((ValueError, IndexError)):
        nat.mass_properties_solid(np.zeros((3, 3), np.float32), np.array([[0, 1, 5]], np.uint32), 1.0)


@pytest.mark.parametrize("method", ["explicit_euler", "semi_implicit_euler"])
def test_integration_kernels_agree(method):
    x0, v0, a = rng.normal(size=(7, 3)), rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    x1, v1, x2, v2 = x0.copy(), v0.copy(), x0.copy(), v0.copy()
    nat.integrate_linear(method, x1, v1, a, 0.013)
    fb.integrate_linear(method, x2, v2, a, 0.013)
    np.testing.assert_allclose(x1, x2, atol=1e-14)
    np.testing.assert_allclose(v1, v2, atol=1e-14)
    q0 = np.array([fb.quat_normalize(rng.normal(size=4)) for _ in range(7)])
    w = rng.normal(size=(7, 3))
    w[0] = 0.0
    q1, q2 = q0.copy(), q0.copy()
    nat.quat_integrate(q1, w, 0.05)
    fb.quat_integrate(q2, w, 0.05)
    np.testing.assert_allclose(q1, q2, atol=1e-12)
    with pytest.raises((TypeError, ValueError)):
        nat.integrate_linear(method, x0.astype(np.float32), v0, a, 0.1)


def test_batched_rotation_kernels_agree():
    q = np.array([fb.quat_normalize(rng.normal(size=4)) for _ in range(9)])
    v = rng.normal(size=(9, 3))
    t = rng.normal(size=(9, 3, 3))
    np.testing.assert_allclose(nat.quat_rotate_batch(q, v), fb.quat_rotate_batch(q, v), atol=1e-12)
    np.testing.assert_allclose(nat.rotate_tensors(q, t), fb.rotate_tensors(q, t), atol=1e-12)
    r = fb.quat_to_matrix(q[0])
    np.testing.assert_allclose(nat.rotate_tensors(q[:1], t[:1])[0], r @ t[0] @ r.T, atol=1e-12)
