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
def test_every_pair_kind_is_exercised():
    seen = set()
    for seed in range(12):
        args = _scene(seed)
        d = numerics.contact_rows(*args)
        for p, q in d["pairs"]:
            seen.add(tuple(sorted((KINDS[args[0][p]], KINDS[args[0][q]]))))
    expected = {tuple(sorted((a, b))) for a in KINDS for b in KINDS if not (a == b == "plane")}
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
