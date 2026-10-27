"""Joint constraint rows: C++ ≡ NumPy reference (psim/physics/joint_kernel.py), and input validation."""
from __future__ import annotations

import numpy as np
import pytest

from psim import numerics
from psim.physics.joint_kernel import FLAG_LIMIT, FLAG_MOTOR, FLAG_ROPE, joint_rows as reference, rotvec

native = pytest.mark.skipif(numerics.BACKEND != "native", reason="native core not built")


def _rotations(rng, n):
    q = rng.normal(size=(n, 4))
    return np.array([numerics.FALLBACK.quat_to_matrix(x / np.linalg.norm(x)) for x in q], float)


def _config(rng, nb):
    return rng.normal(size=(nb, 3)), rng.normal(size=(nb, 3)) * 0.2, _rotations(rng, nb)


def _joints(seed: int, n: int = 40, nb: int = 8):
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 5, n)
    body_a = rng.integers(0, nb, n)
    body_b = np.where(rng.random(n) < 0.3, -1, (body_a + rng.integers(1, nb, n)) % nb)
    axis_a = rng.normal(size=(n, 3))
    axis_a /= np.linalg.norm(axis_a, axis=1, keepdims=True)
    axis_b = rng.normal(size=(n, 3))
    axis_b /= np.linalg.norm(axis_b, axis=1, keepdims=True)
    flags = rng.integers(0, 8, n)
    # narrow limit ranges around 0 so that both limit rows (and motors) appear
    lower, upper = rng.uniform(-0.04, 0.0, n), rng.uniform(0.0, 0.04, n)
    lower[::3], upper[::3] = -3.0, 3.0
    length = rng.uniform(0.1, 3.0, n)
    args = [kind, body_a, body_b, rng.normal(size=(n, 3)), rng.normal(size=(n, 3)), axis_a, axis_b,
            _rotations(rng, n), flags, lower, upper, rng.normal(size=n), rng.uniform(-1, 5, n), length]
    return args, _config(rng, nb), _config(rng, nb)


def _compare(nat, ref):
    for key in ("ia", "ib", "mode", "block", "joint"):
        np.testing.assert_array_equal(nat[key], ref[key], err_msg=key)
    for key in ("J", "B", "rhs", "lo", "hi", "err"):
        np.testing.assert_allclose(nat[key], ref[key], rtol=1e-9, atol=1e-11, err_msg=key)


@native
@pytest.mark.parametrize("seed", range(8))
def test_native_joint_rows_match_the_reference(seed):
    args, end, mid = _joints(seed)
    for m in (mid, (None, None, None)):
        ref = reference(*args, *end, 0.005, 0.002, *m)
        nat = numerics.joint_rows(*args, *end, 0.005, 0.002, *m)
        _compare(nat, ref)
        assert set(ref["mode"].tolist()) == {0, 1, 2}               # equality, limit/rope and motor rows
        assert {1, 3, 5, 6} <= set(ref["block"].tolist())            # ball, fixed, revolute/prismatic blocks
        if m[0] is None:
            np.testing.assert_array_equal(ref["B"], ref["J"])
        else:
            assert np.abs(ref["B"] - ref["J"]).max() > 1e-3


def test_row_counts_and_blocks():
    nb = 2
    rng = np.random.default_rng(1)
    origin, com, R = _config(rng, nb)
    end = (origin, com, np.tile(R[0], (nb, 1, 1)))                 # same orientation: hinge angle 0
    one = lambda k, flags=0, lower=-1.0, upper=1.0, length=1.0, b=1: (           # noqa: E731
        [k], [0], [b], np.zeros((1, 3)), np.zeros((1, 3)), [[0, 0, 1.0]], [[0, 0, 1.0]], np.eye(3)[None],
        [flags], [lower], [upper], [0.5], [2.0], [length])
    counts = {0: 6, 1: 3, 2: 5, 3: 5, 4: 1}
    for k, c in counts.items():
        d = reference(*one(k), *end, 0.01, 0.002)
        assert len(d["ia"]) == c
        assert d["block"].tolist() == ([c] + [0] * (c - 1) if c > 1 else [1])
    d = reference(*one(2, FLAG_MOTOR | FLAG_LIMIT, lower=-0.01, upper=0.01), *end, 0.01, 0.002)
    assert d["block"].tolist()[5:] == [1, 1, 1] and d["mode"].tolist()[5:] == [1, 1, 2]
    assert d["lo"][-1] == pytest.approx(-0.02) and d["rhs"][-1] == 0.5             # motor: ±max·Δt at its speed
    # a slack rope has no row; a taut one is an inequality
    pts = (np.zeros((nb, 3)), np.zeros((nb, 3)), np.tile(np.eye(3), (nb, 1, 1)))
    slack = list(one(4, FLAG_ROPE, length=2.0))
    slack[4] = np.array([[1.0, 0, 0]])
    assert len(reference(*slack, *pts, 0.01, 0.002)["ia"]) == 0
    slack[13] = [0.5]
    d = reference(*slack, *pts, 0.01, 0.002)
    assert d["mode"].tolist() == [1] and d["err"][0] == pytest.approx(-0.5)


def test_rotvec():
    from psim.numerics import FALLBACK
    for axis, ang in (([0, 0, 1], 0.3), ([1, 1, 0], 2.5), ([0, 1, 0], 1e-14), ([1, 0, 0], 3.1)):
        r = FALLBACK.quat_to_matrix(FALLBACK.quat_from_axis_angle(axis, ang))
        np.testing.assert_allclose(rotvec(np.asarray(r)), np.asarray(axis) / np.linalg.norm(axis) * ang, atol=1e-12)


@native
def test_invalid_joint_input_is_rejected():
    args, end, _ = _joints(0)
    bad = list(args)
    bad[0] = np.full(len(args[0]), 7)
    with pytest.raises(ValueError, match="kind"):
        numerics.joint_rows(*bad, *end, 0.005, 0.002)
    bad = list(args)
    bad[1] = np.full(len(args[0]), 99)
    with pytest.raises(ValueError, match="missing body"):
        numerics.joint_rows(*bad, *end, 0.005, 0.002)
    bad = list(args)
    bad[3] = bad[3][:-1]
    with pytest.raises(ValueError, match="same number"):
        numerics.joint_rows(*bad, *end, 0.005, 0.002)
    with pytest.raises(ValueError, match="dt"):
        numerics.joint_rows(*args, *end, 0.0, 0.002)
