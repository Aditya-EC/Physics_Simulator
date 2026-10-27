"""Joint constraint rows for one step (array interface).

This is the NumPy reference of the native routine ``joint_rows`` (native/src/joints.cpp): the same rows in the
same order, so both backends feed the solver identically.

Per joint: kind (JOINT_CODES), bodies a and b (solver body indices, b = −1 for the fixed world), anchors and axes in
each body's local frame, the reference rotation, flags (FLAG_LIMIT | FLAG_MOTOR | FLAG_ROPE), the limits, motor
speed and maximum force, and the rod/rope length. Per solver body: origin, centre of mass and rotation at the end
of the step (where the constraints are enforced) and optionally at the midpoint of the step, whose Jacobian gives
the impulse directions B (without it B = J).

Rows per joint type (the equality rows of one joint form one block, solved together):
  ball       3 point rows                                   block 3
  fixed      3 point rows + 3 angular rows                  block 6
  revolute   3 point rows + 2 rows keeping the axes aligned block 5, then angle limits and motor
  prismatic  3 angular rows + 2 rows ⟂ to the axis          block 5, then travel limits and motor
  distance   1 row along the anchor separation (a rope only when nearly taut, as an inequality)

Output dict: ia, ib, J (R, 12), B (R, 12), rhs, lo, hi, err, mode (0 = equality, 1 = inequality with position
correction, 2 = no position correction: motors), block (1 scalar, k at the first of k block rows, 0 inside),
joint (index of the joint each row belongs to). Distance joints with FLAG_SPRING have no row (the spring, applied
by the solver set-up, replaces the rod).
"""
from __future__ import annotations

import math

import numpy as np

JOINT_CODES = {"fixed": 0, "ball": 1, "revolute": 2, "prismatic": 3, "distance": 4}
BLOCK_SIZE = {0: 6, 1: 3, 2: 5, 3: 5, 4: 0}
FLAG_LIMIT, FLAG_MOTOR, FLAG_ROPE, FLAG_SPRING = 1, 2, 4, 8
LIMIT_MARGIN = 0.05             # limit rows are added when the coordinate is this close to a limit
INF = 1e30
E3 = np.eye(3)
ZERO = np.zeros(3)


def _cross(a, b) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def perp(n) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing n to an orthonormal basis."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = _cross(n, a)
    t1 = t1 / math.sqrt(float(t1 @ t1))
    return t1, _cross(n, t1)


def matrix_to_quat(m) -> np.ndarray:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / math.sqrt(float(q @ q))


def rotvec(r) -> np.ndarray:
    """Rotation vector (axis × angle) of a rotation matrix."""
    q = matrix_to_quat(r)
    if q[0] < 0:
        q = -q
    s = math.sqrt(float(q[1:] @ q[1:]))
    if s < 1e-12:
        return 2.0 * q[1:]
    return q[1:] / s * 2.0 * math.atan2(s, q[0])


def hinge_angle(g, axis_a_local, ref_rot) -> float:
    """Rotation of A relative to B about the hinge axis (the revolute joint coordinate; its rate is
    aw·(ω_a − ω_b))."""
    ua_l = perp(np.asarray(axis_a_local, float))[0]
    ua = g.Ra @ ua_l
    ub = g.Rb @ (np.asarray(ref_rot, float).T @ ua_l)
    return -math.atan2(float(_cross(ua, ub) @ g.aw), float(ua @ ub))


def slider_travel(g) -> float:
    """Displacement of A's anchor relative to B's along the slider axis (the prismatic joint coordinate)."""
    return -float(g.d @ g.aw)


class _Geo:
    """Anchor geometry of one joint in one configuration."""
    __slots__ = ("Ra", "Rb", "ra", "rb", "d", "aw", "bw")

    def __init__(self, a, b, anchor_a, anchor_b, axis_a, axis_b, origin, com, R):
        self.Ra = R[a]
        self.Rb = R[b] if b >= 0 else E3
        pa = origin[a] + self.Ra @ anchor_a
        pb = origin[b] + self.Rb @ anchor_b if b >= 0 else np.asarray(anchor_b, float)
        self.d = pb - pa
        self.ra = pa - com[a]
        self.rb = pb - com[b] if b >= 0 else ZERO
        self.aw = self.Ra @ axis_a
        self.bw = self.Rb @ axis_b


def joint_rows(kind, body_a, body_b, anchor_a, anchor_b, axis_a, axis_b, ref_rot, flags, lower, upper,
               motor_speed, motor_max, length, origin, com, R, dt: float, slop: float,
               mid_origin=None, mid_com=None, mid_R=None) -> dict:
    kind = np.asarray(kind, np.int64)
    body_a, body_b, flags = (np.asarray(x, np.int64) for x in (body_a, body_b, flags))
    anchor_a, anchor_b, axis_a, axis_b = (np.asarray(x, float).reshape(-1, 3)
                                          for x in (anchor_a, anchor_b, axis_a, axis_b))
    ref_rot = np.asarray(ref_rot, float).reshape(-1, 3, 3)
    origin, com = np.asarray(origin, float).reshape(-1, 3), np.asarray(com, float).reshape(-1, 3)
    R = np.asarray(R, float).reshape(-1, 3, 3)
    mid = mid_origin is not None
    if mid:
        mid_origin, mid_com = np.asarray(mid_origin, float).reshape(-1, 3), np.asarray(mid_com, float).reshape(-1, 3)
        mid_R = np.asarray(mid_R, float).reshape(-1, 3, 3)
    out = {k: [] for k in ("ia", "ib", "J", "B", "rhs", "lo", "hi", "err", "mode", "block", "joint")}
    pending = [0, 0]                 # [block size to open on the next row, rows left inside the open block]

    for j in range(len(kind)):
        a, b, t = int(body_a[j]), int(body_b[j]), int(kind[j])
        geo_args = (a, b, anchor_a[j], anchor_b[j], axis_a[j], axis_b[j])
        g = _Geo(*geo_args, origin, com, R)
        h = _Geo(*geo_args, mid_origin, mid_com, mid_R) if mid else g
        d = g.d

        def add(jac, rhs=0.0, lo=-INF, hi=INF, err=0.0, mode=0):
            if pending[0]:
                out["block"].append(pending[0])
                pending[1], pending[0] = pending[0] - 1, 0
            elif pending[1]:
                out["block"].append(0)
                pending[1] -= 1
            else:
                out["block"].append(1)
            row = np.concatenate(jac(g))
            for k, x in zip(("ia", "ib", "J", "B", "rhs", "lo", "hi", "err", "mode", "joint"),
                            (a, b, row, row if h is g else np.concatenate(jac(h)), rhs, lo, hi, err, mode, j)):
                out[k].append(x)

        def lin(dir_of, extra=False, **kw):
            """Row along dir_of(geometry) for the relative motion of the anchors."""
            def jac(x):
                n = dir_of(x)
                arm_a = x.ra + x.d if extra else x.ra
                return [-n, -_cross(arm_a, n), n if b >= 0 else ZERO, _cross(x.rb, n) if b >= 0 else ZERO]
            add(jac, **kw)

        def ang(dir_of, **kw):
            def jac(x):
                n = dir_of(x)
                return [ZERO, -n, ZERO, n if b >= 0 else ZERO]
            add(jac, **kw)

        def transported(tv):
            """A direction ⟂ to the hinge axis at the end of the step, carried continuously to each geometry."""
            def dir_of(x):
                u = tv - (tv @ x.aw) * x.aw
                return u / max(math.sqrt(float(u @ u)), 1e-12)
            return dir_of

        def fixed(e):
            return lambda _x: e

        def limits_and_motor(value, positive, negative):
            if flags[j] & FLAG_LIMIT:
                if value - lower[j] < LIMIT_MARGIN:
                    c = float(value - lower[j])
                    positive(rhs=-max(c, 0.0) / dt, lo=0.0, hi=INF, err=c, mode=1)
                if upper[j] - value < LIMIT_MARGIN:
                    c = float(upper[j] - value)
                    negative(rhs=-max(c, 0.0) / dt, lo=0.0, hi=INF, err=c, mode=1)
            if flags[j] & FLAG_MOTOR and motor_max[j] > 0:
                impulse = float(motor_max[j]) * dt
                positive(rhs=float(motor_speed[j]), lo=-impulse, hi=impulse, mode=2)

        size = BLOCK_SIZE[t]
        pending[0] = size if size >= 2 else 0
        if t in (0, 1, 2):
            for k in range(3):
                lin(fixed(E3[k]), err=float(d[k]))
        if t in (0, 3):
            theta = rotvec(g.Rb @ (g.Ra @ ref_rot[j]).T)
            for k in range(3):
                ang(fixed(E3[k]), err=float(theta[k]))
        if t == 2:
            aw = g.aw
            for tv in perp(aw):
                ang(transported(tv), err=float(_cross(aw, g.bw) @ tv))
            angle = hinge_angle(g, axis_a[j], ref_rot[j])          # row ang(-aw) = dθ/dt
            limits_and_motor(angle, lambda **kw: ang(lambda x: -x.aw, **kw), lambda **kw: ang(lambda x: x.aw, **kw))
        if t == 3:
            aw = g.aw
            for tv in perp(aw):
                lin(transported(tv), extra=True, err=float(d @ tv))
            travel = slider_travel(g)
            limits_and_motor(travel, lambda **kw: lin(lambda x: -x.aw, extra=True, **kw),
                             lambda **kw: lin(lambda x: x.aw, extra=True, **kw))
        if t == 4:
            def along(x):
                n = math.sqrt(float(x.d @ x.d))
                return x.d / n if n > 1e-9 else x.aw
            dist = math.sqrt(float(d @ d))
            if flags[j] & FLAG_SPRING:
                pass                                 # a soft distance joint: the spring replaces the rod
            elif flags[j] & FLAG_ROPE:
                if dist > length[j] - slop:          # near taut: may extend only up to the length
                    slack = float(length[j] - dist)
                    lin(lambda x: -along(x), rhs=-max(slack, 0.0) / dt, lo=0.0, hi=INF, err=slack, mode=1)
            else:
                lin(along, err=float(dist - length[j]))

    m = len(out["ia"])
    return {"ia": np.array(out["ia"], np.int64), "ib": np.array(out["ib"], np.int64),
            "J": np.array(out["J"], float).reshape(m, 12), "B": np.array(out["B"], float).reshape(m, 12),
            "rhs": np.array(out["rhs"], float), "lo": np.array(out["lo"], float), "hi": np.array(out["hi"], float),
            "err": np.array(out["err"], float), "mode": np.array(out["mode"], np.int64),
            "block": np.array(out["block"], np.int64), "joint": np.array(out["joint"], np.int64)}


def pack_joints(entries, index: dict) -> dict:
    """Arrays for joint_rows from physics.setup JointEntry objects (`index`: body key → solver body index)."""
    n = len(entries)
    out = {"kind": np.array([JOINT_CODES[je.joint.type] for je in entries], np.int64),
           "body_a": np.array([index[je.body_a] for je in entries], np.int64),
           "body_b": np.array([index[je.body_b] if je.body_b else -1 for je in entries], np.int64)}
    for key in ("anchor_a", "anchor_b", "axis_a", "axis_b"):
        out[key] = np.array([getattr(je, key) for je in entries], float).reshape(n, 3)
    out["ref_rot"] = np.array([je.ref_rot for je in entries], float).reshape(n, 3, 3)
    out["flags"] = np.array([(FLAG_LIMIT if je.joint.limit_enabled else 0) | (FLAG_ROPE if je.joint.rope else 0)
                             | (FLAG_MOTOR if je.joint.motor_enabled else 0)
                             | (FLAG_SPRING if je.joint.is_soft else 0) for je in entries], np.int64)
    for key in ("lower", "upper", "motor_speed", "motor_max", "length"):
        out[key] = np.array([float(getattr(je.joint, key)) for je in entries], float)
    return out
