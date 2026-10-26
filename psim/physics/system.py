"""The rigid-body constraint stage: contacts and joints, applied after each integration step.

Per step (state at t + Δt, after the integrator, update and prescribed processes):

1. Contacts are generated between the colliders at their new poses.
2. Contacts (normal + two friction rows) and joints (point, angular, hinge, slider,
   distance, limit and motor rows) become generic constraint rows J·v = rhs with bounds.
3. Velocity pass: projected Gauss–Seidel on the body velocities (restitution, friction, motors).
   Each dynamic body is then displaced by Δv·Δt (and rotated by Δω·Δt), which makes the
   result the symplectic-Euler step with constraint impulses.
4. Position pass ("split impulse"): the same rows on pseudo-velocities that remove the remaining
   penetration and joint drift (Baumgarte factor β, contact slop) without adding kinetic energy.

The velocity change Δv/Δt of each body is reported as its constraint acceleration (contact and
joint forces divided by mass), so recordings and net-force plots include contact forces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .. import numerics
from .collision import WorldShape, broadphase, collide
from .setup import PhysicsSetup

INF = 1e30
E3 = np.eye(3)


@dataclass
class PhysicsSettings:
    iterations: int = 20                 # velocity iterations
    position_iterations: int = 8
    baumgarte: float = 1.0               # fraction of the position error corrected per step (split impulse:
                                         # the correction never adds velocity, so a full correction is safe)
    slop: float = 0.002                  # allowed penetration (m) before position correction
    restitution_threshold: float = 0.2   # m/s: slower impacts do not bounce (avoids jitter at rest)


def _cross(a, b) -> np.ndarray:
    """Cross product of two 3-vectors (np.cross has a large per-call overhead for single vectors)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _perp(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = _cross(n, a)
    t1 /= np.linalg.norm(t1)
    return t1, _cross(n, t1)


def _rotvec(r: np.ndarray) -> np.ndarray:
    """Rotation vector (axis × angle) of a rotation matrix."""
    q = np.asarray(numerics.matrix_to_quat(r), float)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-12:
        return 2.0 * q[1:]
    return q[1:] / s * 2.0 * math.atan2(s, q[0])


class _Rows:
    def __init__(self) -> None:
        self.ia, self.ib, self.J, self.rhs, self.lo, self.hi, self.parent, self.mu = ([] for _ in range(8))
        self.err, self.mode = [], []        # position error C and correction mode (eq | ineq | none)
        self.block = []                     # 1 scalar row; k at the first of k rows solved together; 0 inside
        self.B = []                         # impulse directions (the row's J unless given)
        self.has_impulse = False

    def begin_block(self, size: int) -> None:
        """The next `size` rows (unbounded, same bodies) are solved as one block (0 = no block)."""
        self._pending = size if size >= 2 else 0

    def add(self, a, b, lin_a, ang_a, lin_b, ang_b, rhs=0.0, lo=-INF, hi=INF, parent=-1, mu=0.0, err=0.0,
            mode="eq", impulse=None) -> int:
        pending = getattr(self, "_pending", 0)
        if pending:
            self.block.append(pending)
            self._pending = 0
            self._inside = pending - 1
        elif getattr(self, "_inside", 0):
            self.block.append(0)
            self._inside -= 1
        else:
            self.block.append(1)
        self.ia.append(a)
        self.ib.append(b)
        self.J.append(np.concatenate([lin_a, ang_a, lin_b, ang_b]))
        self.B.append(self.J[-1] if impulse is None else np.asarray(impulse, float))
        self.has_impulse = self.has_impulse or impulse is not None
        self.rhs.append(rhs)
        self.lo.append(lo)
        self.hi.append(hi)
        self.parent.append(parent)
        self.mu.append(mu)
        self.err.append(err)
        self.mode.append(mode)
        return len(self.ia) - 1

    def arrays(self):
        m = len(self.ia)
        return (np.array(self.ia, np.int64), np.array(self.ib, np.int64),
                np.array(self.J, float).reshape(m, 12), np.array(self.rhs, float), np.array(self.lo, float),
                np.array(self.hi, float), np.array(self.parent, np.int64), np.array(self.mu, float),
                np.array(self.block, np.int64))


class PhysicsSystem:
    def __init__(self, setup: PhysicsSetup, bodies, trans: np.ndarray, rot: np.ndarray,
                 settings: PhysicsSettings | None = None) -> None:
        self.setup = setup
        self.settings = settings or PhysicsSettings()
        self.keys = list(setup.bodies)
        self.index = {k: i for i, k in enumerate(self.keys)}
        n = len(self.keys)
        self.engine_rows = np.array([setup.bodies[k].engine_index if setup.bodies[k].engine_index is not None
                                     else -1 for k in self.keys], int)
        self.dynamic = np.array([setup.bodies[k].role == "dynamic" for k in self.keys], bool)
        self.inv_mass = np.zeros(n)
        self.inv_inertia_local = np.zeros((n, 3, 3))
        self.com_local = np.zeros((n, 3))
        for i, k in enumerate(self.keys):
            e = self.engine_rows[i]
            if e >= 0:
                self.com_local[i] = bodies.com_local[e]
                if self.dynamic[i] and trans[e]:
                    self.inv_mass[i] = bodies.inv_mass[e]
                if self.dynamic[i] and rot[e]:
                    self.inv_inertia_local[i] = bodies.inv_inertia_local[e]
        connected = set()
        for je in setup.joints:
            if not je.joint.collide_connected:
                connected.add(frozenset((je.body_a, je.body_b or "@world")))
        self.excluded = connected
        self.touching: set[frozenset] = set()
        self.new_contacts: list[tuple[str, str]] = []   # newly touching collider pairs (for the event log)
        self.last_contacts: list = []                   # (source a, source b, point, normal, depth, impulse)
        self._last_dv, self._last_dw = np.zeros((n, 3)), np.zeros((n, 3))   # constraint Δv, Δω of the last step

    # ---- state -------------------------------------------------------------------------------------------------
    def _gather(self, s):
        n = len(self.keys)
        origin, com = np.zeros((n, 3)), np.zeros((n, 3))
        R = np.tile(E3, (n, 1, 1))
        v, w = np.zeros((n, 3)), np.zeros((n, 3))
        for i, k in enumerate(self.keys):
            e = self.engine_rows[i]
            if e >= 0:
                R[i] = np.asarray(numerics.quat_to_matrix(s.q[e]), float)
                com[i] = s.x[e]
                origin[i] = s.x[e] - R[i] @ self.com_local[i]
                v[i], w[i] = s.v[e], s.w[e]
            else:
                p, q = self.setup.bodies[k].pose
                R[i] = np.asarray(numerics.quat_to_matrix(q), float)
                origin[i] = p
                com[i] = p
        inv_i = np.einsum("nij,njk,nlk->nil", R, self.inv_inertia_local, R)
        return origin, com, R, v, w, inv_i

    def _midpoint(self, s0, s1, dt: float):
        """(origin, com, R) halfway between the start of the step and its predicted end: the integrated state
        `s1` moved on by the previous step's constraint velocity change (which varies slowly), since the joint
        impulse of this step will move it too. Static bodies do not move."""
        o0, c0, R0, *_ = self._gather(s0)
        o1, c1, R1, *_ = self._gather(s1)
        c1 = c1 + self._last_dv * dt
        com = 0.5 * (c0 + c1)
        R = R1.copy()
        origin = o1.copy()
        for i in range(len(self.keys)):
            e = self.engine_rows[i]
            if e < 0:
                continue
            q0, q1 = np.asarray(s0.q[e], float), np.ascontiguousarray(s1.q[e:e + 1], float).copy()
            if np.any(self._last_dw[i]):
                numerics.quat_integrate(q1, np.ascontiguousarray(self._last_dw[i][None, :]), dt)
            q1 = q1[0]
            q = q0 + (q1 if q0 @ q1 >= 0 else -q1)
            R[i] = np.asarray(numerics.quat_to_matrix(q / np.linalg.norm(q)), float)
            origin[i] = com[i] - R[i] @ self.com_local[i]
        return origin, com, R

    # ---- step ----------------------------------------------------------------------------------------------------
    def step(self, s, dt: float, start=None) -> tuple[np.ndarray, np.ndarray]:
        """Apply contacts and joints to the engine state `s` in place. Returns Δv, Δω per engine row.

        `start` is the state at the beginning of the step: joint impulses then act along the Jacobian of the
        midpoint configuration, which turns velocities with the joint instead of projecting them onto the
        end-of-step tangent space (a projection loses ≈ KE·ω²Δt per second, e.g. a damped pendulum).
        """
        st = self.settings
        origin, com, R, v, w, inv_i = self._gather(s)
        rows = _Rows()
        contacts = self._contacts(rows, origin, com, R, v, w, dt)
        mid = self._midpoint(start, s, dt) if start is not None and self.setup.joints else None
        self._joints(rows, (origin, com, R), mid, dt)
        n_eng = len(s.v)
        dv_eng, dw_eng = np.zeros((n_eng, 3)), np.zeros((n_eng, 3))
        if not rows.ia:
            self.last_contacts = []
            self._last_dv, self._last_dw = np.zeros_like(self._last_dv), np.zeros_like(self._last_dw)
            return dv_eng, dw_eng
        ia, ib, J, rhs, lo, hi, parent, mu, block = rows.arrays()
        v1, w1 = v.copy(), w.copy()
        B = np.array(rows.B, float).reshape(-1, 12) if rows.has_impulse else None
        lam = numerics.pgs_solve(ia, ib, J, rhs, lo, hi, parent, mu, block, self.inv_mass, inv_i, v1, w1,
                                 st.iterations, B)
        dv, dw = v1 - v, w1 - w
        self._last_dv, self._last_dw = dv, dw
        # position pass on pseudo-velocities: remaining error after the Δv·Δt displacement
        jd = self._row_velocities(ia, ib, J, dv, dw)
        err = np.array(rows.err) + jd * dt
        rhs_p = np.zeros(len(err))
        lo_p, hi_p = lo.copy(), hi.copy()
        par_p = np.full(len(err), -1, np.int64)
        for r, mode in enumerate(rows.mode):
            if mode == "eq":
                rhs_p[r] = -st.baumgarte * err[r] / dt
            elif mode == "ineq":
                rhs_p[r] = st.baumgarte * max(-err[r] - st.slop, 0.0) / dt
            else:  # friction and motors do not correct positions
                lo_p[r], hi_p[r] = 0.0, 0.0
        vp, wp = np.zeros_like(v), np.zeros_like(w)
        if np.any(rhs_p):
            numerics.pgs_solve(ia, ib, J, rhs_p, lo_p, hi_p, par_p, np.zeros(len(err)), block, self.inv_mass, inv_i,
                               vp, wp, st.position_iterations)
        for i in np.flatnonzero(self.dynamic):
            e = self.engine_rows[i]
            s.v[e] += dv[i]
            s.w[e] += dw[i]
            s.x[e] += (dv[i] + vp[i]) * dt
            spin = dw[i] + wp[i]
            if np.any(spin):
                q = np.ascontiguousarray(s.q[e:e + 1].copy())
                numerics.quat_integrate(q, np.ascontiguousarray(spin[None, :]), dt)
                s.q[e] = q[0]
            dv_eng[e], dw_eng[e] = dv[i], dw[i]
        self.last_contacts = [(c[0], c[1], c[2], c[3], c[4], float(lam[c[5]])) for c in contacts]
        return dv_eng, dw_eng

    @staticmethod
    def _row_velocities(ia, ib, J, v, w) -> np.ndarray:
        out = np.zeros(len(ia))
        for r in range(len(ia)):
            if ia[r] >= 0:
                out[r] += J[r, :3] @ v[ia[r]] + J[r, 3:6] @ w[ia[r]]
            if ib[r] >= 0:
                out[r] += J[r, 6:9] @ v[ib[r]] + J[r, 9:] @ w[ib[r]]
        return out

    # ---- contacts ------------------------------------------------------------------------------------------------
    def _contacts(self, rows: _Rows, origin, com, R, v, w, dt) -> list:
        st = self.settings
        entries = self.setup.colliders
        shapes = []
        for c in entries:
            i = self.index[c.body]
            shapes.append(WorldShape.place(c.shape, origin[i] + R[i] @ c.rel_pos, R[i] @ c.rel_rot))
        out, touching = [], set()
        for p, q in broadphase(shapes, margin=st.slop):
            ca, cb = entries[p], entries[q]
            a, b = self.index[ca.body], self.index[cb.body]
            if a == b or not (self.dynamic[a] or self.dynamic[b]):
                continue
            if frozenset((ca.body, cb.body)) in self.excluded:
                continue
            found = collide(shapes[p], shapes[q])
            if not found:
                continue
            pair = frozenset((ca.source, cb.source))
            touching.add(pair)
            if pair not in self.touching:
                self.new_contacts.append((ca.source, cb.source))
            e = max(ca.restitution, cb.restitution)
            mu = math.sqrt(max(ca.friction, 0.0) * max(cb.friction, 0.0))
            for c in found:
                n = c.normal
                ra, rb = c.point - com[a], c.point - com[b]
                vn0 = float(((v[b] + _cross(w[b], rb)) - (v[a] + _cross(w[a], ra))) @ n)
                target = -e * vn0 if vn0 < -st.restitution_threshold else 0.0
                k = rows.add(a, b, -n, -_cross(ra, n), n, _cross(rb, n), rhs=target, lo=0.0, hi=INF,
                             err=-c.depth, mode="ineq")
                if mu > 0:
                    for t in _perp(n):
                        rows.add(a, b, -t, -_cross(ra, t), t, _cross(rb, t), parent=k, mu=mu, mode="none")
                out.append((ca.source, cb.source, c.point, n, c.depth, k))
        self.touching = touching
        return out

    # ---- joints ----------------------------------------------------------------------------------------------------
    def _joint_geometry(self, je, a: int, b: int, origin, com, R) -> SimpleNamespace:
        Ra = R[a]
        Rb = R[b] if b >= 0 else E3
        pa = origin[a] + Ra @ je.anchor_a
        pb = origin[b] + Rb @ je.anchor_b if b >= 0 else je.anchor_b
        d = pb - pa
        return SimpleNamespace(Ra=Ra, Rb=Rb, ra=pa - com[a], rb=pb - com[b] if b >= 0 else np.zeros(3), d=d,
                               aw=Ra @ je.axis_a, bw=Rb @ je.axis_b)

    def _joints(self, rows: _Rows, geo, mid, dt) -> None:
        """Joint rows. `geo` = (origin, com, R) at the end of the step, where the constraints are enforced;
        `mid` = the same at the midpoint of the step, whose Jacobian gives the impulse directions (None = geo)."""
        zero = np.zeros(3)
        for je in self.setup.joints:
            j = je.joint
            a = self.index[je.body_a]
            b = self.index[je.body_b] if je.body_b else -1
            g = self._joint_geometry(je, a, b, *geo)
            h = g if mid is None else self._joint_geometry(je, a, b, *mid)
            d = g.d

            def lin(dir_of, extra=False, **kw):
                """Linear row along dir_of(geometry) for the relative motion of the anchors."""
                def jac(x):
                    n = dir_of(x)
                    arm_a = x.ra + x.d if extra else x.ra
                    return [-n, -_cross(arm_a, n), n if b >= 0 else zero, _cross(x.rb, n) if b >= 0 else zero]
                return rows.add(a, b, *jac(g), impulse=None if h is g else np.concatenate(jac(h)), **kw)

            def ang(dir_of, **kw):
                def jac(x):
                    n = dir_of(x)
                    return [zero, -n, zero, n if b >= 0 else zero]
                return rows.add(a, b, *jac(g), impulse=None if h is g else np.concatenate(jac(h)), **kw)

            def transported(t):
                """A direction ⟂ to the hinge axis at the end of the step, carried continuously to `x`."""
                def dir_of(x):
                    u = t - (t @ x.aw) * x.aw
                    return u / max(float(np.linalg.norm(u)), 1e-12)
                return dir_of

            def fixed(e):
                return lambda _x: e

            # the equality rows of one joint are solved together (exact for the joint, fast convergence
            # even with very different masses): ball 3, revolute 5, prismatic 5, fixed 6
            rows.begin_block({"ball": 3, "revolute": 5, "prismatic": 5, "fixed": 6}.get(j.type, 0))
            if j.type in ("ball", "fixed", "revolute"):
                for k in range(3):
                    lin(fixed(E3[k]), err=d[k])
            if j.type in ("fixed", "prismatic"):
                theta = _rotvec(g.Rb @ (g.Ra @ je.ref_rot).T)
                for k in range(3):
                    ang(fixed(E3[k]), err=theta[k])
            if j.type == "revolute":
                aw = g.aw
                for t in _perp(aw):
                    ang(transported(t), err=float(_cross(aw, g.bw) @ t))
                ua_l = _perp(je.axis_a)[0]
                ua = g.Ra @ ua_l
                ub = g.Rb @ (je.ref_rot.T @ ua_l)
                # joint coordinate: rotation of A relative to B about the hinge axis (row ang(-aw) = dθ/dt)
                angle = -math.atan2(float(_cross(ua, ub) @ aw), float(ua @ ub))
                self._limits_and_motor(rows, j, angle, dt, lambda **kw: ang(lambda x: -x.aw, **kw),
                                       lambda **kw: ang(lambda x: x.aw, **kw))
            if j.type == "prismatic":
                aw = g.aw
                for t in _perp(aw):
                    lin(transported(t), extra=True, err=float(d @ t))
                # joint coordinate: displacement of A's anchor relative to B's along the axis
                travel = -float(d @ aw)
                self._limits_and_motor(rows, j, travel, dt, lambda **kw: lin(lambda x: -x.aw, extra=True, **kw),
                                       lambda **kw: lin(lambda x: x.aw, extra=True, **kw))
            if j.type == "distance":
                def along(x):
                    n = float(np.linalg.norm(x.d))
                    return x.d / n if n > 1e-9 else x.aw
                length = float(np.linalg.norm(d))
                if j.rope:
                    if length > j.length - self.settings.slop:   # near taut: may extend only up to the length
                        slack = j.length - length
                        lin(lambda x: -along(x), rhs=-max(slack, 0.0) / dt, lo=0.0, hi=INF, err=slack, mode="ineq")
                else:
                    lin(along, err=length - j.length)

    @staticmethod
    def _limits_and_motor(rows: _Rows, j, value: float, dt: float, positive, negative) -> None:
        """Limit rows (active near the limits) and the motor row for one joint coordinate."""
        if j.limit_enabled:
            margin = 0.05
            if value - j.lower < margin:
                c = value - j.lower
                positive(rhs=-max(c, 0.0) / dt, lo=0.0, hi=INF, err=c, mode="ineq")
            if j.upper - value < margin:
                c = j.upper - value
                negative(rhs=-max(c, 0.0) / dt, lo=0.0, hi=INF, err=c, mode="ineq")
        if j.motor_enabled and j.motor_max > 0:
            impulse = j.motor_max * dt
            positive(rhs=j.motor_speed, lo=-impulse, hi=impulse, mode="none")

    def take_new_contacts(self) -> list[tuple[str, str]]:
        out, self.new_contacts = self.new_contacts, []
        return out
