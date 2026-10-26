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
from .contact_kernel import COLLIDER_CODES
from .hull import box_hull, pack_hulls
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
    continuous: bool = True              # speculative contacts for pairs that could tunnel in one step


def _quat_matrices(q: np.ndarray) -> np.ndarray:
    """(n, 4) unit quaternions (w, x, y, z) → (n, 3, 3) rotation matrices (vectorised)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


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

    def extend(self, d: dict) -> None:
        """Append the scalar rows produced by numerics.contact_rows (they come first, so parents need no offset)."""
        if self.ia:
            raise ValueError("contact rows must be added before any other rows")
        m = len(d["ia"])
        self.ia += d["ia"].tolist()
        self.ib += d["ib"].tolist()
        self.J += list(d["J"])
        self.B += list(d["J"])
        self.rhs += d["rhs"].tolist()
        self.lo += d["lo"].tolist()
        self.hi += d["hi"].tolist()
        self.parent += d["parent"].tolist()
        self.mu += d["mu"].tolist()
        self.err += d["err"].tolist()
        self.mode += ["ineq" if x == 1 else "none" for x in d["mode"].tolist()]
        self.block += [1] * m

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
        # static bodies never move: their poses are computed once
        self._sim_rows = np.flatnonzero(self.engine_rows >= 0)
        self._static_R = np.tile(E3, (n, 1, 1))
        self._static_origin = np.zeros((n, 3))
        for i, k in enumerate(self.keys):
            if self.engine_rows[i] < 0:
                p, q = setup.bodies[k].pose
                self._static_R[i] = _quat_matrices(np.asarray(q, float)[None, :])[0]
                self._static_origin[i] = p
        connected = set()
        for je in setup.joints:
            if not je.joint.collide_connected:
                connected.add(frozenset((je.body_a, je.body_b or "@world")))
        self.excluded = connected
        # collider data that does not change during the run (placed in the world every step)
        cols = setup.colliders
        self.col_kind = np.array([COLLIDER_CODES[c.shape.kind] for c in cols], np.int64)
        # hull table: convex colliders, plus boxes when hulls are present (hull–box pairs use the box as a hull)
        self.col_hull_id = np.full(len(cols), -1, np.int64)
        hull_list = []
        if any(c.shape.kind == "convex" for c in cols):
            for i, c in enumerate(cols):
                if c.shape.kind in ("convex", "box"):
                    self.col_hull_id[i] = len(hull_list)
                    hull_list.append(c.shape.hull if c.shape.kind == "convex" else box_hull(c.shape.half_extents))
        self.hulls = pack_hulls(hull_list) if hull_list else None
        self.col_body = np.array([self.index[c.body] for c in cols], np.int64)
        self.col_rel_pos = np.array([c.rel_pos for c in cols], float).reshape(-1, 3)
        self.col_rel_rot = np.array([c.rel_rot for c in cols], float).reshape(-1, 3, 3)
        self.col_center = np.array([c.shape.center for c in cols], float).reshape(-1, 3)
        self.col_radius = np.array([c.shape.radius for c in cols], float)
        self.col_half = np.array([c.shape.half_extents if c.shape.half_extents is not None else np.zeros(3)
                                  for c in cols], float).reshape(-1, 3)
        self.col_half_len = np.array([c.shape.half_length for c in cols], float)
        self.col_restitution = np.array([c.restitution for c in cols], float)
        self.col_friction = np.array([c.friction for c in cols], float)
        self.col_bound = np.array([0.0 if c.shape.kind == "plane" else c.shape.bounding_radius() for c in cols],
                                  float)
        self.excluded_index = np.array([sorted(self.index[k] for k in pair) for pair in connected
                                        if all(k in self.index for k in pair) and len(pair) == 2],
                                       np.int64).reshape(-1, 2)
        self.touching: set[frozenset] = set()
        self.new_contacts: list[tuple[str, str]] = []   # newly touching collider pairs (for the event log)
        self.last_contacts: list = []                   # (source a, source b, point, normal, depth, impulse)
        self._last_dv, self._last_dw = np.zeros((n, 3)), np.zeros((n, 3))   # constraint Δv, Δω of the last step

    # ---- state -------------------------------------------------------------------------------------------------
    def _gather(self, s):
        """Origins, centres of mass, rotations, velocities and world inverse inertias of all solver bodies."""
        origin, com, R = self._static_origin.copy(), self._static_origin.copy(), self._static_R.copy()
        n = len(self.keys)
        v, w = np.zeros((n, 3)), np.zeros((n, 3))
        sim, e = self._sim_rows, self.engine_rows[self._sim_rows]
        if len(sim):
            R[sim] = _quat_matrices(s.q[e])
            com[sim] = s.x[e]
            origin[sim] = s.x[e] - np.einsum("nij,nj->ni", R[sim], self.com_local[sim])
            v[sim], w[sim] = s.v[e], s.w[e]
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
        contacts = self._contacts(rows, origin, com, R, v, w, dt, start)
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
        mode = np.array(rows.mode)
        eq, ineq = mode == "eq", mode == "ineq"
        rhs_p = np.zeros(len(err))
        rhs_p[eq] = -st.baumgarte * err[eq] / dt
        rhs_p[ineq] = st.baumgarte * np.maximum(-err[ineq] - st.slop, 0.0) / dt
        lo_p, hi_p = lo.copy(), hi.copy()
        other = ~(eq | ineq)              # friction and motors do not correct positions
        lo_p[other], hi_p[other] = 0.0, 0.0
        par_p = np.full(len(err), -1, np.int64)
        vp, wp = np.zeros_like(v), np.zeros_like(w)
        if np.any(rhs_p):
            numerics.pgs_solve(ia, ib, J, rhs_p, lo_p, hi_p, par_p, np.zeros(len(err)), block, self.inv_mass, inv_i,
                               vp, wp, st.position_iterations)
        dyn = np.flatnonzero(self.dynamic)
        e = self.engine_rows[dyn]
        s.v[e] += dv[dyn]
        s.w[e] += dw[dyn]
        s.x[e] += (dv[dyn] + vp[dyn]) * dt
        spin = dw[dyn] + wp[dyn]
        turning = np.any(spin != 0.0, axis=1)
        if turning.any():
            q = np.ascontiguousarray(s.q[e[turning]])
            numerics.quat_integrate(q, np.ascontiguousarray(spin[turning]), dt)
            s.q[e[turning]] = q
        dv_eng[e], dw_eng[e] = dv[dyn], dw[dyn]
        self.last_contacts = [(c[0], c[1], c[2], c[3], c[4], float(lam[c[5]])) for c in contacts]
        return dv_eng, dw_eng

    @staticmethod
    def _row_velocities(ia, ib, J, v, w) -> np.ndarray:
        """J·[v_a, w_a, v_b, w_b] for every row (index −1 = the fixed world)."""
        out = np.zeros(len(ia))
        a, b = ia >= 0, ib >= 0
        out[a] += np.einsum("ij,ij->i", J[a, :3], v[ia[a]]) + np.einsum("ij,ij->i", J[a, 3:6], w[ia[a]])
        out[b] += np.einsum("ij,ij->i", J[b, 6:9], v[ib[b]]) + np.einsum("ij,ij->i", J[b, 9:], w[ib[b]])
        return out

    # ---- contacts ------------------------------------------------------------------------------------------------
    def _contacts(self, rows: _Rows, origin, com, R, v, w, dt, start=None) -> list:
        """Contacts of all colliders (numerics.contact_rows: C++ or the NumPy kernel) and their rows."""
        st = self.settings
        entries = self.setup.colliders
        if not entries:
            self.touching = set()
            return []
        b = self.col_body
        rot = np.einsum("nij,njk->nik", R[b], self.col_rel_rot)
        center = origin[b] + np.einsum("nij,nj->ni", R[b], self.col_rel_pos) + np.einsum("nij,nj->ni", rot,
                                                                                         self.col_center)
        sweep = center0 = rot0 = com0 = None
        if st.continuous and dt > 0:   # how far each shape's surface can move this step: (|v| + |ω|·reach)·Δt
            reach = np.linalg.norm(center - com[b], axis=1) + self.col_bound
            sweep = (np.linalg.norm(v[b], axis=1) + np.linalg.norm(w[b], axis=1) * reach) * dt
            if start is not None:      # speculative pairs are tested where the step began
                origin0, com0, R0, *_ = self._gather(start)
                rot0 = np.einsum("nij,njk->nik", R0[b], self.col_rel_rot)
                center0 = origin0[b] + np.einsum("nij,nj->ni", R0[b], self.col_rel_pos) + \
                    np.einsum("nij,nj->ni", rot0, self.col_center)
        d = numerics.contact_rows(self.col_kind, b, center, rot, self.col_radius, self.col_half, self.col_half_len,
                                  self.col_restitution, self.col_friction, self.dynamic.astype(np.uint8),
                                  self.excluded_index, com, v, w, st.slop, st.restitution_threshold, sweep, dt,
                                  center0, rot0, com0, self.col_hull_id, self.hulls)
        rows.extend(d)
        touching = set()
        for p, q in d["pairs"].tolist():
            pair = frozenset((entries[p].source, entries[q].source))
            touching.add(pair)
            if pair not in self.touching:
                self.new_contacts.append((entries[p].source, entries[q].source))
        self.touching = touching
        return [(entries[p].source, entries[q].source, d["point"][k], d["normal"][k], float(d["depth"][k]), int(r))
                for k, (p, q, r) in enumerate(zip(d["shape_a"].tolist(), d["shape_b"].tolist(),
                                                  d["row"].tolist()))]

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
