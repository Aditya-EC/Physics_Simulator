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

import numpy as np

from .. import numerics
from .contact_kernel import COLLIDER_CODES
from .hull import box_hull, pack_hulls
from .joint_kernel import pack_joints
from .setup import PhysicsSetup

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


_ROW_KEYS = ("ia", "ib", "J", "B", "rhs", "lo", "hi", "parent", "mu", "err", "mode", "block")


def _rows(contacts: dict | None, joints: dict | None) -> dict:
    """All constraint rows of a step: contact rows first (their friction rows refer to parents by index), then
    joint rows. mode: 0 equality, 1 inequality with position correction, 2 no position correction."""
    parts = []
    if contacts is not None and len(contacts["ia"]):
        m = len(contacts["ia"])
        parts.append(dict(contacts, B=contacts["J"], block=np.ones(m, np.int64)))
    if joints is not None and len(joints["ia"]):
        m = len(joints["ia"])
        parts.append(dict(joints, parent=np.full(m, -1, np.int64), mu=np.zeros(m)))
    if len(parts) == 1:
        return {k: parts[0][k] for k in _ROW_KEYS}
    if not parts:
        return {"ia": np.zeros(0, np.int64)}
    return {k: np.concatenate([p[k] for p in parts]) for k in _ROW_KEYS}


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
        self.joint_entries = list(setup.joints)
        self.broken: list[str] = []                     # ids of joints that broke during the run, in order
        self.new_breaks: list[tuple] = []               # (joint id, name, force, torque) for the event log
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
        static = np.array([c.static_friction for c in cols], float)
        # static coefficients only matter where they exceed the kinetic ones
        self.col_static = static if np.any(static > self.col_friction) else None
        self.col_bound = np.array([0.0 if c.shape.kind == "plane" else c.shape.bounding_radius() for c in cols],
                                  float)
        self._set_joints()
        # touching object pairs as integer keys a·S + b (a < b indices into self.sources)
        self.sources = list(dict.fromkeys(c.source for c in cols))
        src = {k: i for i, k in enumerate(self.sources)}
        self.col_source = np.array([src[c.source] for c in cols], np.int64)
        self._touching_keys = np.zeros(0, np.int64)
        self.new_contacts: list[tuple[str, str]] = []   # newly touching collider pairs (for the event log)
        self._last = None                               # contact arrays and impulses of the last step
        self._last_dv, self._last_dw = np.zeros((n, 3)), np.zeros((n, 3))   # constraint Δv, Δω of the last step

    def _set_joints(self) -> None:
        """Arrays of the joints that act (not broken), and the body pairs they keep from colliding."""
        self.active_joints = [je for je in self.joint_entries if je.joint.id not in self.broken]
        self.joint_arrays = pack_joints(self.active_joints, self.index)
        self._springs = [je for je in self.active_joints if je.joint.has_spring]
        self._breakable = [(k, je) for k, je in enumerate(self.active_joints)
                           if je.joint.break_force > 0 or je.joint.break_torque > 0]
        connected = {frozenset((je.body_a, je.body_b or "@world")) for je in self.active_joints
                     if not je.joint.collide_connected}
        self.excluded_index = np.array([sorted(self.index[k] for k in pair) for pair in connected
                                        if all(k in self.index for k in pair) and len(pair) == 2],
                                       np.int64).reshape(-1, 2)

    def restore_broken(self, ids) -> None:
        """Joints that had broken at a saved state (see SimulationEngine.import_state)."""
        known = {je.joint.id for je in self.joint_entries}
        self.broken = [i for i in ids if i in known]
        self._set_joints()

    def take_breaks(self) -> list[tuple]:
        out, self.new_breaks = self.new_breaks, []
        return out

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

    def _midpoint(self, s0, s1, dt: float, end):
        """(origin, com, R) halfway between the start of the step and its predicted end: the integrated state
        `s1` moved on by the previous step's constraint velocity change (which varies slowly), since the joint
        impulse of this step will move it too. `end` = (origin, com, R) of `s1`. Static bodies do not move."""
        _, c0, *_ = self._gather(s0)
        origin, c1, R = (x.copy() for x in end)
        com = 0.5 * (c0 + c1 + self._last_dv * dt)
        sim, e = self._sim_rows, self.engine_rows[self._sim_rows]
        if len(sim):
            q0 = np.asarray(s0.q[e], float)
            q1 = np.array(s1.q[e], float)
            spin = np.any(self._last_dw[sim] != 0.0, axis=1)
            if spin.any():
                q = np.ascontiguousarray(q1[spin])
                numerics.quat_integrate(q, np.ascontiguousarray(self._last_dw[sim][spin]), dt)
                q1[spin] = q
            q1[np.einsum("ij,ij->i", q0, q1) < 0] *= -1.0
            R[sim] = _quat_matrices(q0 + q1)
            origin[sim] = com[sim] - np.einsum("nij,nj->ni", R[sim], self.com_local[sim])
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
        v_start, w_start = v.copy(), w.copy()
        if self._springs and dt > 0:                 # spring impulses first: contacts and joints see their effect
            self._apply_springs(origin, com, R, v, w, inv_i, dt)
        crows = self._contacts(origin, com, R, v, w, dt, start)
        jrows = None
        if self.active_joints:
            ja = self.joint_arrays
            mid = self._midpoint(start, s, dt, (origin, com, R)) if start is not None else (None, None, None)
            jrows = numerics.joint_rows(ja["kind"], ja["body_a"], ja["body_b"], ja["anchor_a"], ja["anchor_b"],
                                        ja["axis_a"], ja["axis_b"], ja["ref_rot"], ja["flags"], ja["lower"],
                                        ja["upper"], ja["motor_speed"], ja["motor_max"], ja["length"], origin, com,
                                        R, dt, st.slop, *mid)
        rows = _rows(crows, jrows)
        if not len(rows["ia"]):
            self._last = None
            zero = np.zeros_like(v)
            self._last_dv, self._last_dw = v - v_start, w - w_start
            return self._apply(s, self._last_dv, self._last_dw, zero, zero, dt)
        ia, ib, J, rhs, lo, hi = (rows[k] for k in ("ia", "ib", "J", "rhs", "lo", "hi"))
        block = rows["block"]
        v1, w1 = v.copy(), w.copy()
        # joints pass the midpoint Jacobian as impulse directions (contact rows use their own J)
        B = rows["B"] if jrows is not None and start is not None else None
        lam = numerics.pgs_solve(ia, ib, J, rhs, lo, hi, rows["parent"], rows["mu"], block, self.inv_mass, inv_i,
                                 v1, w1, st.iterations, B)
        dv, dw = v1 - v_start, w1 - w_start         # constraint and spring velocity change of the step
        self._last_dv, self._last_dw = dv, dw
        # position pass on pseudo-velocities: remaining error after the Δv·Δt displacement
        jd = self._row_velocities(ia, ib, J, dv, dw)
        err = rows["err"] + jd * dt
        mode = rows["mode"]
        eq, ineq = mode == 0, mode == 1
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
        self._last = (crows, lam) if crows is not None else None
        if self._breakable and jrows is not None and dt > 0:
            first = len(rows["ia"]) - len(jrows["ia"])          # joint rows follow the contact rows
            self._check_breaks(jrows, lam[first:], dt)
        return self._apply(s, dv, dw, vp, wp, dt)

    def _apply(self, s, dv, dw, vp, wp, dt):
        """Add the velocity changes to the dynamic bodies of `s` and move them by (Δv + position pass)·Δt."""
        n_eng = len(s.v)
        dv_eng, dw_eng = np.zeros((n_eng, 3)), np.zeros((n_eng, 3))
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
        return dv_eng, dw_eng

    # ---- joint springs and breaking ----------------------------------------------------------------------------------
    def _spring_row(self, je, origin, com, R):
        """(Jacobian (12,), coordinate, rest value) of a joint spring, or None when it does not act (a slack
        rope)."""
        from .joint_kernel import perp
        j = je.joint
        a = self.index[je.body_a]
        b = self.index[je.body_b] if je.body_b else -1
        Ra = R[a]
        Rb = R[b] if b >= 0 else E3
        pa = origin[a] + Ra @ je.anchor_a
        pb = origin[b] + Rb @ je.anchor_b if b >= 0 else np.asarray(je.anchor_b, float)
        d, ra = pb - pa, pa - com[a]
        rb = pb - com[b] if b >= 0 else np.zeros(3)
        aw = Ra @ je.axis_a
        zero = np.zeros(3)
        if j.type == "revolute":                     # angle of A relative to B about the hinge (as joint_kernel)
            ua_l = perp(np.asarray(je.axis_a, float))[0]
            ua, ub = Ra @ ua_l, Rb @ (je.ref_rot.T @ ua_l)
            x = -math.atan2(float(np.cross(ua, ub) @ aw), float(ua @ ub))
            x = j.spring_rest + math.remainder(x - j.spring_rest, 2 * math.pi)   # nearest turn
            return np.concatenate([zero, aw, zero, -aw]), x, j.spring_rest
        if j.type == "prismatic":                    # travel of A's anchor relative to B's along the axis
            n = -aw
            return np.concatenate([-n, -np.cross(ra + d, n), n, np.cross(rb, n)]), -float(d @ aw), j.spring_rest
        length = float(np.linalg.norm(d))            # distance joint: the anchor distance, rest = its length
        if j.rope and length <= j.length:
            return None                              # a slack rope does not pull
        n = d / length if length > 1e-9 else aw
        return np.concatenate([-n, -np.cross(ra, n), n, np.cross(rb, n)]), length, j.length

    def _apply_springs(self, origin, com, R, v, w, inv_i, dt) -> None:
        """Implicit (backward Euler) spring-damper impulses along each spring's joint coordinate. The poses are
        already advanced by Δt·ẋ, so with the start-of-step coordinate x − Δt·ẋ the backward Euler impulse is
        λ = −Δt (k (x − x₀) + c ẋ) / (1 + Δt (k Δt + c) m⁻¹): stable for any stiffness."""
        for je in self._springs:
            row = self._spring_row(je, origin, com, R)
            if row is None:
                continue
            Jr, x, rest = row
            a = self.index[je.body_a]
            b = self.index[je.body_b] if je.body_b else -1
            rate = float(Jr[:3] @ v[a] + Jr[3:6] @ w[a])
            minv = float(self.inv_mass[a] * (Jr[:3] @ Jr[:3]) + Jr[3:6] @ inv_i[a] @ Jr[3:6])
            if b >= 0:
                rate += float(Jr[6:9] @ v[b] + Jr[9:] @ w[b])
                minv += float(self.inv_mass[b] * (Jr[6:9] @ Jr[6:9]) + Jr[9:] @ inv_i[b] @ Jr[9:])
            if minv <= 0:
                continue
            k, c = je.joint.spring_stiffness, je.joint.spring_damping
            lam = -dt * (k * (x - rest) + c * rate) / (1 + dt * (k * dt + c) * minv)
            v[a] += self.inv_mass[a] * Jr[:3] * lam
            w[a] += inv_i[a] @ Jr[3:6] * lam
            if b >= 0:
                v[b] += self.inv_mass[b] * Jr[6:9] * lam
                w[b] += inv_i[b] @ Jr[9:] * lam

    def _check_breaks(self, jrows: dict, lam: np.ndarray, dt: float) -> None:
        """Break the joints whose transmitted force or torque (row impulses / Δt) exceeded their limit."""
        linear = np.any(jrows["J"][:, :3] != 0.0, axis=1) | np.any(jrows["J"][:, 6:9] != 0.0, axis=1)
        broke = []
        for k, je in self._breakable:
            sel = jrows["joint"] == k
            force = float(np.linalg.norm(lam[sel & linear])) / dt
            torque = float(np.linalg.norm(lam[sel & ~linear])) / dt
            j = je.joint
            if (j.break_force > 0 and force > j.break_force) or (j.break_torque > 0 and torque > j.break_torque):
                broke.append(j.id)
                self.new_breaks.append((j.id, j.name, force, torque))
        if broke:
            self.broken += broke
            self._set_joints()

    @staticmethod
    def _row_velocities(ia, ib, J, v, w) -> np.ndarray:
        """J·[v_a, w_a, v_b, w_b] for every row (index −1 = the fixed world)."""
        out = np.zeros(len(ia))
        a, b = ia >= 0, ib >= 0
        out[a] += np.einsum("ij,ij->i", J[a, :3], v[ia[a]]) + np.einsum("ij,ij->i", J[a, 3:6], w[ia[a]])
        out[b] += np.einsum("ij,ij->i", J[b, 6:9], v[ib[b]]) + np.einsum("ij,ij->i", J[b, 9:], w[ib[b]])
        return out

    # ---- contacts ------------------------------------------------------------------------------------------------
    @property
    def last_contacts(self) -> list:
        """Contacts of the last step: (source a, source b, point, normal, depth, normal impulse)."""
        if self._last is None:
            return []
        d, lam = self._last
        names = self.sources
        sa, sb = self.col_source[d["shape_a"]], self.col_source[d["shape_b"]]
        return [(names[a], names[b], d["point"][k], d["normal"][k], float(d["depth"][k]), float(lam[r]))
                for k, (a, b, r) in enumerate(zip(sa.tolist(), sb.tolist(), d["row"].tolist()))]

    @property
    def touching(self) -> set[frozenset]:
        """Object pairs whose colliders touched in the last step."""
        n = max(len(self.sources), 1)
        return {frozenset((self.sources[k // n], self.sources[k % n])) for k in self._touching_keys.tolist()}

    @touching.setter
    def touching(self, pairs) -> None:
        n = max(len(self.sources), 1)
        index = {k: i for i, k in enumerate(self.sources)}
        keys = []
        for pair in pairs:
            ids = sorted(index[k] for k in pair if k in index)
            if len(ids) == 2:
                keys.append(ids[0] * n + ids[1])
        self._touching_keys = np.unique(np.array(keys, np.int64))

    def _contacts(self, origin, com, R, v, w, dt, start=None) -> dict | None:
        """Contacts of all colliders and their rows (numerics.contact_rows: C++ or the NumPy kernel)."""
        st = self.settings
        entries = self.setup.colliders
        if not entries:
            self._touching_keys = np.zeros(0, np.int64)
            return None
        b = self.col_body
        rot = np.einsum("nij,njk->nik", R[b], self.col_rel_rot)
        center = origin[b] + np.einsum("nij,nj->ni", R[b], self.col_rel_pos) + np.einsum("nij,nj->ni", rot,
                                                                                         self.col_center)
        sweep = center0 = rot0 = com0 = v0 = w0 = None
        if start is not None and self.col_static is not None:   # stick or slip is judged at the start of the step
            *_, v0, w0, _ = self._gather(start)
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
                                  center0, rot0, com0, self.col_hull_id, self.hulls, self.col_static, v0, w0)
        # touching object pairs; pairs that were not touching in the previous step are logged, in pair order
        n = max(len(self.sources), 1)
        sa, sb = self.col_source[d["pairs"][:, 0]], self.col_source[d["pairs"][:, 1]]
        keys = np.minimum(sa, sb) * n + np.maximum(sa, sb)
        new = ~np.isin(keys, self._touching_keys)
        if new.any():
            _, first = np.unique(keys[new], return_index=True)
            for k in np.sort(first):
                p, q = d["pairs"][new][k]
                self.new_contacts.append((entries[p].source, entries[q].source))
        self._touching_keys = np.unique(keys)
        return d

    def take_new_contacts(self) -> list[tuple[str, str]]:
        out, self.new_contacts = self.new_contacts, []
        return out
