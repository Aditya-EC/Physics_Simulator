"""Builds a validated execution plan from process assignments.

Diagnostics produced here (before any step runs):
* unknown process ids, invalid parameters, missing or ineligible targets;
* hierarchy problems (a simulated object inside a simulated assembly);
* conflicting writes: two prescribed writers of the same quantity on one
  object, or force/torque contributions on an object whose position or
  orientation is prescribed (there is no valid combination rule for these).

Combination rules
* Contributions to forces/torques are summed, so any number of contributors
  may target one object.
* A *global* (multi-body) contribution whose target is kinematically
  prescribed acts only on the other bodies: the prescribed body is a driven
  anchor and its reaction is ignored (reported as a warning). The same
  situation for a per-object contribution has no meaningful interpretation and
  is reported as a conflict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..properties.property_system import PropertyError
from ..scene.scene_manager import Scene
from .process_base import (W_ANGULAR_VELOCITY, W_FORCE, W_ORIENTATION, W_POSITION, W_TORQUE, W_VELOCITY,
                           ProcessAssignment, ProcessDef, ProcessError)
from .process_registry import process_registry


@dataclass
class PlannedProcess:
    assignment: ProcessAssignment
    pdef: ProcessDef
    params: dict
    target_ids: list[str]
    idx: np.ndarray = field(default_factory=lambda: np.zeros(0, int))


@dataclass
class Diagnostic:
    level: str          # "error" | "warning" | "info"
    message: str
    assignment: str = ""


@dataclass
class ExecutionPlan:
    body_ids: list[str]
    contributions: list[PlannedProcess]
    prescribed: list[PlannedProcess]
    updates: list[PlannedProcess]          # in dependency order
    integrate_translation: np.ndarray      # (N,) bool
    integrate_rotation: np.ndarray         # (N,) bool
    writers: dict[tuple[str, str], list[str]]   # (object id, quantity) -> assignment uids
    diagnostics: list[Diagnostic]
    physics: object = None                      # psim.physics.setup.PhysicsSetup (collisions / joints)
    conditions: list = field(default_factory=list)   # compiled stop conditions (psim.simulation.conditions)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def reads_writes_table(self) -> list[tuple[str, str, str, str]]:
        """(assignment uid, process name, reads, writes) for the dependency viewer."""
        rows = []
        for p in self.updates + self.contributions + self.prescribed:
            rows.append((p.assignment.uid, p.pdef.name, ", ".join(p.pdef.reads) or "—",
                         ", ".join(p.pdef.writes) or "(integration only)"))
        return rows


def build_plan(scene: Scene, assignments: list[ProcessAssignment]) -> ExecutionPlan:
    reg = process_registry()
    diags: list[Diagnostic] = []
    planned: list[PlannedProcess] = []
    for a in assignments:
        if not a.enabled:
            continue
        try:
            pdef = reg.get(a.process_id)
        except ProcessError as exc:
            diags.append(Diagnostic("error", str(exc), a.uid))
            continue
        try:
            params = pdef.coerce_params(a.params)
        except PropertyError as exc:
            diags.append(Diagnostic("error", f"{a.uid} ({pdef.name}): {exc}", a.uid))
            continue
        missing = [i for i in a.targets.ids if i not in scene]
        for i in missing:
            diags.append(Diagnostic("error", f"{a.uid} ({pdef.name}): missing object reference '{i}'", a.uid))
        targets = []
        for o in scene.objects.values():
            if not a.targets.matches(o):
                continue
            ok, why = pdef.is_eligible(o)
            if ok:
                targets.append(o.id)
            elif o.id in a.targets.ids:
                diags.append(Diagnostic("error", f"{a.uid} ({pdef.name}): {o.id} is not eligible — {why}", a.uid))
            else:
                diags.append(Diagnostic("warning", f"{a.uid} ({pdef.name}): skipped {o.id} — {why}", a.uid))
        if not targets:
            diags.append(Diagnostic("warning", f"{a.uid} ({pdef.name}): no eligible target objects", a.uid))
            continue
        if len(targets) < pdef.min_targets or (pdef.max_targets is not None and len(targets) > pdef.max_targets):
            exp = (f"exactly {pdef.min_targets}" if pdef.min_targets == pdef.max_targets
                   else f"at least {pdef.min_targets}")
            diags.append(Diagnostic("error", f"{a.uid} ({pdef.name}): needs {exp} target objects, "
                                             f"got {len(targets)}", a.uid))
            continue
        if pdef.is_formula:
            bad = _formula_checks(scene, pdef)
            if bad:
                diags += [Diagnostic("error", f"{a.uid} ({pdef.name}): {m}", a.uid) for m in bad]
                continue
        planned.append(PlannedProcess(a, pdef, params, targets))

    # ---- write sets and conflicts -------------------------------------------------------------
    writers: dict[tuple[str, str], list[str]] = {}
    global_writers: dict[tuple[str, str], list[str]] = {}
    for p in planned:
        for oid in p.target_ids:
            for wq in p.pdef.writes:
                writers.setdefault((oid, wq), []).append(p.assignment.uid)
                if p.pdef.mode == "global":
                    global_writers.setdefault((oid, wq), []).append(p.assignment.uid)
    for (oid, wq), uids in writers.items():
        if wq not in (W_FORCE, W_TORQUE) and len(uids) > 1:
            diags.append(Diagnostic("error", f"Conflicting writes: {wq} of {oid} is written by "
                                             f"{', '.join(uids)} (no combination rule for exclusive values)"))
    for oid in {k[0] for k in writers}:
        for prescribed_q, contrib_q in ((W_POSITION, W_FORCE), (W_VELOCITY, W_FORCE), (W_ORIENTATION, W_TORQUE),
                                        (W_ANGULAR_VELOCITY, W_TORQUE)):
            if (oid, prescribed_q) in writers and (oid, contrib_q) in writers:
                per_object = [u for u in writers[(oid, contrib_q)] if u not in global_writers.get((oid, contrib_q), [])]
                if per_object:
                    diags.append(Diagnostic(
                        "error", f"Conflicting writes on {oid}: {prescribed_q} is prescribed by "
                                 f"{', '.join(writers[(oid, prescribed_q)])} but {contrib_q} contributions from "
                                 f"{', '.join(per_object)} would also change it"))
                else:
                    diags.append(Diagnostic(
                        "warning", f"{oid} is driven by {', '.join(writers[(oid, prescribed_q)])}; the {contrib_q} that "
                                   f"{', '.join(writers[(oid, contrib_q)])} applies to it is ignored (driven anchor)"))

    body_ids: list[str] = []
    for p in planned:
        for oid in p.target_ids:
            if oid not in body_ids:
                body_ids.append(oid)
    for oid in body_ids:
        sim_ancestors = [a for a in scene.ancestors(oid) if a in body_ids]
        if sim_ancestors:
            diags.append(Diagnostic("error", f"{oid} is simulated but its parent {sim_ancestors[0]} is also "
                                             "simulated; simulate the assembly or its parts, not both"))
    index = {oid: i for i, oid in enumerate(body_ids)}
    for p in planned:
        p.idx = np.array([index[o] for o in p.target_ids], dtype=int)
    n = len(body_ids)
    trans = np.zeros(n, bool)
    rot = np.zeros(n, bool)
    for p in planned:
        if p.pdef.kind == "contribution":
            trans[p.idx] = True
            rot[p.idx] = True
    for p in planned:
        if p.pdef.kind == "prescribed":
            if W_POSITION in p.pdef.writes:
                trans[p.idx] = False
            if W_ORIENTATION in p.pdef.writes:
                rot[p.idx] = False
    for p in planned:
        if p.pdef.kind == "update":
            if {W_POSITION, W_VELOCITY} & set(p.pdef.writes):
                trans[p.idx] = False
            if {W_ORIENTATION, W_ANGULAR_VELOCITY} & set(p.pdef.writes):
                rot[p.idx] = False
    for oid in body_ids:
        if (oid, W_VELOCITY) in writers and (oid, W_POSITION) not in writers:
            diags.append(Diagnostic("warning", f"The velocity of {oid} is updated but no process updates its "
                                               "position, so it will not move"))
    updates, order_diags = _order_updates([p for p in planned if p.pdef.kind == "update"])
    diags += order_diags
    from ..physics.setup import build_physics_setup
    physics = build_physics_setup(scene, body_ids, trans, rot, lambda level, msg: diags.append(Diagnostic(level, msg)))
    return ExecutionPlan(body_ids, [p for p in planned if p.pdef.kind == "contribution"],
                         [p for p in planned if p.pdef.kind == "prescribed"], updates, trans, rot, writers, diags,
                         physics)


def _formula_checks(scene: Scene, pdef: ProcessDef) -> list[str]:
    """References that can only be checked against the scene: object ids and required globals."""
    from ..units.unit_registry import default_registry
    problems = [f"missing object reference '{oid}'" for oid in sorted(pdef.compiled.object_ids) if oid not in scene]
    for fid in sorted(pdef.compiled.frame_ids):
        f = scene.frame(fid)
        if f is None:
            problems.append(f"missing reference frame '{fid}' (Coordinates ▸ Reference Frames)")
        elif f.parent and f.parent not in scene:
            problems.append(f"reference frame '{f.name}' ({fid}) is attached to the missing object '{f.parent}'")
    reg = default_registry()
    for g in pdef.formula_spec.globals:
        have = scene.global_parameter(g["name"])
        if have is None:
            problems.append(f"requires the global parameter '{g['name']}' (Processes ▸ Global Parameters)")
            continue
        if have["type"] != g.get("type", "float"):
            problems.append(f"global parameter '{g['name']}' must be a {g.get('type', 'float')}")
            continue
        want = reg.quantity(g["quantity"]).dimension if g.get("quantity") else None
        got = reg.quantity(have["quantity"]).dimension if have.get("quantity") else None
        if (want or got) and want != got:
            problems.append(f"global parameter '{g['name']}' has the wrong quantity "
                            f"({have.get('quantity') or 'dimensionless'}, expected {g.get('quantity') or 'dimensionless'})")
    return problems


def _order_updates(updates: list[PlannedProcess]) -> tuple[list[PlannedProcess], list[Diagnostic]]:
    """Topologically order update processes by declared and implicit (updated(...)) dependencies."""
    diags: list[Diagnostic] = []
    n = len(updates)
    edges: dict[int, set[int]] = {i: set() for i in range(n)}
    for j, b in enumerate(updates):
        spec = b.pdef.formula_spec
        for dep in (spec.after if spec else []):
            sources = [i for i, a in enumerate(updates) if a.pdef.process_id == dep]
            if not sources:
                diags.append(Diagnostic("warning", f"{b.assignment.uid} ({b.pdef.name}) runs after '{dep}', which "
                                                   "is not active"))
            for i in sources:
                if i != j:
                    edges[i].add(j)
        for q in (b.pdef.compiled.updated_refs if b.pdef.compiled else ()):
            if q in b.pdef.writes:
                diags.append(Diagnostic("error", f"{b.assignment.uid} ({b.pdef.name}) reads updated({q}), which it "
                                                 "writes itself"))
                continue
            writers = [i for i, a in enumerate(updates) if q in a.pdef.writes and set(a.target_ids) & set(b.target_ids)]
            if not writers:
                diags.append(Diagnostic("error", f"{b.assignment.uid} ({b.pdef.name}) reads updated({q}) but no "
                                                 "active update process writes it for the same objects"))
            for i in writers:
                edges[i].add(j)
    indeg = {i: 0 for i in range(n)}
    for i in edges:
        for j in edges[i]:
            indeg[j] += 1
    ready = [i for i in range(n) if indeg[i] == 0]
    order: list[int] = []
    while ready:
        i = min(ready)  # stable: keep assignment order where unconstrained
        ready.remove(i)
        order.append(i)
        for j in sorted(edges[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    if len(order) < n:
        cyc = [updates[i].assignment.uid for i in range(n) if i not in order]
        diags.append(Diagnostic("error", "Process dependency cycle between " + ", ".join(cyc)))
        order += [i for i in range(n) if i not in order]
    return [updates[i] for i in order], diags
