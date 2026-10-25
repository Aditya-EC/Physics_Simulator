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
from .process_base import (W_FORCE, W_ORIENTATION, W_POSITION, W_TORQUE, ProcessAssignment, ProcessDef,
                           ProcessError)
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
    level: str          # "error" | "warning"
    message: str
    assignment: str = ""


@dataclass
class ExecutionPlan:
    body_ids: list[str]
    contributions: list[PlannedProcess]
    prescribed: list[PlannedProcess]
    integrate_translation: np.ndarray      # (N,) bool
    integrate_rotation: np.ndarray         # (N,) bool
    writers: dict[tuple[str, str], list[str]]   # (object id, quantity) -> assignment uids
    diagnostics: list[Diagnostic]

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def reads_writes_table(self) -> list[tuple[str, str, str, str]]:
        """(assignment uid, process name, reads, writes) for the dependency viewer."""
        rows = []
        for p in self.contributions + self.prescribed:
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
        if wq in (W_POSITION, W_ORIENTATION) and len(uids) > 1:
            diags.append(Diagnostic("error", f"Conflicting writes: {wq} of {oid} is prescribed by "
                                             f"{', '.join(uids)} (no combination rule for prescribed values)"))
    for oid in {k[0] for k in writers}:
        for prescribed_q, contrib_q in ((W_POSITION, W_FORCE), (W_ORIENTATION, W_TORQUE)):
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
    return ExecutionPlan(body_ids, [p for p in planned if p.pdef.kind == "contribution"],
                         [p for p in planned if p.pdef.kind == "prescribed"], trans, rot, writers, diags)
