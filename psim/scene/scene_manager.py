"""Scene: the set of objects, their hierarchy and the current selection.

The scene is the single source of truth for object state. Views (viewport,
inspector, outliner, tables) observe it through Qt signals; all *undoable*
modifications go through psim.scene.editing.SceneEditor, which records
before/after object snapshots.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from PySide6.QtCore import QObject, Signal

from .. import numerics
from ..core.ids import IdAllocator
from ..properties.property_system import PropertyError
from .object_base import SceneObject
from .object_registry import registry


class SceneError(Exception):
    pass


class Scene(QObject):
    objectsAdded = Signal(list)       # ids
    objectsRemoved = Signal(list)     # ids
    objectsChanged = Signal(list)     # ids (any property change, including transform)
    hierarchyChanged = Signal()
    selectionChanged = Signal(list)   # ids (primary selection last)
    sceneReset = Signal()
    processesChanged = Signal()       # process assignments added / removed / edited
    globalsChanged = Signal()         # global parameters edited
    framesChanged = Signal()          # named reference frames added / removed / edited
    jointsChanged = Signal()          # joints added / removed / edited
    measurementsChanged = Signal()    # distance / angle measurements added / removed / edited

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.objects: dict[str, SceneObject] = {}
        self.ids = IdAllocator()
        self._selection: list[str] = []
        self.assignments: list = []            # ProcessAssignment, in evaluation order
        self.next_process_number = 1           # PRC-0001, ... never reused
        # project-wide named parameters readable by formulas as global.<name>:
        # {"name", "type": "float"|"vec3", "quantity": str|None, "value": SI value, "description"}
        self.global_parameters: list[dict] = []
        self.frames: list = []                 # ReferenceFrame, named user-defined frames
        self.next_frame_number = 1             # FRM-0001, ... never reused
        self.joints: list = []                 # psim.physics.joints.Joint
        self.next_joint_number = 1             # JNT-0001, ... never reused
        self.measurements: list = []           # psim.scene.measurements.Measurement
        self.next_measurement_number = 1       # MSR-0001, ... never reused
        self.simulating = False                # a simulation session is active (set by the controller)
        # values produced by a running simulation that are not object properties
        # (e.g. current total acceleration); cleared when a session ends
        self.runtime: dict[str, dict] = {}

    # ---- basic access --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.objects)

    def __contains__(self, oid: str) -> bool:
        return oid in self.objects

    def get(self, oid: str) -> SceneObject:
        try:
            return self.objects[oid]
        except KeyError:
            raise SceneError(f"Missing object reference '{oid}'") from None

    def find(self, oid: str | None) -> SceneObject | None:
        return self.objects.get(oid) if oid else None

    def clear(self) -> None:
        self.objects.clear()
        self.ids.reset()
        self._selection = []
        self.assignments = []
        self.next_process_number = 1
        self.global_parameters = []
        self.frames = []
        self.next_frame_number = 1
        self.joints = []
        self.next_joint_number = 1
        self.measurements = []
        self.next_measurement_number = 1
        self.simulating = False
        self.runtime = {}
        self.sceneReset.emit()

    # ---- process assignments ----------------------------------------------------------------
    def allocate_process_uid(self) -> str:
        used = {a.uid for a in self.assignments}
        while True:
            uid = f"PRC-{self.next_process_number:04d}"
            self.next_process_number += 1
            if uid not in used:
                return uid

    def assignment(self, uid: str):
        return next((a for a in self.assignments if a.uid == uid), None)

    def assignments_for(self, oid: str) -> list:
        obj = self.find(oid)
        return [a for a in self.assignments if obj is not None and a.targets.matches(obj)]

    def process_snapshot(self) -> list[dict]:
        return [a.to_dict() for a in self.assignments]

    def global_parameter(self, name: str) -> dict | None:
        return next((g for g in self.global_parameters if g["name"] == name), None)

    def apply_globals(self, data: list[dict]) -> None:
        import copy as _copy
        self.global_parameters = _copy.deepcopy(data)
        self.globalsChanged.emit()

    # ---- named reference frames --------------------------------------------------------------
    def allocate_frame_id(self) -> str:
        used = {f.id for f in self.frames}
        while True:
            fid = f"FRM-{self.next_frame_number:04d}"
            self.next_frame_number += 1
            if fid not in used:
                return fid

    def frame(self, fid: str | None):
        return next((f for f in self.frames if f.id == fid), None)

    def frames_snapshot(self) -> list[dict]:
        return [f.to_dict() for f in self.frames]

    def apply_frames(self, data: list[dict]) -> None:
        from ..coordinates.frames import FRAME_ID, validate_frames
        self.frames = validate_frames(data)
        for f in self.frames:
            self.next_frame_number = max(self.next_frame_number, int(FRAME_ID.match(f.id).group(1)) + 1)
        self.framesChanged.emit()

    # ---- joints -------------------------------------------------------------------------------------------
    def allocate_joint_id(self) -> str:
        used = {j.id for j in self.joints}
        while True:
            jid = f"JNT-{self.next_joint_number:04d}"
            self.next_joint_number += 1
            if jid not in used:
                return jid

    # ---- measurements -------------------------------------------------------------------------------------
    def allocate_measurement_id(self) -> str:
        used = {m.id for m in self.measurements}
        while True:
            mid = f"MSR-{self.next_measurement_number:04d}"
            self.next_measurement_number += 1
            if mid not in used:
                return mid

    def measurements_snapshot(self) -> list[dict]:
        return [m.to_dict() for m in self.measurements]

    def apply_measurements(self, data: list[dict]) -> None:
        from .measurements import MEASUREMENT_ID, validate_measurements
        self.measurements = validate_measurements(data)
        for m in self.measurements:
            self.next_measurement_number = max(self.next_measurement_number,
                                               int(MEASUREMENT_ID.match(m.id).group(1)) + 1)
        self.measurementsChanged.emit()

    def joint(self, jid: str | None):
        return next((j for j in self.joints if j.id == jid), None)

    def joints_snapshot(self) -> list[dict]:
        return [j.to_dict() for j in self.joints]

    def apply_joints(self, data: list[dict]) -> None:
        from ..physics.joints import JOINT_ID, validate_joints
        self.joints = validate_joints(data)
        for j in self.joints:
            self.next_joint_number = max(self.next_joint_number, int(JOINT_ID.match(j.id).group(1)) + 1)
        self.jointsChanged.emit()

    def frame_world_pose(self, fid: str) -> tuple[np.ndarray, np.ndarray]:
        f = self.frame(fid)
        if f is None:
            raise SceneError(f"Missing reference frame '{fid}'")
        if f.parent is None:
            return np.asarray(f.position, float), np.asarray(f.orientation, float)
        if f.parent not in self.objects:
            raise SceneError(f"Reference frame '{f.name}' is attached to the missing object '{f.parent}'")
        pp, pq = self.world_pose(f.parent)
        origin = np.asarray(pp, float) + numerics.quat_rotate(pq, f.position)
        q = numerics.quat_multiply(pq, f.orientation) if f.follow_rotation else np.asarray(f.orientation, float)
        return np.asarray(origin, float), np.asarray(numerics.quat_normalize(q), float)

    def reference_pose(self, ref: str | None) -> tuple[np.ndarray, np.ndarray]:
        """Global pose of a reference: None (global frame), an object id or a frame id."""
        if not ref:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        if ref.startswith("FRM-"):
            return self.frame_world_pose(ref)
        return self.world_pose(ref)

    def reference_name(self, ref: str | None) -> str:
        if not ref:
            return "Global"
        f = self.frame(ref)
        if f is not None:
            return f"{f.name} ({f.id})"
        o = self.find(ref)
        return f"{o.name} ({o.id})" if o else ref

    def apply_process_snapshot(self, data: list[dict]) -> None:
        from ..processes.process_base import ProcessAssignment
        self.assignments = [ProcessAssignment.from_dict(d) for d in data]
        self.processesChanged.emit()

    # ---- creation --------------------------------------------------------------------
    def create_object(self, type_id: str, *, name: str | None = None, params: dict | None = None,
                      position=None) -> SceneObject:
        """Create (but do not add) an object with a freshly allocated ID."""
        typedef = registry().get(type_id)
        obj = SceneObject.new(self.ids.allocate(), typedef, name=name or self.unique_name(typedef.name), params=params)
        from .materials import PALETTE
        if typedef.default_color is None:
            obj.color = list(PALETTE[(int(obj.id[4:]) - 1) % len(PALETTE)])
        if position is not None:
            obj.position = [float(c) for c in position]
        return obj

    def unique_name(self, base: str) -> str:
        base = re.sub(r"\s+\d+$", "", base).strip() or "Object"
        used = {o.name for o in self.objects.values()}
        n = 1
        while f"{base} {n}" in used:
            n += 1
        return f"{base} {n}"

    def add_objects(self, objs: Iterable[SceneObject]) -> None:
        objs = list(objs)
        for o in objs:
            if o.id in self.objects:
                raise SceneError(f"Duplicate object ID '{o.id}'")
            if not self.ids.is_used(o.id):
                self.ids.reserve(o.id)
            self.objects[o.id] = o
        if objs:
            self.objectsAdded.emit([o.id for o in objs])
            self.hierarchyChanged.emit()

    def remove_objects(self, ids: Iterable[str]) -> None:
        ids = [i for i in ids if i in self.objects]
        for i in ids:
            del self.objects[i]
            self.ids.release(i)
        if ids:
            sel = [s for s in self._selection if s in self.objects]
            self.objectsRemoved.emit(ids)
            self.hierarchyChanged.emit()
            if sel != self._selection:
                self.set_selection(sel)

    def notify_changed(self, ids: Iterable[str], hierarchy: bool = False) -> None:
        ids = [i for i in ids if i in self.objects]
        if ids:
            self.objectsChanged.emit(ids)
        if hierarchy:
            self.hierarchyChanged.emit()

    # ---- hierarchy ----------------------------------------------------------------------
    def children(self, oid: str | None) -> list[SceneObject]:
        return [o for o in self.objects.values() if o.parent_id == oid]

    def roots(self) -> list[SceneObject]:
        return [o for o in self.objects.values() if not o.parent_id or o.parent_id not in self.objects]

    def descendants(self, oid: str) -> list[str]:
        out, stack = [], [oid]
        while stack:
            cur = stack.pop()
            for c in self.children(cur):
                out.append(c.id)
                stack.append(c.id)
        return out

    def ancestors(self, oid: str) -> list[str]:
        out = []
        cur = self.find(oid)
        seen = set()
        while cur is not None and cur.parent_id and cur.parent_id in self.objects and cur.parent_id not in seen:
            seen.add(cur.parent_id)
            out.append(cur.parent_id)
            cur = self.objects[cur.parent_id]
        return out

    def depth(self, oid: str) -> int:
        return len(self.ancestors(oid))

    def can_reparent(self, oid: str, new_parent: str | None) -> bool:
        if new_parent is None:
            return True
        return new_parent != oid and new_parent in self.objects and new_parent not in self.descendants(oid)

    def reparent(self, oid: str, new_parent: str | None, keep_world: bool = True) -> None:
        """Change parent (not undoable by itself; use SceneEditor)."""
        if not self.can_reparent(oid, new_parent):
            raise SceneError("Invalid parent: this would create a cycle")
        obj = self.get(oid)
        p, q = self.world_pose(oid)
        obj.parent_id = new_parent
        if keep_world:
            self.set_world_pose(oid, p, q)

    def is_effectively_visible(self, oid: str) -> bool:
        obj = self.find(oid)
        if obj is None or not obj.visible:
            return False
        return all(self.objects[a].visible for a in self.ancestors(oid))

    def is_effectively_locked(self, oid: str) -> bool:
        obj = self.find(oid)
        if obj is None:
            return True
        return obj.locked or any(self.objects[a].locked for a in self.ancestors(oid))

    # ---- poses -------------------------------------------------------------------------
    def world_pose(self, oid: str) -> tuple[np.ndarray, np.ndarray]:
        """(position, quaternion) of the object's frame in global coordinates."""
        obj = self.get(oid)
        p = np.asarray(obj.position, float)
        q = np.asarray(obj.orientation, float)
        for aid in self.ancestors(oid):
            a = self.objects[aid]
            p, q = numerics.pose_compose(a.position, a.orientation, p, q)
        return np.asarray(p, float), np.asarray(q, float)

    def parent_pose(self, oid: str) -> tuple[np.ndarray, np.ndarray]:
        obj = self.get(oid)
        if obj.parent_id and obj.parent_id in self.objects:
            return self.world_pose(obj.parent_id)
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def set_world_pose(self, oid: str, position=None, orientation=None) -> None:
        obj = self.get(oid)
        pp, pq = self.parent_pose(oid)
        ip, iq = numerics.pose_inverse(pp, pq)
        wp, wq = self.world_pose(oid)
        position = wp if position is None else np.asarray(position, float)
        orientation = wq if orientation is None else np.asarray(orientation, float)
        lp, lq = numerics.pose_compose(ip, iq, position, orientation)
        obj.position = [float(c) for c in lp]
        obj.set_orientation(lq)

    def pose_in_frame(self, oid: str, frame_id: str | None) -> tuple[np.ndarray, np.ndarray]:
        """Pose of `oid` expressed in the frame of `frame_id` (None = global)."""
        wp, wq = self.world_pose(oid)
        if not frame_id:
            return wp, wq
        fp, fq = self.reference_pose(frame_id)
        ip, iq = numerics.pose_inverse(fp, fq)
        p, q = numerics.pose_compose(ip, iq, wp, wq)
        return np.asarray(p, float), np.asarray(q, float)

    def set_pose_in_frame(self, oid: str, frame_id: str | None, position=None, orientation=None) -> None:
        if not frame_id:
            self.set_world_pose(oid, position, orientation)
            return
        cp, cq = self.pose_in_frame(oid, frame_id)
        position = cp if position is None else np.asarray(position, float)
        orientation = cq if orientation is None else np.asarray(orientation, float)
        fp, fq = self.reference_pose(frame_id)
        wp, wq = numerics.pose_compose(fp, fq, position, orientation)
        self.set_world_pose(oid, wp, wq)

    def world_matrix(self, oid: str) -> np.ndarray:
        """Model matrix including the object's own geometry scale."""
        p, q = self.world_pose(oid)
        return np.asarray(numerics.trs_matrix(p, q, self.get(oid).scale), float)

    def world_bounds(self, ids: Iterable[str] | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        ids = list(self.objects) if ids is None else list(ids)
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for oid in ids:
            obj = self.find(oid)
            if obj is None:
                continue
            a, b = obj.mesh().bounds()
            corners = np.array([[x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])])
            w = numerics.transform_points(self.world_matrix(oid), corners)
            lo, hi = np.minimum(lo, w.min(axis=0)), np.maximum(hi, w.max(axis=0))
        if not np.all(np.isfinite(lo)):
            return None
        return lo, hi

    # ---- selection ------------------------------------------------------------------------
    @property
    def selection(self) -> list[str]:
        return list(self._selection)

    @property
    def primary(self) -> SceneObject | None:
        return self.objects.get(self._selection[-1]) if self._selection else None

    def selected_objects(self) -> list[SceneObject]:
        return [self.objects[i] for i in self._selection if i in self.objects]

    def set_selection(self, ids: Iterable[str]) -> None:
        new: list[str] = []
        for i in ids:
            if i in self.objects and i not in new:
                new.append(i)
        if new != self._selection:
            self._selection = new
            self.selectionChanged.emit(list(new))

    def toggle_selection(self, oid: str) -> None:
        sel = self.selection
        if oid in sel:
            sel.remove(oid)
        else:
            sel.append(oid)
        self.set_selection(sel)

    # ---- snapshots (undo support) -----------------------------------------------------
    def snapshot(self, ids: Iterable[str]) -> dict[str, dict | None]:
        return {i: (self.objects[i].to_dict() if i in self.objects else None) for i in ids}

    def apply_states(self, states: dict[str, dict | None]) -> None:
        """Make the listed objects match `states` (None = absent). Emits change signals."""
        removed = [i for i, s in states.items() if s is None and i in self.objects]
        to_create = {i: s for i, s in states.items() if s is not None and i not in self.objects}
        to_update = {i: s for i, s in states.items() if s is not None and i in self.objects}
        hierarchy = bool(removed or to_create)
        for i in removed:
            del self.objects[i]
            self.ids.release(i)
        created = []
        for i, s in to_create.items():
            obj = SceneObject.from_dict(s)
            if not self.ids.is_used(i):
                self.ids.reserve(i)
            self.objects[i] = obj
            created.append(i)
        changed = []
        for i, s in to_update.items():
            old = self.objects[i]
            new = SceneObject.from_dict(s)
            if old.parent_id != new.parent_id:
                hierarchy = True
            # keep caches when geometry did not change
            if old.type_id == new.type_id and old.params == new.params:
                new._mesh_cache = old._mesh_cache
            self.objects[i] = new
            changed.append(i)
        if removed:
            self._selection = [s for s in self._selection if s in self.objects]
            self.objectsRemoved.emit(removed)
        if created:
            self.objectsAdded.emit(created)
        if changed:
            self.objectsChanged.emit(changed)
        if hierarchy:
            self.hierarchyChanged.emit()

    # ---- validation -------------------------------------------------------------------
    def validate(self) -> list[str]:
        problems = []
        for o in self.objects.values():
            if o.parent_id and o.parent_id not in self.objects:
                problems.append(f"{o.id}: parent '{o.parent_id}' does not exist (treated as root)")
            if o.id in self.ancestors(o.id):
                problems.append(f"{o.id}: hierarchy cycle")
            try:
                o.typedef.coerce_params(o.params)
            except PropertyError as exc:
                problems.append(f"{o.id}: {exc}")
        for f in self.frames:
            if f.parent and f.parent not in self.objects:
                problems.append(f"{f.id} ({f.name}): attached to the missing object '{f.parent}'")
        for j in self.joints:
            for b in (j.body_a, j.body_b):
                if b and b not in self.objects:
                    problems.append(f"{j.id} ({j.name}): connects the missing object '{b}'")
        for m in self.measurements:
            for oid in m.objects:
                if oid not in self.objects:
                    problems.append(f"{m.id} ({m.name}): measures the missing object '{oid}'")
                    break
        return problems
