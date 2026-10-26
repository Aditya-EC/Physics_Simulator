"""Undoable scene editing.

Every edit is recorded as a StateCommand: the serialized state of each
affected object before and after the change (None meaning "absent"). This
single mechanism covers adding, deleting, property edits, transforms,
reparenting, grouping and pasting, and guarantees that undo/redo restore
object IDs exactly.
"""
from __future__ import annotations

import copy
import time
from typing import Callable, Iterable

import numpy as np
from PySide6.QtGui import QUndoCommand, QUndoStack

from ..properties.custom_properties import CustomProperty
from ..properties.property_system import PropertyError
from .object_base import SceneObject
from .scene_manager import Scene, SceneError

_MERGE_ID = 0x5053


class StateCommand(QUndoCommand):
    def __init__(self, scene: Scene, before: dict, after: dict, text: str, sel_before: list[str],
                 sel_after: list[str], merge_key: str | None = None) -> None:
        super().__init__(text)
        self.scene = scene
        self.before = before
        self.after = after
        self.sel_before = sel_before
        self.sel_after = sel_after
        self.merge_key = merge_key
        self.stamp = time.monotonic()
        self._first = True

    def id(self) -> int:  # noqa: A003 - Qt API
        return _MERGE_ID if self.merge_key else -1

    def mergeWith(self, other: QUndoCommand) -> bool:  # noqa: N802 - Qt API
        if not isinstance(other, StateCommand) or other.merge_key != self.merge_key or not self.merge_key:
            return False
        if set(other.after) != set(self.after) or other.stamp - self.stamp > 1.5:
            return False
        self.after = other.after
        self.sel_after = other.sel_after
        self.stamp = other.stamp
        return True

    def redo(self) -> None:
        if self._first:  # the change has already been applied by the editor
            self._first = False
            return
        self.scene.apply_states(_ordered(self.scene, self.after))
        self.scene.set_selection(self.sel_after)

    def undo(self) -> None:
        self.scene.apply_states(_ordered(self.scene, self.before))
        self.scene.set_selection(self.sel_before)


def _ordered(scene: Scene, states: dict) -> dict:
    """Create parents before children so hierarchy is valid at every step."""
    def depth(i):
        s = states[i]
        d, seen = 0, set()
        while s and s.get("parent") and s["parent"] not in seen:
            seen.add(s["parent"])
            d += 1
            s = states.get(s["parent"]) or (scene.objects[s["parent"]].to_dict() if s["parent"] in scene.objects else None)
        return d
    return {i: states[i] for i in sorted(states, key=depth)}


class ProcessesCommand(QUndoCommand):
    """Before/after snapshot of the process assignment list."""

    def __init__(self, scene: Scene, before: list[dict], after: list[dict], text: str) -> None:
        super().__init__(text)
        self.scene, self.before, self.after = scene, before, after
        self._first = True

    def redo(self) -> None:
        if self._first:
            self._first = False
            return
        self.scene.apply_process_snapshot(self.after)

    def undo(self) -> None:
        self.scene.apply_process_snapshot(self.before)


class GlobalsCommand(QUndoCommand):
    def __init__(self, scene: Scene, before: list[dict], after: list[dict], text: str) -> None:
        super().__init__(text)
        self.scene, self.before, self.after = scene, before, after
        self._first = True

    def redo(self) -> None:
        if self._first:
            self._first = False
            return
        self.scene.apply_globals(self.after)

    def undo(self) -> None:
        self.scene.apply_globals(self.before)


class FramesCommand(QUndoCommand):
    def __init__(self, scene: Scene, before: list[dict], after: list[dict], text: str) -> None:
        super().__init__(text)
        self.scene, self.before, self.after = scene, before, after
        self._first = True

    def redo(self) -> None:
        if self._first:
            self._first = False
            return
        self.scene.apply_frames(self.after)

    def undo(self) -> None:
        self.scene.apply_frames(self.before)


class JointsCommand(QUndoCommand):
    def __init__(self, scene: Scene, before: list[dict], after: list[dict], text: str) -> None:
        super().__init__(text)
        self.scene, self.before, self.after = scene, before, after
        self._first = True

    def redo(self) -> None:
        if self._first:
            self._first = False
            return
        self.scene.apply_joints(self.after)

    def undo(self) -> None:
        self.scene.apply_joints(self.before)


class SceneEditor:
    """High-level, undoable operations on a Scene.

    While a simulation session is active, ``recording`` is False: edits are
    applied (and validated) immediately but not pushed to the undo stack, because
    the simulation owns the dynamic state; the session itself is recorded as one
    command when it ends.
    """

    def __init__(self, scene: Scene, stack: QUndoStack | None = None) -> None:
        self.scene = scene
        self.stack = stack or QUndoStack()
        self.recording = True

    def _push(self, cmd: QUndoCommand) -> None:
        if self.recording:
            self.stack.push(cmd)

    # ---- core primitive ---------------------------------------------------------------------
    def edit(self, ids: Iterable[str], mutate: Callable[[], None], text: str, merge_key: str | None = None,
             extra_ids: Callable[[], Iterable[str]] | None = None, hierarchy: bool = False) -> bool:
        """Run `mutate` and record the before/after state of `ids` (plus `extra_ids()` afterwards).

        If `mutate` raises, every touched object is restored and the error re-raised.
        """
        ids = list(dict.fromkeys(ids))
        sel_before = self.scene.selection
        before = self.scene.snapshot(ids)
        try:
            mutate()
        except Exception:
            self.scene.apply_states(before)
            raise
        if extra_ids:
            for i in extra_ids():
                if i not in before:
                    before[i] = None
        after = self.scene.snapshot(before.keys())
        if after == before:
            return False
        self.scene.notify_changed([i for i in after if after[i] is not None and before[i] is not None], hierarchy)
        self._push(StateCommand(self.scene, before, after, text, sel_before, self.scene.selection, merge_key))
        return True

    # ---- object lifecycle ------------------------------------------------------------------------
    def add_objects(self, objs: list[SceneObject], text: str | None = None, select: bool = True) -> None:
        if not objs:
            return
        sel_before = self.scene.selection
        before = {o.id: None for o in objs}
        self.scene.add_objects(objs)
        if select:
            self.scene.set_selection([o.id for o in objs if not o.parent_id or o.parent_id not in before])
        after = self.scene.snapshot(before.keys())
        label = text or (f"Add {objs[0].name}" if len(objs) == 1 else f"Add {len(objs)} objects")
        self._push(StateCommand(self.scene, before, after, label, sel_before, self.scene.selection))

    def add_new(self, type_id: str, position=None, params: dict | None = None, name: str | None = None,
                parent_id: str | None = None) -> SceneObject:
        obj = self.scene.create_object(type_id, name=name, params=params, position=position)
        if parent_id:
            obj.parent_id = parent_id
        self.add_objects([obj])
        return obj

    def delete(self, ids: Iterable[str]) -> int:
        ids = [i for i in ids if i in self.scene]
        allids: list[str] = []
        for i in ids:
            for j in [i, *self.scene.descendants(i)]:
                if j not in allids:
                    allids.append(j)
        if not allids:
            return 0
        sel_before = self.scene.selection
        before = self.scene.snapshot(allids)
        self.scene.remove_objects(allids)
        after = {i: None for i in allids}
        n = len(allids)
        self._push(StateCommand(self.scene, before, after, f"Delete {n} object{'s' * (n != 1)}", sel_before,
                                     self.scene.selection))
        return n

    # ---- property edits ---------------------------------------------------------------------------
    def set_property(self, ids: Iterable[str], path: str, value, merge: bool = True) -> bool:
        ids = [i for i in ids if i in self.scene]
        if not ids:
            return False
        label = self.scene.get(ids[0]).property_def(path).label

        def mutate():
            for i in ids:
                self.scene.get(i).set(path, value)
        return self.edit(ids, mutate, f"Set {label}", merge_key=f"{path}|{','.join(ids)}" if merge else None,
                         hierarchy=path == "visible")

    def set_world_poses(self, poses: dict[str, tuple], text: str = "Transform", merge_key: str | None = None) -> bool:
        """poses: id -> (position or None, quaternion or None, scale or None) in global coordinates."""
        def mutate():
            for i, (p, q, s) in poses.items():
                if p is not None or q is not None:
                    self.scene.set_world_pose(i, p, q)
                if s is not None:
                    self.scene.get(i).set("scale", s)
        return self.edit(poses.keys(), mutate, text, merge_key=merge_key)

    def set_pose_in_frame(self, oid: str, frame_id: str | None, position=None, orientation=None) -> bool:
        return self.edit([oid], lambda: self.scene.set_pose_in_frame(oid, frame_id, position, orientation),
                         "Set Transform", merge_key=f"pose|{oid}|{frame_id}")

    # ---- process assignments -----------------------------------------------------------------------
    def edit_processes(self, mutate, text: str) -> bool:
        before = self.scene.process_snapshot()
        try:
            mutate()
        except Exception:
            self.scene.apply_process_snapshot(before)
            raise
        after = self.scene.process_snapshot()
        if after == before:
            return False
        self.scene.processesChanged.emit()
        self._push(ProcessesCommand(self.scene, before, after, text))
        return True

    def assign_process(self, process_id: str, targets, params: dict | None = None, label: str = ""):
        from ..processes.process_base import ProcessAssignment
        from ..processes.process_registry import process_registry
        pdef = process_registry().get(process_id)
        params = pdef.coerce_params(params or {})
        a = ProcessAssignment(self.scene.allocate_process_uid(), process_id, targets, params, True, label)
        self.edit_processes(lambda: self.scene.assignments.append(a), f"Assign {pdef.name}")
        return a

    def update_assignment(self, uid: str, **changes) -> bool:
        from ..processes.process_registry import process_registry

        def mutate():
            a = self.scene.assignment(uid)
            if a is None:
                raise SceneError(f"Unknown process assignment '{uid}'")
            if "params" in changes:
                changes["params"] = process_registry().get(a.process_id).coerce_params(changes["params"])
            for k, v in changes.items():
                setattr(a, k, v)
        return self.edit_processes(mutate, f"Edit {uid}")

    def remove_assignments(self, uids) -> bool:
        uids = set(uids)
        return self.edit_processes(
            lambda: setattr(self.scene, "assignments", [a for a in self.scene.assignments if a.uid not in uids]),
            f"Remove {len(uids)} process assignment(s)")

    def move_assignment(self, uid: str, delta: int) -> bool:
        def mutate():
            lst = self.scene.assignments
            i = next(k for k, a in enumerate(lst) if a.uid == uid)
            j = max(0, min(len(lst) - 1, i + delta))
            lst.insert(j, lst.pop(i))
        return self.edit_processes(mutate, "Reorder processes")

    # ---- global parameters ------------------------------------------------------------------------------
    # ---- named reference frames ----------------------------------------------------------------
    def set_frames(self, frames: list[dict], text: str = "Edit Reference Frames") -> bool:
        from ..coordinates.frames import validate_frames
        after = [f.to_dict() for f in validate_frames(copy.deepcopy(frames))]
        for f in after:
            if f["parent"] is not None and f["parent"] not in self.scene.objects:
                raise SceneError(f"{f['name']}: missing object reference '{f['parent']}'")
        before = self.scene.frames_snapshot()
        if after == before:
            return False
        self.scene.apply_frames(after)
        self._push(FramesCommand(self.scene, before, copy.deepcopy(after), text))
        return True

    def add_frame(self, name: str, parent: str | None = None, position=(0.0, 0.0, 0.0),
                  orientation=(1.0, 0.0, 0.0, 0.0), **options) -> str:
        fid = self.scene.allocate_frame_id()
        data = {"id": fid, "name": name, "parent": parent, "position": list(position),
                "orientation": list(orientation), **options}
        self.set_frames(self.scene.frames_snapshot() + [data], f"Add Reference Frame '{name}'")
        return fid

    def update_frame(self, fid: str, **changes) -> bool:
        frames = self.scene.frames_snapshot()
        for f in frames:
            if f["id"] == fid:
                f.update(changes)
                return self.set_frames(frames, f"Edit Reference Frame '{f['name']}'")
        raise SceneError(f"Missing reference frame '{fid}'")

    def remove_frames(self, ids) -> bool:
        ids = set(ids)
        return self.set_frames([f for f in self.scene.frames_snapshot() if f["id"] not in ids],
                               "Remove Reference Frame" + ("s" if len(ids) > 1 else ""))

    # ---- joints ---------------------------------------------------------------------------------------------
    def set_joints(self, joints: list[dict], text: str = "Edit Joints") -> bool:
        from ..physics.joints import validate_joints
        after = [j.to_dict() for j in validate_joints(copy.deepcopy(joints))]
        for j in after:
            for b in (j["body_a"], j["body_b"]):
                if b is not None and b not in self.scene.objects:
                    raise SceneError(f"{j['name']}: missing object reference '{b}'")
        before = self.scene.joints_snapshot()
        if after == before:
            return False
        self.scene.apply_joints(after)
        self._push(JointsCommand(self.scene, before, copy.deepcopy(after), text))
        return True

    def add_joint(self, jtype: str, body_a: str, body_b: str | None, anchor_world, axis_world=(0.0, 0.0, 1.0),
                  name: str | None = None, **options) -> str:
        from ..physics.joints import make_joint
        jid = self.scene.allocate_joint_id()
        if name is None:
            names = {j.name.lower() for j in self.scene.joints}
            k = 1
            while f"{jtype.capitalize()} {k}".lower() in names:
                k += 1
            name = f"{jtype.capitalize()} {k}"
        data = make_joint(self.scene, jid, name, jtype, body_a, body_b, anchor_world, axis_world, **options)
        self.set_joints(self.scene.joints_snapshot() + [data], f"Add Joint '{name}'")
        return jid

    def update_joint(self, jid: str, **changes) -> bool:
        joints = self.scene.joints_snapshot()
        for j in joints:
            if j["id"] == jid:
                j.update(changes)
                return self.set_joints(joints, f"Edit Joint '{j['name']}'")
        raise SceneError(f"Missing joint '{jid}'")

    def remove_joints(self, ids) -> bool:
        ids = set(ids)
        return self.set_joints([j for j in self.scene.joints_snapshot() if j["id"] not in ids],
                               "Remove Joint" + ("s" if len(ids) > 1 else ""))

    def set_global_parameters(self, params: list[dict], text: str = "Edit Global Parameters") -> bool:
        from ..processes.global_parameters import validate_globals
        params = validate_globals(params)
        before = copy.deepcopy(self.scene.global_parameters)
        if params == before:
            return False
        self.scene.apply_globals(params)
        self._push(GlobalsCommand(self.scene, before, copy.deepcopy(params), text))
        return True

    # ---- custom properties -------------------------------------------------------------------------
    def add_custom_property(self, ids: Iterable[str], prop_data: dict) -> bool:
        ids = list(ids)

        def mutate():
            for i in ids:
                self.scene.get(i).add_custom(CustomProperty.from_dict(prop_data))
        return self.edit(ids, mutate, f"Add Property '{prop_data.get('name')}'")

    def remove_custom_property(self, ids: Iterable[str], key: str) -> bool:
        ids = list(ids)
        return self.edit(ids, lambda: [self.scene.get(i).remove_custom(key) for i in ids], "Remove Property")

    def replace_custom_property(self, oid: str, old_key: str, prop_data: dict) -> bool:
        def mutate():
            obj = self.scene.get(oid)
            new = CustomProperty.from_dict(prop_data)
            if new.key != old_key and new.key in obj.custom:
                raise PropertyError(f"A custom property with key '{new.key}' already exists")
            items = [(k, (new if k == old_key else v)) for k, v in obj.custom.items()]
            obj.custom = {(new.key if k == old_key else k): v for k, v in items}
        return self.edit([oid], mutate, f"Edit Property '{prop_data.get('name')}'")

    # ---- hierarchy ------------------------------------------------------------------------------
    def reparent(self, ids: Iterable[str], new_parent: str | None) -> bool:
        ids = [i for i in ids if i in self.scene and i != new_parent]
        for i in ids:
            if not self.scene.can_reparent(i, new_parent):
                raise SceneError("Cannot move an object into one of its own descendants")
        return self.edit(ids, lambda: [self.scene.reparent(i, new_parent) for i in ids],
                         "Reparent" if new_parent else "Unparent", hierarchy=True)

    def group(self, ids: Iterable[str], name: str | None = None) -> SceneObject | None:
        ids = [i for i in ids if i in self.scene]
        # only group top-most selected objects (children travel with their parents)
        ids = [i for i in ids if not any(a in ids for a in self.scene.ancestors(i))]
        if not ids:
            return None
        parents = {self.scene.get(i).parent_id for i in ids}
        common_parent = parents.pop() if len(parents) == 1 else None
        lo, hi = self.scene.world_bounds(ids) or (np.zeros(3), np.zeros(3))
        center = (lo + hi) / 2
        group = self.scene.create_object("group.assembly", name=name or self.scene.unique_name("Assembly"))
        group.parent_id = common_parent

        def mutate():
            self.scene.add_objects([group])
            self.scene.set_world_pose(group.id, center, None)
            for i in ids:
                self.scene.reparent(i, group.id)
            self.scene.set_selection([group.id])
        self.edit(ids, mutate, "Group", extra_ids=lambda: [group.id], hierarchy=True)
        return group

    def ungroup(self, ids: Iterable[str]) -> list[str]:
        groups = [i for i in ids if i in self.scene and self.scene.children(i)]
        if not groups:
            return []
        freed: list[str] = []
        touched = list(groups)
        for g in groups:
            touched += [c.id for c in self.scene.children(g)]

        def mutate():
            for g in groups:
                gp = self.scene.get(g).parent_id
                for c in self.scene.children(g):
                    self.scene.reparent(c.id, gp)
                    freed.append(c.id)
                if self.scene.get(g).type_id == "group.assembly":
                    self.scene.remove_objects([g])
            self.scene.set_selection(freed)
        self.edit(touched, mutate, "Ungroup", hierarchy=True)
        return freed

    # ---- copy / paste / duplicate -------------------------------------------------------------------
    def serialize(self, ids: Iterable[str]) -> list[dict]:
        """States of the given objects and all their descendants (parents first)."""
        out, seen = [], set()
        for i in ids:
            if i in seen or i not in self.scene or any(a in ids for a in self.scene.ancestors(i)):
                continue
            for j in [i, *self.scene.descendants(i)]:
                if j not in seen:
                    seen.add(j)
                    out.append(self.scene.get(j).to_dict())
        return out

    def instantiate(self, states: list[dict], offset=None, text: str = "Paste", parent_id: str | None = None,
                    rename: bool = True) -> list[SceneObject]:
        """Create copies of serialized objects with fresh IDs, preserving internal hierarchy."""
        id_map = {s["id"]: self.scene.ids.allocate() for s in states}
        objs = []
        for s in states:
            s = dict(s)
            old_parent = s.get("parent")
            s["id"] = id_map[s["id"]]
            if old_parent in id_map:
                s["parent"] = id_map[old_parent]
            else:
                s["parent"] = parent_id if parent_id in self.scene else (old_parent if old_parent in self.scene else None)
                if offset is not None:
                    s["position"] = list(np.asarray(s.get("position", [0, 0, 0]), float) + np.asarray(offset, float))
            obj = SceneObject.from_dict(s)
            if rename:
                obj.name = self.scene.unique_name(obj.name)
            objs.append(obj)
        self.add_objects(objs, text=f"{text} {len(objs)} object{'s' * (len(objs) != 1)}")
        return objs

    def duplicate(self, ids: Iterable[str], offset=(0.5, 0.5, 0.0)) -> list[SceneObject]:
        return self.instantiate(self.serialize(list(ids)), offset=offset, text="Duplicate")

    # ---- align / distribute ---------------------------------------------------------------------------
    def align(self, ids: Iterable[str], axis: int, mode: str) -> bool:
        """Align bounding boxes along `axis` (0..2) to the min / center / max of the selection."""
        ids = [i for i in ids if i in self.scene]
        if len(ids) < 2:
            return False
        bounds = {i: self.scene.world_bounds([i]) for i in ids}
        lo = min(b[0][axis] for b in bounds.values())
        hi = max(b[1][axis] for b in bounds.values())
        target = {"min": lo, "max": hi, "center": (lo + hi) / 2}[mode]

        def mutate():
            for i, (blo, bhi) in bounds.items():
                ref = {"min": blo[axis], "max": bhi[axis], "center": (blo[axis] + bhi[axis]) / 2}[mode]
                p, _ = self.scene.world_pose(i)
                p = p.copy()
                p[axis] += target - ref
                self.scene.set_world_pose(i, p, None)
        return self.edit(ids, mutate, f"Align {'XYZ'[axis]} {mode}")

    def distribute(self, ids: Iterable[str], axis: int) -> bool:
        ids = [i for i in ids if i in self.scene]
        if len(ids) < 3:
            return False
        centers = {}
        for i in ids:
            lo, hi = self.scene.world_bounds([i])
            centers[i] = (lo[axis] + hi[axis]) / 2
        order = sorted(ids, key=lambda i: centers[i])
        first, last = centers[order[0]], centers[order[-1]]
        step = (last - first) / (len(order) - 1)

        def mutate():
            for k, i in enumerate(order):
                p, _ = self.scene.world_pose(i)
                p = p.copy()
                p[axis] += first + k * step - centers[i]
                self.scene.set_world_pose(i, p, None)
        return self.edit(ids, mutate, f"Distribute {'XYZ'[axis]}")
