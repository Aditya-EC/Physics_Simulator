"""Physics ▸ Joints: create and edit joints between bodies (applied as one undoable step).

Anchors and axes are entered in world coordinates at the objects' current poses and stored in the
bodies' local frames, so moving the objects afterwards keeps the joint attached to the same points.
"""
from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from ... import numerics
from ...physics.joints import JOINT_TYPES, JointError, make_joint, validate_joints
from ...scene.editing import SceneEditor
from ...scene.scene_manager import Scene, SceneError
from ...units.unit_registry import UnitRegistry
from ..widgets.inputs import NumberField, VectorField

TYPE_HELP = {
    "fixed": "Locks all six degrees of freedom: the two bodies move as one.",
    "ball": "The anchor points coincide; free rotation in every direction.",
    "revolute": "Hinge: rotation about the axis only. Angle limits and motor act on A relative to B.",
    "prismatic": "Slider: translation along the axis only, no relative rotation. Limits and motor act on the "
                 "displacement of A's anchor relative to B.",
    "distance": "Keeps the anchors at a fixed distance (rod), or at most that distance (rope).",
}


class JointsDialog(QDialog):
    def __init__(self, scene: Scene, editor: SceneEditor, units: UnitRegistry, select: str | None = None,
                 body: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Joints")
        self.resize(900, 600)
        self.scene, self.editor, self.units = scene, editor, units
        self.joints: list[dict] = scene.joints_snapshot()
        self._body = body          # preselected body for new joints
        self._next_id = None
        self._loading = False
        lay = QVBoxLayout(self)
        note = QLabel("Joints connect two bodies, or a body and the fixed world. Anchor and axis are given in world "
                      "coordinates at the current object poses. Only bodies moved by force processes (dynamic) are "
                      "moved by joints; bodies with prescribed motion drive the joint, and objects without "
                      "processes are fixed.")
        note.setWordWrap(True)
        note.setObjectName("Muted")
        lay.addWidget(note)
        body_lay = QHBoxLayout()
        lay.addLayout(body_lay, 1)

        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show)
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, slot in (("Add", self._add), ("Remove", self._remove)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(slot)
            row.addWidget(b)
        self.add_button = row.itemAt(0).widget()
        self.add_button.setEnabled(any(o.typedef.physical for o in scene.objects.values()))
        self.add_button.setToolTip("Add a joint on the selected object (or the first body)")
        left.addLayout(row)
        body_lay.addLayout(left, 2)

        lu, au = units.preferred_unit("length"), units.preferred_unit("angle")
        self.form_widget = QWidget()
        form = QFormLayout(self.form_widget)
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self._store("name", self.name.text().strip()))
        form.addRow("Name", self.name)
        self.jid = QLabel()
        form.addRow("ID", self.jid)
        self.type = QComboBox()
        self.type.addItems(list(JOINT_TYPES))
        self.type.activated.connect(lambda _i: self._store("type", self.type.currentText()))
        form.addRow("Type", self.type)
        self.type_help = QLabel()
        self.type_help.setWordWrap(True)
        self.type_help.setObjectName("Muted")
        form.addRow(self.type_help)   # spans the form: room for the wrapped help text
        self.body_a = QComboBox()
        self.body_a.activated.connect(lambda _i: self._rebind())
        form.addRow("Body A", self.body_a)
        self.body_b = QComboBox()
        self.body_b.activated.connect(lambda _i: self._rebind())
        form.addRow("Body B", self.body_b)
        self.anchor = VectorField(3, units, lu, 4, -1e7, 1e7)
        self.anchor.committed.connect(lambda v: self._rebind(anchor=v))
        form.addRow(f"Anchor (world) [{lu}]", self.anchor)
        self.anchor_b = VectorField(3, units, lu, 4, -1e7, 1e7)
        self.anchor_b.committed.connect(lambda v: self._rebind(anchor_b=v))
        self.anchor_b_label = QLabel(f"Anchor on B (world) [{lu}]")
        form.addRow(self.anchor_b_label, self.anchor_b)
        self.axis = VectorField(3, None, None, 4, -1e6, 1e6)
        self.axis.committed.connect(lambda v: self._rebind(axis=v))
        self.axis_label = QLabel("Axis (world)")
        form.addRow(self.axis_label, self.axis)
        self.length = NumberField(units, lu, 4, 0.0, 1e7)
        self.length.committed.connect(lambda v: self._store("length", float(v)))
        self.length_label = QLabel(f"Length [{lu}]")
        form.addRow(self.length_label, self.length)
        self.rope = QCheckBox("Rope (only the maximum length is enforced)")
        self.rope.toggled.connect(lambda v: self._store("rope", bool(v)))
        form.addRow("", self.rope)
        self.limit = QCheckBox("Limits")
        self.limit.toggled.connect(lambda v: self._store("limit_enabled", bool(v)))
        form.addRow("", self.limit)
        self.lower = NumberField(units, au, 4, -1e7, 1e7)
        self.lower.committed.connect(lambda v: self._store("lower", float(v)))
        self.lower_label = QLabel()
        form.addRow(self.lower_label, self.lower)
        self.upper = NumberField(units, au, 4, -1e7, 1e7)
        self.upper.committed.connect(lambda v: self._store("upper", float(v)))
        self.upper_label = QLabel()
        form.addRow(self.upper_label, self.upper)
        self.motor = QCheckBox("Motor")
        self.motor.toggled.connect(lambda v: self._store("motor_enabled", bool(v)))
        form.addRow("", self.motor)
        self.speed = NumberField(units, None, 4, -1e7, 1e7)
        self.speed.committed.connect(lambda v: self._store("motor_speed", float(v)))
        self.speed_label = QLabel()
        form.addRow(self.speed_label, self.speed)
        self.fmax = NumberField(units, None, 4, 0.0, 1e12)
        self.fmax.committed.connect(lambda v: self._store("motor_max", float(v)))
        self.fmax_label = QLabel()
        form.addRow(self.fmax_label, self.fmax)
        # spring on the joint coordinate (SI units: there are no display units for stiffness and damping)
        self.stiffness = NumberField(None, None, 4, 0.0, 1e15)
        self.stiffness.committed.connect(lambda v: self._store("spring_stiffness", float(v)))
        self.stiffness_label = QLabel()
        form.addRow(self.stiffness_label, self.stiffness)
        self.damping = NumberField(None, None, 4, 0.0, 1e15)
        self.damping.committed.connect(lambda v: self._store("spring_damping", float(v)))
        self.damping_label = QLabel()
        form.addRow(self.damping_label, self.damping)
        self.rest = NumberField(units, None, 4, -1e7, 1e7)
        self.rest.committed.connect(lambda v: self._store("spring_rest", float(v)))
        self.rest_label = QLabel()
        form.addRow(self.rest_label, self.rest)
        fu, tu = units.preferred_unit("force"), units.preferred_unit("torque")
        self.break_force = NumberField(units, fu, 4, 0.0, 1e15)
        self.break_force.setToolTip("The joint breaks when the force it transmits exceeds this (0 = never)")
        self.break_force.committed.connect(lambda v: self._store("break_force", float(v)))
        form.addRow(f"Break force [{fu}]", self.break_force)
        self.break_torque = NumberField(units, tu, 4, 0.0, 1e15)
        self.break_torque.setToolTip("The joint breaks when the torque it transmits exceeds this (0 = never)")
        self.break_torque.committed.connect(lambda v: self._store("break_torque", float(v)))
        form.addRow(f"Break torque [{tu}]", self.break_torque)
        self.collide = QCheckBox("Connected bodies collide with each other")
        self.collide.toggled.connect(lambda v: self._store("collide_connected", bool(v)))
        form.addRow("", self.collide)
        self.enabled = QCheckBox("Enabled")
        self.enabled.toggled.connect(lambda v: self._store("enabled", bool(v)))
        form.addRow("", self.enabled)
        self.description = QLineEdit()
        self.description.editingFinished.connect(lambda: self._store("description", self.description.text()))
        form.addRow("Description", self.description)
        body_lay.addWidget(self.form_widget, 3)

        self.error = QLabel()
        self.error.setStyleSheet("color: #e5534b;")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Apply
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        lay.addWidget(buttons)
        if select is None and body is not None:
            select = next((j["id"] for j in self.joints if body in (j["body_a"], j["body_b"])), None)
        self._refresh_list(select)

    # ---- list ------------------------------------------------------------------------------------------
    def _refresh_list(self, select: str | None = None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for j in self.joints:
            item = QListWidgetItem(self._item_text(j))
            item.setData(Qt.ItemDataRole.UserRole, j["id"])
            self.list.addItem(item)
        self.list.blockSignals(False)
        ids = [j["id"] for j in self.joints]
        row = ids.index(select) if select in ids else (0 if ids else -1)
        self.list.setCurrentRow(row)
        self._show(row)

    def _item_text(self, j: dict) -> str:
        return f"{j['name']}  ({j['id']}, {j['type']})" + ("" if j.get("enabled", True) else " — disabled")

    def _current(self) -> dict | None:
        row = self.list.currentRow()
        return self.joints[row] if 0 <= row < len(self.joints) else None

    def _allocate_id(self) -> str:
        used = {j["id"] for j in self.joints} | {j.id for j in self.scene.joints}
        n = self.scene.next_joint_number
        while f"JNT-{n:04d}" in used:
            n += 1
        return f"JNT-{n:04d}"

    def _add(self) -> None:
        bodies = [o for o in self.scene.objects.values() if o.typedef.physical]
        if not bodies:
            return
        a = self._body if self._body in self.scene.objects else None
        if a is None:
            p = self.scene.primary
            a = p.id if p is not None and p.typedef.physical else bodies[0].id
        names = {j["name"].lower() for j in self.joints}
        k = 1
        while f"Revolute {k}".lower() in names:
            k += 1
        anchor = self.scene.world_pose(a)[0]
        try:
            data = make_joint(self.scene, self._allocate_id(), f"Revolute {k}", "revolute", a, None, anchor)
        except (JointError, SceneError) as exc:
            self.error.setText(str(exc))
            return
        self.joints.append(data)
        self._refresh_list(data["id"])

    def _remove(self) -> None:
        cur = self._current()
        if cur is not None:
            self.joints.remove(cur)
            self._refresh_list()

    # ---- world-space views of the stored local data ------------------------------------------------------
    def _to_world(self, oid: str | None, point, direction: bool = False) -> np.ndarray:
        if oid is None or oid not in self.scene.objects:
            return np.asarray(point, float)
        p, q = self.scene.world_pose(oid)
        v = np.asarray(numerics.quat_rotate(q, np.asarray(point, float)), float)
        return v if direction else np.asarray(p, float) + v

    # ---- editor --------------------------------------------------------------------------------------------
    def _fill_bodies(self, j: dict) -> None:
        self.body_a.clear()
        self.body_b.clear()
        self.body_b.addItem("— World (fixed) —", None)
        for o in self.scene.objects.values():
            if o.typedef.physical or o.type_id.startswith("group."):
                self.body_a.addItem(f"{o.name} ({o.id})", o.id)
                self.body_b.addItem(f"{o.name} ({o.id})", o.id)
        for combo, ref in ((self.body_a, j["body_a"]), (self.body_b, j["body_b"])):
            if ref and combo.findData(ref) < 0:
                combo.addItem(f"⚠ missing object {ref}", ref)
            combo.setCurrentIndex(max(combo.findData(ref), 0))

    def _show(self, _row: int) -> None:
        j = self._current()
        self.form_widget.setEnabled(j is not None)
        if j is None:
            return
        self._loading = True
        try:
            self.name.setText(j["name"])
            self.jid.setText(j["id"])
            self.type.setCurrentText(j["type"])
            self._fill_bodies(j)
            self.anchor.set_si(self._to_world(j["body_a"], j["anchor_a"]))
            self.anchor_b.set_si(self._to_world(j["body_b"], j["anchor_b"]))
            self.axis.set_si(self._to_world(j["body_a"], j["axis_a"], direction=True))
            self.length.set_si(float(j.get("length", 1.0)))
            self.rope.setChecked(bool(j.get("rope", False)))
            self.limit.setChecked(bool(j.get("limit_enabled", False)))
            self.motor.setChecked(bool(j.get("motor_enabled", False)))
            self.collide.setChecked(bool(j.get("collide_connected", False)))
            self.enabled.setChecked(bool(j.get("enabled", True)))
            self.description.setText(j.get("description", ""))
            self._update_kind(j)
            self.lower.set_si(float(j.get("lower", 0.0)))
            self.upper.set_si(float(j.get("upper", 0.0)))
            self.speed.set_si(float(j.get("motor_speed", 0.0)))
            self.fmax.set_si(float(j.get("motor_max", 0.0)))
            self.stiffness.set_si(float(j.get("spring_stiffness", 0.0)))
            self.damping.set_si(float(j.get("spring_damping", 0.0)))
            self.rest.set_si(float(j.get("spring_rest", 0.0)))
            self.break_force.set_si(float(j.get("break_force", 0.0)))
            self.break_torque.set_si(float(j.get("break_torque", 0.0)))
        finally:
            self._loading = False

    def _update_kind(self, j: dict) -> None:
        t = j["type"]
        self.type_help.setText(TYPE_HELP[t])
        linear = t == "prismatic"
        q = "length" if linear else "angle"
        u = self.units.preferred_unit(q)
        su = self.units.preferred_unit("velocity" if linear else "angular_velocity")
        fu = self.units.preferred_unit("force" if linear else "torque")
        for f in (self.lower, self.upper):
            f.set_unit(u)
        self.speed.set_unit(su)
        self.fmax.set_unit(fu)
        what = "travel" if linear else "angle"
        self.lower_label.setText(f"Lower {what} [{u}]")
        self.upper_label.setText(f"Upper {what} [{u}]")
        self.speed_label.setText(f"Motor speed [{su}]")
        self.fmax_label.setText(f"Maximum motor {'force' if linear else 'torque'} [{fu}]")
        limited = t in ("revolute", "prismatic")
        for w in (self.limit, self.lower, self.upper, self.lower_label, self.upper_label, self.motor, self.speed,
                  self.fmax, self.speed_label, self.fmax_label):
            w.setVisible(limited)
        distance = t == "distance"
        for w in (self.anchor_b, self.anchor_b_label, self.length, self.length_label, self.rope):
            w.setVisible(distance)
        spring = t in ("revolute", "prismatic", "distance")
        lin = t != "revolute"
        self.stiffness_label.setText("Spring stiffness [" + ("N/m" if lin else "N·m/rad") + "]")
        self.damping_label.setText("Spring damping [" + ("N·s/m" if lin else "N·m·s/rad") + "]")
        self.rest.set_unit(u)
        self.rest_label.setText(f"Spring rest {what} [{u}]")
        for w in (self.stiffness, self.stiffness_label, self.damping, self.damping_label):
            w.setVisible(spring)
        for w in (self.rest, self.rest_label):     # a distance spring rests at the joint's length
            w.setVisible(limited)
        axis = t in ("revolute", "prismatic")
        self.axis.setVisible(axis)
        self.axis_label.setVisible(axis)

    def _store(self, key: str, value) -> None:
        j = self._current()
        if j is None or self._loading:
            return
        j[key] = value
        if key == "type" and value not in ("revolute", "prismatic", "distance"):
            j["spring_stiffness"] = j["spring_damping"] = 0.0     # this type has no spring (its fields are hidden)
        self.list.currentItem().setText(self._item_text(j))
        if key == "type":
            self._show(self.list.currentRow())
        self.error.clear()

    def _rebind(self, anchor=None, axis=None, anchor_b=None) -> None:
        """Recompute the local anchors and axes from the world-space fields at the current poses."""
        j = self._current()
        if j is None or self._loading:
            return
        a, b = self.body_a.currentData(), self.body_b.currentData()
        if a == b:
            self.error.setText("Body A and body B must be different")
            self._show(self.list.currentRow())
            return
        anchor = self.anchor.si() if anchor is None else anchor
        axis = self.axis.si() if axis is None else axis
        remeasure = anchor_b is not None
        anchor_b = self.anchor_b.si() if anchor_b is None else anchor_b
        options = {k: v for k, v in j.items() if k not in ("id", "name", "type", "body_a", "body_b", "anchor_a",
                                                            "anchor_b", "axis_a", "axis_b", "ref_orientation")}
        try:
            if j["type"] == "distance":
                # editing anchor B re-measures the length; editing body/anchor A keeps the rod length
                keep = None if remeasure else options.get("length")
                options.pop("length", None)
                data = make_joint(self.scene, j["id"], j["name"], j["type"], a, b, anchor, axis,
                                  anchor_world_b=anchor_b, **options)
                if keep is not None:
                    data["length"] = float(keep)
            else:
                data = make_joint(self.scene, j["id"], j["name"], j["type"], a, b, anchor, axis, **options)
        except (JointError, SceneError) as exc:
            self.error.setText(str(exc))
            self._show(self.list.currentRow())
            return
        j.clear()
        j.update(data)
        self.list.currentItem().setText(self._item_text(j))
        self.error.clear()
        self._show(self.list.currentRow())

    # ---- apply ---------------------------------------------------------------------------------------------
    def apply(self) -> bool:
        cur = self._current()
        try:
            validate_joints(copy.deepcopy(self.joints))
            self.editor.set_joints(self.joints)
        except (JointError, SceneError) as exc:
            self.error.setText(str(exc))
            return False
        self.error.clear()
        self.joints = self.scene.joints_snapshot()
        self._refresh_list(cur["id"] if cur else None)
        return True

    def _ok(self) -> None:
        if self.apply():
            self.accept()
