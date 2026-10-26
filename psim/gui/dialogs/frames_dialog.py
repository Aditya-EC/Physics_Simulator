"""Coordinates ▸ Reference Frames: create and edit named reference frames (applied as one undoable step)."""
from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from ... import numerics
from ...coordinates.frames import FrameError, validate_frames
from ...scene.editing import SceneEditor
from ...scene.scene_manager import Scene, SceneError
from ...units.unit_registry import UnitRegistry
from ..widgets.inputs import ColorButton, NumberField, VectorField


class ReferenceFramesDialog(QDialog):
    def __init__(self, scene: Scene, editor: SceneEditor, units: UnitRegistry, select: str | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reference Frames")
        self.resize(820, 520)
        self.scene, self.editor, self.units = scene, editor, units
        self.frames: list[dict] = scene.frames_snapshot()
        self._loading = False
        lay = QVBoxLayout(self)
        note = QLabel("Named frames can be fixed in the global frame or attached to an object or assembly. An "
                      "attached frame's origin is a point fixed on its parent; its axes either rotate with the "
                      "parent or keep a fixed global orientation. Frames are used to read and edit poses, to "
                      "express vectors, in the Relative Motion panel and in formulas as "
                      "<code>frame(\"FRM-0001\")</code>.")
        note.setWordWrap(True)
        note.setObjectName("Muted")
        lay.addWidget(note)
        body = QHBoxLayout()
        lay.addLayout(body, 1)

        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show)
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, slot in (("Add", self._add), ("Add at Selection", self._add_at_selection),
                           ("Remove", self._remove)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(slot)
            row.addWidget(b)
            if text == "Add at Selection":
                b.setEnabled(scene.primary is not None)
                b.setToolTip("Attach a new frame to the selected object, at its origin")
        left.addLayout(row)
        body.addLayout(left, 2)

        self.form_widget = QWidget()
        form = QFormLayout(self.form_widget)
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self._store("name", self.name.text().strip()))
        form.addRow("Name", self.name)
        self.fid = QLabel()
        form.addRow("ID", self.fid)
        self.parent_combo = QComboBox()
        self.parent_combo.currentIndexChanged.connect(self._parent_changed)
        form.addRow("Attached to", self.parent_combo)
        lu = units.preferred_unit("length")
        au = units.preferred_unit("angle")
        self.position = VectorField(3, units, lu, 4, -1e7, 1e7)
        self.position.committed.connect(lambda v: self._store("position", [float(c) for c in v]))
        self.position_label = QLabel()
        form.addRow(self.position_label, self.position)
        self.euler = VectorField(3, units, au, 4)
        self.euler.committed.connect(
            lambda v: self._store("orientation", [float(c) for c in numerics.quat_from_euler(np.asarray(v))]))
        self.euler_label = QLabel()
        form.addRow(self.euler_label, self.euler)
        self.follow = QCheckBox("Axes rotate with the parent")
        self.follow.toggled.connect(self._follow_changed)
        form.addRow("", self.follow)
        self.visible = QCheckBox("Show in viewport")
        self.visible.toggled.connect(lambda v: self._store("visible", bool(v)))
        form.addRow("", self.visible)
        self.show_label = QCheckBox("Show name label")
        self.show_label.toggled.connect(lambda v: self._store("show_label", bool(v)))
        form.addRow("", self.show_label)
        self.axis_length = NumberField(units, lu, 4, 1e-6, 1e6)
        self.axis_length.committed.connect(lambda v: self._store("axis_length", float(v)))
        form.addRow(f"Axis length [{lu}]", self.axis_length)
        self.color = ColorButton()
        self.color.colorChanged.connect(lambda v: self._store("color", list(v)))
        form.addRow("Label colour", self.color)
        self.description = QLineEdit()
        self.description.editingFinished.connect(lambda: self._store("description", self.description.text()))
        form.addRow("Description", self.description)
        body.addWidget(self.form_widget, 3)

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
        self._refresh_list(select)

    # ---- list ------------------------------------------------------------------------------------------
    def _refresh_list(self, select: str | None = None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for f in self.frames:
            item = QListWidgetItem(f"{f['name']}  ({f['id']})")
            item.setData(Qt.ItemDataRole.UserRole, f["id"])
            self.list.addItem(item)
        self.list.blockSignals(False)
        ids = [f["id"] for f in self.frames]
        row = ids.index(select) if select in ids else (0 if ids else -1)
        self.list.setCurrentRow(row)
        self._show(row)

    def _current(self) -> dict | None:
        row = self.list.currentRow()
        return self.frames[row] if 0 <= row < len(self.frames) else None

    def _unique_name(self, base: str) -> str:
        names = {f["name"].lower() for f in self.frames}
        k = 1
        while f"{base} {k}".lower() in names:
            k += 1
        return f"{base} {k}"

    def _add(self, parent: str | None = None) -> None:
        fid = self.scene.allocate_frame_id()
        self.frames.append({"id": fid, "name": self._unique_name("Frame"), "parent": parent,
                            "position": [0.0, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0],
                            "follow_rotation": True, "visible": True, "show_label": True, "axis_length": 0.6,
                            "color": [0.98, 0.78, 0.32], "description": ""})
        self._refresh_list(fid)

    def _add_at_selection(self) -> None:
        o = self.scene.primary
        if o is not None:
            self._add(o.id)

    def _remove(self) -> None:
        cur = self._current()
        if cur is not None:
            self.frames.remove(cur)
            self._refresh_list()

    # ---- editor --------------------------------------------------------------------------------------------
    def _show(self, row: int) -> None:
        f = self._current()
        self.form_widget.setEnabled(f is not None)
        if f is None:
            return
        self._loading = True
        try:
            self.name.setText(f["name"])
            self.fid.setText(f["id"])
            self.parent_combo.clear()
            self.parent_combo.addItem("— Global (fixed) —", None)
            for o in self.scene.objects.values():
                self.parent_combo.addItem(f"{o.name} ({o.id})", o.id)
            if f["parent"] and self.parent_combo.findData(f["parent"]) < 0:
                self.parent_combo.addItem(f"⚠ missing object {f['parent']}", f["parent"])
            self.parent_combo.setCurrentIndex(max(self.parent_combo.findData(f["parent"]), 0))
            self.position.set_si(f["position"])
            self.euler.set_si(numerics.quat_to_euler(np.asarray(f["orientation"], float)))
            self.follow.setChecked(bool(f.get("follow_rotation", True)))
            self.visible.setChecked(bool(f.get("visible", True)))
            self.show_label.setChecked(bool(f.get("show_label", True)))
            self.axis_length.set_si(float(f.get("axis_length", 0.6)))
            self.color.set_rgb(f.get("color", [0.98, 0.78, 0.32]))
            self.description.setText(f.get("description", ""))
            self._update_labels(f)
        finally:
            self._loading = False

    def _update_labels(self, f: dict) -> None:
        lu, au = self.units.preferred_unit("length"), self.units.preferred_unit("angle")
        attached = f["parent"] is not None
        self.position_label.setText(f"Origin offset, parent axes [{lu}]" if attached else f"Origin (global) [{lu}]")
        rotating = attached and f.get("follow_rotation", True)
        self.euler_label.setText(f"Orientation relative to parent [{au}]" if rotating
                                 else f"Orientation (global) [{au}]")
        self.follow.setEnabled(attached)

    def _store(self, key: str, value) -> None:
        f = self._current()
        if f is None or self._loading:
            return
        f[key] = value
        if key == "name":
            self.list.currentItem().setText(f"{value}  ({f['id']})")
        self._update_labels(f)
        self.error.clear()

    def _parent_changed(self, _i: int) -> None:
        if self._loading:
            return
        f = self._current()
        if f is None:
            return
        new = self.parent_combo.currentData()
        if new == f["parent"]:
            return
        # keep the frame where it is: convert its pose into the new parent's frame
        try:
            wp, wq = self._world_pose(f)
            if new is None:
                f["position"], q = [float(c) for c in wp], wq
            else:
                pp, pq = self.scene.world_pose(new)
                ip, iq = numerics.pose_inverse(pp, pq)
                lp, lq = numerics.pose_compose(ip, iq, wp, wq)
                f["position"] = [float(c) for c in lp]
                q = lq if f.get("follow_rotation", True) else wq
            f["orientation"] = [float(c) for c in numerics.quat_normalize(q)]
        except SceneError:
            pass
        f["parent"] = new
        self._show(self.list.currentRow())

    def _follow_changed(self, on: bool) -> None:
        f = self._current()
        if f is None or self._loading or bool(f.get("follow_rotation", True)) == on:
            return
        if f["parent"] is not None:  # keep the axes' global orientation; only the stored form changes
            try:
                _wp, wq = self._world_pose(f)
                _pp, pq = self.scene.world_pose(f["parent"])
                q = numerics.quat_multiply(numerics.quat_conjugate(pq), wq) if on else wq
                f["orientation"] = [float(c) for c in numerics.quat_normalize(q)]
            except SceneError:
                pass
        f["follow_rotation"] = on
        self._show(self.list.currentRow())

    def _world_pose(self, f: dict):
        if f["parent"] is None:
            return np.asarray(f["position"], float), np.asarray(f["orientation"], float)
        pp, pq = self.scene.world_pose(f["parent"])
        origin = np.asarray(pp, float) + numerics.quat_rotate(pq, f["position"])
        q = numerics.quat_multiply(pq, f["orientation"]) if f.get("follow_rotation", True) else f["orientation"]
        return np.asarray(origin, float), np.asarray(q, float)

    # ---- apply ---------------------------------------------------------------------------------------------
    def apply(self) -> bool:
        cur = self._current()
        try:
            validate_frames(copy.deepcopy(self.frames))
            self.editor.set_frames(self.frames)
        except (FrameError, SceneError) as exc:
            self.error.setText(str(exc))
            return False
        self.error.clear()
        self.frames = self.scene.frames_snapshot()
        self._refresh_list(cur["id"] if cur else None)
        return True

    def _ok(self) -> None:
        if self.apply():
            self.accept()
