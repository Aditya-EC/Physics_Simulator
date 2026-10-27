"""Place Relative dialog: put the selection at a numeric offset from another object, a named frame or the
global frame, optionally with an orientation relative to it (psim/scene/placement.py)."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QPushButton,
                               QVBoxLayout)

from ... import numerics
from ...coordinates.scene_kinematics import reference_choices
from ...scene.editing import SceneEditor
from ...scene.placement import PlacementError, check_reference, current_offset, place
from ...scene.scene_manager import Scene, SceneError
from ...units.unit_registry import UnitRegistry
from ..widgets.inputs import VectorField


class PlaceRelativeDialog(QDialog):
    def __init__(self, scene: Scene, editor: SceneEditor, units: UnitRegistry, ids: list[str], parent=None) -> None:
        super().__init__(parent)
        self.scene, self.editor = scene, editor
        self.ids = [i for i in ids if i in scene.objects]
        self.primary = self.ids[-1] if self.ids else None      # the most recently selected object
        self.setWindowTitle("Place Relative To")
        lay = QVBoxLayout(self)
        names = ", ".join(scene.get(i).name for i in self.ids[:4]) + (" …" if len(self.ids) > 4 else "")
        info = QLabel(f"Moves <b>{names}</b>" + (
            f" as one rigid group; the offset applies to <b>{scene.get(self.primary).name}</b>."
            if len(self.ids) > 1 else "."))
        info.setWordWrap(True)
        lay.addWidget(info)
        form = QFormLayout()
        self.reference = QComboBox()
        try:
            check_reference(scene, self.ids, None)                 # selection problems (empty, locked)
            problem = ""
        except PlacementError as exc:
            problem = str(exc)
        for text, ref in ([] if problem else reference_choices(scene)):
            try:
                check_reference(scene, self.ids, ref)
            except PlacementError:
                continue                                     # the selection itself, its children, attached frames
            self.reference.addItem(text, ref)
        others = [k for k in range(self.reference.count()) if self.reference.itemData(k)
                  and not str(self.reference.itemData(k)).startswith("FRM-")]
        if others:
            self.reference.setCurrentIndex(others[0])
        form.addRow("Relative to", self.reference)
        self.axes = QComboBox()
        self.axes.addItem("Axes of the reference", "reference")
        self.axes.addItem("Global axes", "global")
        form.addRow("Offset along", self.axes)
        lu, au = units.preferred_unit("length"), units.preferred_unit("angle")
        self.offset = VectorField(3, units, lu, 6, -1e7, 1e7)
        form.addRow(f"Offset [{lu}]", self.offset)
        self.orient_mode = QComboBox()
        self.orient_mode.addItem("Keep the current orientation", "keep")
        self.orient_mode.addItem("Set relative to the reference", "set")
        form.addRow("Orientation", self.orient_mode)
        self.euler = VectorField(3, units, au, 6)
        self.euler.setToolTip("Rotation relative to the reference: extrinsic X, then Y, then Z")
        form.addRow(f"Rotation XYZ [{au}]", self.euler)
        lay.addLayout(form)
        self.current = QPushButton("Use Current Placement")
        self.current.setToolTip("Fill in the current offset and orientation of the object relative to the reference")
        self.current.clicked.connect(self.fill_current)
        lay.addWidget(self.current)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setObjectName("Muted")
        lay.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Apply
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self._ok)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        lay.addWidget(self.buttons)
        self.reference.currentIndexChanged.connect(lambda _i: self.fill_current())
        self.axes.currentIndexChanged.connect(lambda _i: self.fill_current())
        self.orient_mode.currentIndexChanged.connect(lambda _i: self._update_enabled())
        if problem or self.reference.count() == 0:
            self.status.setText(problem or "Nothing to place: select the objects to move first.")
            for b in (QDialogButtonBox.StandardButton.Ok, QDialogButtonBox.StandardButton.Apply):
                self.buttons.button(b).setEnabled(False)
        else:
            self.fill_current()
        self._update_enabled()

    def _update_enabled(self) -> None:
        self.euler.setEnabled(self.orient_mode.currentData() == "set")

    def fill_current(self) -> None:
        if not self.ids or self.reference.count() == 0:
            return
        try:
            d, rel_q = current_offset(self.scene, self.primary, self.reference.currentData(), self.axes.currentData())
        except (PlacementError, SceneError) as exc:
            self.status.setText(str(exc))
            return
        self.offset.set_si(d)
        self.euler.set_si(np.asarray(numerics.quat_to_euler(rel_q), float))
        self.status.setText("")

    def poses(self) -> dict:
        rel = None
        if self.orient_mode.currentData() == "set":
            rel = np.asarray(numerics.quat_from_euler(np.asarray(self.euler.si(), float)), float)
        return place(self.scene, self.ids, self.primary, self.reference.currentData(), self.offset.si(), rel,
                     self.axes.currentData())

    def apply(self) -> bool:
        try:
            poses = self.poses()
        except (PlacementError, SceneError) as exc:
            self.status.setText(str(exc))
            return False
        ref = self.reference.currentData()
        self.editor.set_world_poses(poses, f"Place Relative to {self.scene.reference_name(ref)}")
        self.status.setText(f"Placed relative to {self.scene.reference_name(ref)}.")
        return True

    def _ok(self) -> None:
        if self.apply():
            self.accept()
