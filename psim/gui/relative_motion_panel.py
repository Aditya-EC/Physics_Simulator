"""Relative Motion panel: a point's motion re-expressed in a frame vs. measured relative to that moving frame."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import numerics
from ..coordinates.kinematics import relative_motion, relative_rotation
from ..coordinates.scene_kinematics import reference_choices, reference_state, rigid_state
from ..scene.scene_manager import Scene, SceneError
from ..units.unit_registry import UnitRegistry

UNKNOWN = "— (computed while simulating)"
NA = object()  # not applicable to this column

# (key, label, quantity, tooltip)
ROWS = [
    ("position", "Position", "length", "Coordinates of the point in the frame"),
    ("orientation", "Orientation (Euler XYZ)", "angle", "Orientation of the object's axes relative to the frame"),
    ("velocity", "Velocity", "velocity", ""),
    ("acceleration", "Acceleration", "acceleration", ""),
    ("angular_velocity", "Angular velocity", "angular_velocity", ""),
    ("angular_acceleration", "Angular acceleration", "angular_acceleration", ""),
    ("transport_velocity", "  Transport velocity  v_F + ω_F×r", "velocity",
     "Velocity the point would have if it were fixed in the frame"),
    ("origin_acceleration", "  Frame origin acceleration  a_F", "acceleration", ""),
    ("euler_acceleration", "  Euler term  α_F×r", "acceleration", ""),
    ("centripetal_acceleration", "  Centripetal term  ω_F×(ω_F×r)", "acceleration", ""),
    ("coriolis_acceleration", "  Coriolis term  2ω_F×v_rel", "acceleration", ""),
]


class RelativeMotionPanel(QWidget):
    def __init__(self, scene: Scene, units: UnitRegistry, parent=None) -> None:
        super().__init__(parent)
        self.scene, self.units = scene, units
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        self.obj = QComboBox()
        self.obj.setMinimumWidth(180)
        self.point = QComboBox()
        self.point.addItem("Centre of mass", "com")
        self.point.addItem("Object origin", "origin")
        self.frame = QComboBox()
        self.frame.setMinimumWidth(200)
        self.follow = QCheckBox("Follow selection")
        self.follow.setChecked(True)
        copy_btn = QPushButton("Copy")
        copy_btn.setAutoDefault(False)
        copy_btn.setToolTip("Copy the table as tab-separated text")
        copy_btn.clicked.connect(self.copy)
        for w in (QLabel("Object"), self.obj, QLabel("Point"), self.point, QLabel("Frame"), self.frame, self.follow):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(copy_btn)
        lay.addLayout(bar)
        note = QLabel("<b>Expressed in frame</b> = absolute vector in the frame's axes (rotation only) · "
                      "<b>Relative to frame</b> = measured by an observer moving with the frame")
        note.setToolTip("Expressed in frame: the absolute (global) vector written with the frame's axes; it does "
                        "not depend on how the frame moves.\nRelative to frame: the time derivatives of the "
                        "point's coordinates in the frame, i.e. what an observer carried by the frame measures. "
                        "They differ whenever the frame translates or rotates.\nIndented rows break down the "
                        "acceleration: a = a_rel + a_F + α_F×r + ω_F×(ω_F×r) + 2ω_F×v_rel.")
        note.setObjectName("Muted")
        lay.addWidget(note)
        self.table = QTableWidget(len(ROWS), 3)
        self.table.setHorizontalHeaderLabels(["Quantity", "Expressed in frame", "Relative to frame"])
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(self.fontMetrics().height() + 6)
        for r, (_k, label, _q, tip) in enumerate(ROWS):
            item = QTableWidgetItem(label)
            item.setToolTip(tip)
            self.table.setItem(r, 0, item)
        lay.addWidget(self.table, 1)
        self.status = QLabel()
        self.status.setObjectName("Muted")
        lay.addWidget(self.status)

        self._timer = QTimer(self)
        self._timer.setInterval(100)  # 10 Hz is plenty for reading numbers
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self._choices_dirty = True
        for sig in (scene.objectsAdded, scene.objectsRemoved, scene.hierarchyChanged, scene.framesChanged,
                    scene.sceneReset):
            sig.connect(self._mark_dirty)
        scene.selectionChanged.connect(self._on_selection)
        for combo in (self.obj, self.point, self.frame):
            combo.currentIndexChanged.connect(self.refresh)
        self._rebuild_choices()

    # ---- choices -------------------------------------------------------------------------------------------
    def _mark_dirty(self, *_a) -> None:
        self._choices_dirty = True

    def _rebuild_choices(self) -> None:
        self._choices_dirty = False
        cur_obj, cur_frame = self.obj.currentData(), self.frame.currentData()
        for combo in (self.obj, self.frame):
            combo.blockSignals(True)
            combo.clear()
        for o in self.scene.objects.values():
            self.obj.addItem(f"{o.name} ({o.id})", o.id)
        oid = cur_obj if self.obj.findData(cur_obj) >= 0 else (self.scene.primary.id if self.scene.primary
                                                               else self.obj.itemData(0))
        self.obj.setCurrentIndex(max(self.obj.findData(oid), 0))
        for label, ref in reference_choices(self.scene, include_parent_of=self.obj.currentData()):
            self.frame.addItem(label, ref)
        self.frame.setCurrentIndex(max(self.frame.findData(cur_frame), 0))
        for combo in (self.obj, self.frame):
            combo.blockSignals(False)

    def _on_selection(self, ids: list) -> None:
        if self.follow.isChecked() and ids:
            i = self.obj.findData(ids[-1])
            if i >= 0:
                self.obj.setCurrentIndex(i)
                self._choices_dirty = True  # "Parent (...)" entry depends on the object

    def set_frame(self, ref: str | None) -> None:
        if self._choices_dirty:
            self._rebuild_choices()
        self.frame.setCurrentIndex(max(self.frame.findData(ref), 0))

    # ---- values ----------------------------------------------------------------------------------------------
    def _fmt(self, value, quantity: str) -> str:
        if value is None:
            return UNKNOWN
        unit = self.units.preferred_unit(quantity)
        disp = np.asarray(self.units.from_si(np.asarray(value, float), unit), float)
        disp = np.where(np.abs(disp) < 1e-12 * max(1.0, float(np.max(np.abs(disp)))), 0.0, disp)
        return f"({', '.join(f'{c:.5g}' for c in disp)}) {unit}   |·| = {np.linalg.norm(disp):.5g}"

    def compute(self) -> dict[str, tuple] | None:
        """{row key: (expressed-in-frame value, relative value)} in SI, or None if nothing to show.

        A value is None when it is unknown (accelerations before a run) and NA when it does not apply.
        """
        oid = self.obj.currentData()
        if oid is None or oid not in self.scene.objects:
            return None
        ref = self.frame.currentData()
        if ref == "@parent":
            ref = self.scene.get(oid).parent_id
        rs = rigid_state(self.scene, oid)
        fs = reference_state(self.scene, ref)
        p = rs.com if self.point.currentData() == "com" else rs.origin
        rm = relative_motion(p, rs.point_velocity(p), rs.point_acceleration(p), fs)
        rr = relative_rotation(rs.quaternion, rs.angular_velocity, rs.angular_acceleration, fs)
        rt = fs.rotation.T
        alpha_c = None if rs.angular_acceleration is None else rt @ rs.angular_acceleration
        return {
            "position": (rm.position, rm.position),
            "orientation": (NA, np.asarray(numerics.quat_to_euler(rr.orientation), float)),
            "velocity": (rm.velocity_components, rm.velocity_relative),
            "acceleration": (rm.acceleration_components, rm.acceleration_relative),
            "angular_velocity": (rr.angular_velocity_components, rr.angular_velocity_relative),
            "angular_acceleration": (alpha_c, rr.angular_acceleration_relative),
            "transport_velocity": (NA, rm.transport_velocity),
            "origin_acceleration": (NA, rm.origin_acceleration),
            "euler_acceleration": (NA, rm.euler_acceleration),
            "centripetal_acceleration": (NA, rm.centripetal_acceleration),
            "coriolis_acceleration": (NA, rm.coriolis_acceleration),
        }

    def refresh(self) -> None:
        if not self.isVisible():
            return
        if self._choices_dirty:
            self._rebuild_choices()
        try:
            values = self.compute()
        except (SceneError, ValueError) as exc:
            self.status.setText(str(exc))
            values = None
        else:
            self.status.setText(self._status_text())
        for r, (key, _label, quantity, _tip) in enumerate(ROWS):
            pair = values[key] if values else (NA, NA)
            for c, v in ((1, pair[0]), (2, pair[1])):
                if values is None:
                    text = ""
                elif v is NA:
                    text = "—"
                else:
                    text = self._fmt(v, quantity)
                item = self.table.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(r, c, item)
                if item.text() != text:
                    item.setText(text)

    def _status_text(self) -> str:
        if self.scene.simulating:
            return "Live values from the running simulation."
        return ("Not simulating: velocities are the objects' initial values; accelerations are computed by the "
                "processes once the simulation runs.")

    def copy(self) -> None:
        lines = ["\t".join(self.table.horizontalHeaderItem(c).text() for c in range(3))]
        for r in range(self.table.rowCount()):
            lines.append("\t".join((self.table.item(r, c).text() if self.table.item(r, c) else "") for c in range(3)))
        QApplication.clipboard().setText("\n".join(lines))
