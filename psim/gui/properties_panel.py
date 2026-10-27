"""Properties Inspector: organised, searchable editing of the selected object(s).

Every edit goes through SceneEditor (undoable, validated). Values are shown in
display units (per-property unit selectors) while stored in SI. Each row is
tagged with its origin: U user-controlled, C calculated, D derived,
S simulated.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget)

from .. import numerics
from ..coordinates.kinematics import rotation_of
from ..coordinates.scene_kinematics import reference_choices
from ..properties.property_system import Origin, PropertyError, ValueType
from ..scene.editing import SceneEditor
from ..scene.materials import PHYSICAL_MATERIALS
from ..scene.object_base import ARROW_QUANTITIES, DYNAMIC_STATE, OVERLAY_LABELS, SceneObject
from ..scene.scene_manager import Scene, SceneError
from ..units.unit_registry import UnitRegistry
from . import icons
from .dialogs.custom_property_dialog import CustomPropertyDialog
from .widgets.inputs import ColorButton, IntField, NumberField, UnitCombo, VectorField
from .widgets.section import Section

LABEL_UNIT_QUANTITIES = ("length", "velocity", "acceleration", "angle", "angular_velocity", "force", "mass")
INITIAL_TIP = "Initial value (user-controlled). During a simulation this quantity will be updated by processes."


class PropertiesPanel(QWidget):
    errorOccurred = Signal(str)
    statusMessage = Signal(str)
    assignProcessRequested = Signal(list)   # object ids
    editAssignmentRequested = Signal(str)   # assignment uid
    clearTrajectoryRequested = Signal(list)  # object ids
    exportTrajectoryRequested = Signal(str)  # object id
    jointsRequested = Signal(str)           # object id: open the joints dialog for it

    def __init__(self, scene: Scene, editor: SceneEditor, units: UnitRegistry, parent=None) -> None:
        super().__init__(parent)
        self.scene, self.editor, self.units = scene, editor, units
        self._ids: list[str] = []
        self._signature = None
        self._refreshers: list[Callable[[], None]] = []
        self._sections: list[Section] = []
        self._frame: str | None = None  # pose frame: None = global, "@parent", an object id or a frame id
        self._vec_frame: str | None = None  # frame whose axes physical vectors are expressed in
        self._custom_sort = "definition"
        self.simulated_ids: set[str] = set()   # objects whose dynamic state a running simulation owns

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QWidget()
        hl = QVBoxLayout(head)
        hl.setContentsMargins(10, 8, 10, 8)
        hl.setSpacing(4)
        row = QHBoxLayout()
        self.title = QLabel("No selection")
        self.title.setObjectName("InspectorTitle")
        row.addWidget(self.title, 1)
        self.copy_id = QToolButton()
        self.copy_id.setIcon(icons.icon("copy"))
        self.copy_id.setToolTip("Copy object ID")
        self.copy_id.clicked.connect(self._copy_id)
        row.addWidget(self.copy_id)
        hl.addLayout(row)
        self.subtitle = QLabel("")
        self.subtitle.setObjectName("Muted")
        self.subtitle.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hl.addWidget(self.subtitle)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter properties…")
        self.filter.setClearButtonEnabled(True)
        self.filter.addAction(icons.icon("search"), QLineEdit.ActionPosition.LeadingPosition)
        self.filter.textChanged.connect(self._apply_filter)
        hl.addWidget(self.filter)
        lay.addWidget(head)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay.addWidget(self.scroll, 1)
        legend = QLabel("<b>U</b> user · <b>C</b> calculated · <b>D</b> derived · <b>S</b> simulated")
        legend.setObjectName("Muted")
        legend.setContentsMargins(10, 4, 10, 4)
        lay.addWidget(legend)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self._sync)
        scene.selectionChanged.connect(lambda _ids: self._schedule())
        scene.objectsChanged.connect(self._on_changed)
        scene.objectsRemoved.connect(lambda _ids: self._schedule())
        scene.hierarchyChanged.connect(self._schedule)
        scene.sceneReset.connect(self._schedule)
        scene.processesChanged.connect(self._schedule)
        scene.framesChanged.connect(self._schedule)
        scene.jointsChanged.connect(self._schedule)
        self._sync()

    def set_simulated_ids(self, ids) -> None:
        self.simulated_ids = set(ids)
        self._schedule()

    # ---- update scheduling -----------------------------------------------------------------------
    def _schedule(self) -> None:
        self._refresh_timer.start()

    def _on_changed(self, ids) -> None:
        if set(ids) & set(self._ids):
            self._schedule()

    def _objs(self) -> list[SceneObject]:
        return [self.scene.objects[i] for i in self._ids if i in self.scene.objects]

    def _compute_signature(self, ids):
        objs = [self.scene.objects[i] for i in ids]
        return (tuple(ids), tuple(o.type_id for o in objs), tuple(tuple(o.custom) for o in objs),
                tuple(o.mass_mode for o in objs), tuple(o.parent_id for o in objs), self._frame, self._vec_frame,
                tuple((f.id, f.name) for f in self.scene.frames),
                len(self.scene.objects), self._custom_sort,  # object count: frame/parent choices
                frozenset(self.simulated_ids & set(ids)),
                tuple((a.uid, a.enabled, a.process_id) for i in ids[-1:] for a in self.scene.assignments_for(i)),
                tuple((j.id, j.name, j.type, j.enabled) for i in ids[-1:] for j in self.scene.joints
                      if i in (j.body_a, j.body_b)))

    def _sync(self) -> None:
        ids = [i for i in self.scene.selection if i in self.scene.objects]
        if not self._ref_valid(self._frame, ids):
            self._frame = None
        if not self._ref_valid(self._vec_frame, ids):
            self._vec_frame = None
        sig = self._compute_signature(ids)
        if sig != self._signature:
            self._ids = ids
            self._signature = sig
            self._rebuild()
        else:
            self._refresh()

    def _refresh(self) -> None:
        for r in self._refreshers:
            r()
        self._update_header()

    # ---- building -----------------------------------------------------------------------------------
    def _rebuild(self) -> None:
        old = self.scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self._refreshers = []
        self._sections = []
        body = QWidget()
        self._body_lay = QVBoxLayout(body)
        self._body_lay.setContentsMargins(0, 0, 0, 0)
        self._body_lay.setSpacing(0)
        objs = self._objs()
        if not objs:
            msg = QLabel("Select an object in the viewport or the scene outliner to inspect and edit its "
                         "properties.\n\nDrag objects from the Object Library into the viewport to add them.")
            msg.setWordWrap(True)
            msg.setObjectName("Muted")
            msg.setContentsMargins(14, 14, 14, 14)
            self._body_lay.addWidget(msg)
        elif len(objs) == 1:
            o = objs[0]
            self._build_identity(objs)
            self._build_geometry(objs)
            self._build_transform(o)
            if o.typedef.physical:
                self._build_physical(o)
                self._build_collision(objs)
            self._build_coordinates(o)
            self._build_appearance(objs)
            self._build_display(objs)
            if o.typedef.physical or o.type_id.startswith("group."):
                self._build_trajectory(objs)
            self._build_custom(objs)
            self._build_processes(o)
        else:
            self._build_identity(objs)
            if len({o.type_id for o in objs}) == 1:
                self._build_geometry(objs)
            sec = self._section("transform", "Transform")
            self._add_prop(sec, "scale", objs)
            note = QLabel("Use the viewport gizmo to move or rotate several objects together.")
            note.setObjectName("Muted")
            note.setWordWrap(True)
            sec.add_widget(note)
            if all(o.typedef.physical and o.typedef.mass_model != "point" for o in objs):
                sec = self._section("physical", "Physical Properties")
                self._add_prop(sec, "mass_mode", objs, origin=Origin.USER)
                self._add_prop(sec, "density", objs, origin=Origin.USER)
            if all(o.typedef.physical for o in objs):
                self._build_collision(objs)
            self._build_appearance(objs)
            self._build_display(objs)
            if all(o.typedef.physical or o.type_id.startswith("group.") for o in objs):
                self._build_trajectory(objs)
            self._build_custom(objs)
        self._body_lay.addStretch(1)
        self.scroll.setWidget(body)
        self._update_header()
        self._apply_filter(self.filter.text())

    def _update_header(self) -> None:
        objs = self._objs()
        self.copy_id.setVisible(len(objs) == 1)
        if not objs:
            self.title.setText("No selection")
            self.subtitle.setText("")
        elif len(objs) == 1:
            o = objs[0]
            self.title.setText(o.name)
            lock = "  ·  locked" if o.locked else ""
            self.subtitle.setText(f"{o.id}  ·  {o.typedef.name}{lock}")
        else:
            self.title.setText(f"{len(objs)} objects selected")
            types = sorted({o.typedef.name for o in objs})
            self.subtitle.setText(", ".join(types[:4]) + ("…" if len(types) > 4 else ""))

    def _copy_id(self) -> None:
        objs = self._objs()
        if len(objs) == 1:
            QGuiApplication.clipboard().setText(objs[0].id)
            self.statusMessage.emit(f"Copied {objs[0].id} to the clipboard")

    def _section(self, key: str, title: str) -> Section:
        s = Section(key, title)
        self._body_lay.addWidget(s)
        self._sections.append(s)
        return s

    def _apply_filter(self, text: str) -> None:
        for s in self._sections:
            s.apply_filter(text)

    # ---- generic property editors ---------------------------------------------------------------------
    def _commit(self, objs: list[SceneObject], path: str, value: Any) -> None:
        try:
            self.editor.set_property([o.id for o in objs], path, value)
        except (PropertyError, SceneError) as exc:
            self.errorOccurred.emit(str(exc))
            QMessageBox.warning(self, "Invalid value", str(exc))
            self._refresh()

    def _unit_combo(self, path: str, quantity: str | None, setter: Callable[[str], None]) -> UnitCombo | None:
        if not quantity:
            return None
        combo = UnitCombo(self.units, quantity, self.units.display_unit(path, quantity))

        def changed(sym: str) -> None:
            self.units.set_display_unit(path, quantity, sym)
            setter(sym)
        combo.unitChanged.connect(changed)
        return combo

    def _add_prop(self, sec: Section, path: str, objs: list[SceneObject], *, label: str | None = None,
                  origin: Origin | None = None, read_only: bool = False, tooltip: str = "",
                  quantity: str | None = None, frame_rotation: Callable[[], np.ndarray] | None = None) -> QWidget:
        primary = objs[-1]
        d = primary.property_def(path)
        label = label or d.label
        ro = read_only or d.read_only
        quantity = quantity or d.quantity
        origin = origin or d.origin
        if path in DYNAMIC_STATE and primary.id in self.simulated_ids and path not in ("linear_acceleration",
                                                                                     "angular_acceleration"):
            origin = Origin.SIMULATED
        unit_widget = None
        t = d.vtype

        if t == ValueType.FLOAT:
            w = NumberField(self.units, self.units.display_unit(path, quantity) if quantity else None, d.precision,
                            d.minimum, d.maximum)
            w.setReadOnly(ro)
            w.committed.connect(lambda v: self._commit(objs, path, v))
            unit_widget = self._unit_combo(path, quantity, w.set_unit)
            self._refreshers.append(lambda: w.set_si(primary.get(path)))
        elif t == ValueType.INT:
            w = IntField(int(d.minimum) if d.minimum is not None else -2**31,
                         int(d.maximum) if d.maximum is not None else 2**31 - 1)
            w.setReadOnly(ro)
            w.committed.connect(lambda v: self._commit(objs, path, v))
            self._refreshers.append(lambda: w.set_int(primary.get(path)))
        elif t == ValueType.BOOL:
            w = QCheckBox()
            w.setEnabled(not ro)
            w.toggled.connect(lambda v: self._commit(objs, path, v))
            self._refreshers.append(lambda: _set_blocked(w, w.setChecked, bool(primary.get(path))))
        elif t == ValueType.ENUM:
            w = QComboBox()
            w.addItems(list(d.choices))
            w.setEnabled(not ro)
            w.activated.connect(lambda _i: self._commit(objs, path, w.currentText()))
            self._refreshers.append(lambda: _set_blocked(w, w.setCurrentText, str(primary.get(path))))
        elif t in (ValueType.VEC2, ValueType.VEC3):
            w = VectorField(t.size, self.units, self.units.display_unit(path, quantity) if quantity else None,
                            d.precision, d.minimum, d.maximum)
            w.setReadOnly(ro)
            unit_widget = self._unit_combo(path, quantity, w.set_unit)
            if frame_rotation is None or t != ValueType.VEC3:
                w.committed.connect(lambda v: self._commit(objs, path, v))
                self._refreshers.append(lambda: w.set_si(primary.get(path)))
            else:  # shown in another frame's axes, stored globally
                w.committed.connect(lambda v: self._commit(objs, path, [float(c) for c in
                                                                       frame_rotation() @ np.asarray(v, float)]))
                self._refreshers.append(lambda: w.set_si(frame_rotation().T @ np.asarray(primary.get(path), float)))
        elif t == ValueType.COLOR:
            w = ColorButton()
            w.setEnabled(not ro)
            w.colorChanged.connect(lambda v: self._commit(objs, path, v))
            self._refreshers.append(lambda: w.set_rgb(primary.get(path)))
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(w)
            hl.addStretch(1)
            sec.add_row(label, holder, None, origin, tooltip or d.description, search=path)
            self._refreshers[-1]()
            return w
        else:  # STRING
            w = QLineEdit()
            w.setReadOnly(ro)
            w.editingFinished.connect(lambda: w.text() != str(primary.get(path)) and self._commit(objs, path, w.text()))
            self._refreshers.append(lambda: _set_blocked(w, w.setText, str(primary.get(path))))
        sec.add_row(label, w, unit_widget, origin, tooltip or d.description, search=path)
        self._refreshers[-1]()
        return w

    def _add_readout(self, sec: Section, label: str, getter: Callable[[], Any], quantity: str | None, n: int,
                     origin: Origin, path_key: str, tooltip: str = "", precision: int = 5,
                     labels: tuple[str, ...] | None = None) -> None:
        """Read-only calculated/derived value (scalar if n == 1, else vector)."""
        if n == 1:
            w = NumberField(self.units, self.units.display_unit(path_key, quantity) if quantity else None, precision)
        else:
            w = VectorField(n, self.units, self.units.display_unit(path_key, quantity) if quantity else None,
                            precision, labels=labels)
        w.setReadOnly(True)
        unit_widget = self._unit_combo(path_key, quantity, w.set_unit)

        def refresh():
            v = getter()
            if v is None:
                v = 0.0 if n == 1 else [0.0] * n
            w.set_si(v)
        self._refreshers.append(refresh)
        refresh()
        sec.add_row(label, w, unit_widget, origin, tooltip, search=path_key)

    # ---- sections ---------------------------------------------------------------------------------------
    def _build_identity(self, objs: list[SceneObject]) -> None:
        sec = self._section("identity", "Identity")
        single = len(objs) == 1
        if single:
            o = objs[0]
            idw = QLineEdit(o.id)
            idw.setReadOnly(True)
            idw.setToolTip("Unique, persistent object ID (never changes)")
            sec.add_row("Unique ID", idw, origin=Origin.CALCULATED, search="id")
            self._add_prop(sec, "name", objs)
            tw = QLineEdit(o.typedef.name)
            tw.setReadOnly(True)
            tw.setToolTip(f"{o.type_id}\n{o.typedef.description}")
            sec.add_row("Type", tw, search="type")
            self._add_prop(sec, "description", objs)
        self._add_prop(sec, "tags", objs)
        self._add_prop(sec, "visible", objs)
        self._add_prop(sec, "locked", objs)

    def _build_geometry(self, objs: list[SceneObject]) -> None:
        primary = objs[-1]
        td = primary.typedef
        if not td.geometry:
            return
        sec = self._section("geometry", "Geometry")
        for d in td.geometry:
            if d.key == "asset":
                w = QLineEdit(f"{d.default or ''}")
                from ..scene.builtin_types import mesh_assets
                asset = primary.params.get("asset", "")
                w.setText(mesh_assets.name(asset) if mesh_assets.get(asset) else f"missing: {asset}")
                w.setReadOnly(True)
                sec.add_row("Mesh", w, origin=Origin.USER, search="mesh asset")
                continue
            self._add_prop(sec, f"params.{d.key}", objs, origin=Origin.USER)
        for key, (label, _v, quantity) in primary.derived_geometry().items():
            self._add_readout(sec, label, lambda k=key: primary.derived_geometry()[k][1], quantity, 1,
                              Origin.DERIVED, f"derived.{key}")
        if len(objs) == 1:
            m = primary.mesh()
            info = QLabel(f"{m.triangle_count:,} triangles · {'closed' if m.solid else 'open'} surface")
            info.setObjectName("Muted")
            sec.add_widget(info, search="triangles mesh")

    def _ref_valid(self, ref: str | None, ids: list[str]) -> bool:
        if ref in (None, "@parent", "@self"):
            return True
        if ref.startswith("FRM-"):  # a detached frame (parent deleted) falls back to global
            f = self.scene.frame(ref)
            return f is not None and (f.parent is None or f.parent in self.scene.objects)
        return ref in self.scene.objects and ref not in ids

    def _frame_choices(self, o: SceneObject) -> list[tuple[str, str | None]]:
        return reference_choices(self.scene, exclude=o.id, include_parent_of=o.id)

    def _resolve_ref(self, o: SceneObject, ref: str | None) -> str | None:
        if ref == "@parent":
            return o.parent_id if o.parent_id in self.scene.objects else None
        if ref == "@self":
            return o.id
        return ref

    def _frame_id(self, o: SceneObject) -> str | None:
        return self._resolve_ref(o, self._frame)

    def _build_transform(self, o: SceneObject) -> None:
        sec = self._section("transform", "Transform")
        combo = QComboBox()
        for text, data in self._frame_choices(o):
            combo.addItem(text, data)
        idx = combo.findData(self._frame)
        combo.setCurrentIndex(max(idx, 0))
        combo.setToolTip("Coordinate frame in which position and orientation are displayed and edited")

        def frame_changed(_i):
            self._frame = combo.currentData()
            self._schedule()
        combo.currentIndexChanged.connect(frame_changed)
        sec.add_row("Relative to", combo, search="frame relative coordinates")

        def pose():
            return self.scene.pose_in_frame(o.id, self._frame_id(o))

        def set_pose(position=None, orientation=None):
            try:
                self.editor.set_pose_in_frame(o.id, self._frame_id(o), position, orientation)
            except (PropertyError, SceneError, ValueError) as exc:
                self.errorOccurred.emit(str(exc))
                self._refresh()

        locked = self.scene.is_effectively_locked(o.id)
        pose_origin = Origin.SIMULATED if o.id in self.simulated_ids else Origin.USER
        pos = VectorField(3, self.units, self.units.display_unit("position", "length"), 4, -1e7, 1e7)
        pos.setReadOnly(locked)
        pos.committed.connect(lambda v: set_pose(position=np.asarray(v)))
        self._refreshers.append(lambda: pos.set_si(pose()[0]))
        sec.add_row("Position", pos, self._unit_combo("position", "length", pos.set_unit), pose_origin,
                    "Position of the object's local origin" + ("" if not locked else " (locked)"), search="position")
        eul = VectorField(3, self.units, self.units.display_unit("orientation_euler", "angle"), 4)
        eul.setReadOnly(locked)
        eul.committed.connect(lambda v: set_pose(orientation=numerics.quat_from_euler(np.asarray(v))))
        self._refreshers.append(lambda: eul.set_si(numerics.quat_to_euler(pose()[1])))
        sec.add_row("Euler angles", eul, self._unit_combo("orientation_euler", "angle", eul.set_unit), pose_origin,
                    "Extrinsic rotations about X, then Y, then Z (R = Rz·Ry·Rx)", search="orientation euler rotation")
        quat = VectorField(4, None, None, 6, -1.0, 1.0, labels=("W", "X", "Y", "Z"))
        quat.setReadOnly(locked)
        quat.committed.connect(lambda v: set_pose(orientation=np.asarray(v)) if np.linalg.norm(v) > 1e-9 else None)
        self._refreshers.append(lambda: quat.set_si(pose()[1]))
        sec.add_row("Quaternion", quat, None, pose_origin, "Unit quaternion (w, x, y, z); normalised on entry",
                    search="orientation quaternion")
        self._add_prop(sec, "scale", [o])
        pos.set_si(pose()[0])

    def _build_physical(self, o: SceneObject) -> None:
        sec = self._section("physical", "Physical Properties")
        mp = o.mass_properties
        td = o.typedef
        model_text = {"solid": "Solid body (volume integral of the mesh)", "shell": "Thin shell (area integral)",
                      "point": "Point mass (no rotational inertia)"}[td.mass_model]
        if mp()["approximate"]:
            model_text = "Open mesh treated as a thin shell (approximate)"
        if td.rigid_body:
            model_text += " · rigid body"
        info = QLabel(model_text)
        info.setObjectName("Muted")
        info.setWordWrap(True)
        sec.add_widget(info, search="mass model rigid")

        if td.mass_model != "point":
            mat = QComboBox()
            mat.addItem("Apply material density…", None)
            for name, rho in PHYSICAL_MATERIALS.items():
                mat.addItem(f"{name} ({rho:g} kg/m³)", rho)
            mat.setEnabled(mp()["density_quantity"] == "density")

            def apply_material(i):
                rho = mat.itemData(i)
                if rho is None:
                    return
                self.editor.edit([o.id], lambda: (o.set("mass_mode", "density"), o.set("density", rho)),
                                 f"Apply {mat.itemText(i).split(' (')[0]} density")
                mat.setCurrentIndex(0)
            mat.activated.connect(apply_material)
            sec.add_row("Material", mat, search="physical material density preset")
            self._add_prop(sec, "mass_mode", [o], origin=Origin.USER)
            dq = mp()["density_quantity"]
            if o.mass_mode == "density":
                self._add_prop(sec, "density", [o], origin=Origin.USER, quantity=dq,
                               label="Density" if dq == "density" else "Areal density")
                self._add_readout(sec, "Mass", lambda: mp()["mass"], "mass", 1, Origin.CALCULATED, "mass")
            else:
                self._add_prop(sec, "mass_input", [o], origin=Origin.USER)
                self._add_readout(sec, "Density" if dq == "density" else "Areal density", lambda: mp()["density"],
                                  dq, 1, Origin.DERIVED, "density_out")
            if mp()["volume"] is not None:
                self._add_readout(sec, "Volume", lambda: mp()["volume"], "volume", 1, Origin.CALCULATED, "volume")
            else:
                self._add_readout(sec, "Surface area", lambda: mp()["area"], "area", 1, Origin.CALCULATED, "area")
            self._add_readout(sec, "Center of mass", lambda: mp()["com"], "length", 3, Origin.CALCULATED, "com",
                              "Centre of mass in the object's local frame")
            for k, lab in enumerate(("Inertia (row X)", "Inertia (row Y)", "Inertia (row Z)")):
                self._add_readout(sec, lab, lambda k=k: mp()["inertia"][k], "inertia", 3, Origin.CALCULATED,
                                  "inertia", "Inertia tensor about the centre of mass, local axes", precision=6)
            self._add_readout(sec, "Principal moments", lambda: np.sort(np.linalg.eigvalsh(mp()["inertia"])),
                              "inertia", 3, Origin.CALCULATED, "inertia", precision=6, labels=("1", "2", "3"))
        else:
            self._add_prop(sec, "mass_input", [o], origin=Origin.USER)

        vec_combo = QComboBox()
        choices = self._frame_choices(o)
        choices.insert(1, ("Own local axes", "@self"))
        for text, data in choices:
            vec_combo.addItem(text, data)
        vec_combo.setCurrentIndex(max(vec_combo.findData(self._vec_frame), 0))
        vec_combo.setToolTip("Axes in which the vectors below are shown and entered. This only re-expresses the "
                             "same global vectors (a rotation); values are stored in global coordinates. For "
                             "motion measured relative to a moving frame use the Relative Motion panel.")

        def vec_frame_changed(_i):
            self._vec_frame = vec_combo.currentData()
            self._schedule()
        vec_combo.currentIndexChanged.connect(vec_frame_changed)
        sec.add_row("Vectors in", vec_combo, search="frame vector components representation")

        def vec_rotation():
            ref = self._resolve_ref(o, self._vec_frame)
            return rotation_of(self.scene.reference_pose(ref)[1])
        rot = None if self._vec_frame is None else vec_rotation
        frame_text = "global frame" if self._vec_frame is None else f"axes of {vec_combo.currentText()}"
        for path in ("linear_velocity", "linear_acceleration", "angular_velocity", "angular_acceleration", "force",
                     "torque"):
            self._add_prop(sec, path, [o], origin=Origin.USER, tooltip=f"{INITIAL_TIP} Components in the {frame_text}.",
                           frame_rotation=rot)

        def momentum():
            m = mp()["mass"] or 0.0
            return m * np.asarray(o.linear_velocity)

        def angular_momentum():
            _, q = self.scene.world_pose(o.id)
            r = np.asarray(numerics.quat_to_matrix(q))
            i_world = r @ mp()["inertia"] @ r.T
            return i_world @ np.asarray(o.angular_velocity)

        def kinetic_energy():
            m = mp()["mass"] or 0.0
            v = np.asarray(o.linear_velocity)
            w = np.asarray(o.angular_velocity)
            return 0.5 * m * float(v @ v) + 0.5 * float(w @ angular_momentum())

        if o.id in self.simulated_ids:
            def runtime(key):
                return lambda: self.scene.runtime.get(o.id, {}).get(key, np.zeros(3))
            self._add_readout(sec, "Acceleration (current)", runtime("acceleration"), "acceleration", 3,
                              Origin.SIMULATED, "runtime_acceleration", "Total acceleration of the COM computed by "
                              "the processes in the latest step")
            self._add_readout(sec, "Angular accel. (current)", runtime("angular_acceleration"),
                              "angular_acceleration", 3, Origin.SIMULATED, "runtime_angular_acceleration")
        self._add_readout(sec, "Linear momentum", momentum, "momentum", 3, Origin.DERIVED, "momentum", "p = m·v")
        self._add_readout(sec, "Angular momentum", angular_momentum, "angular_momentum", 3, Origin.DERIVED,
                          "angular_momentum", "L = I·ω about the centre of mass (global frame)")
        self._add_readout(sec, "Kinetic energy", kinetic_energy, "energy", 1, Origin.DERIVED, "kinetic_energy",
                          "½·m·v² + ½·ω·L")

    def _build_collision(self, objs: list[SceneObject]) -> None:
        """Collision geometry (separate from the visual mesh), contact material and joints."""
        from ..physics.colliders import collider_for
        sec = self._section("collision", "Collision & Joints")
        for key in ("collider", "radius", "length", "half_extents", "offset", "restitution", "friction",
                    "static_friction", "rolling_resistance", "show_collider"):
            self._add_prop(sec, f"physics.{key}", objs)
        primary = objs[-1]
        shape = QLabel()
        shape.setWordWrap(True)
        shape.setObjectName("Muted")

        def describe():
            try:
                c = collider_for(primary)
            except Exception as exc:   # geometry errors are reported, not raised from the inspector
                shape.setText(f"Collider unavailable: {exc}")
                return
            shape.setText("Does not collide (mass alone never makes an object collide)." if c is None
                          else f"Collides as: {c.describe()}")
        self._refreshers.append(describe)
        describe()
        sec.add_row("Effective collider", shape, search="collider shape effective approximate")
        if len(objs) != 1:
            return
        joints = [j for j in self.scene.joints if primary.id in (j.body_a, j.body_b)]
        for j in joints:
            other = j.body_b if j.body_a == primary.id else j.body_a
            other_name = self.scene.objects[other].name if other in self.scene.objects else (other or "world")
            text = f"{j.id} · {j.name} — {j.type} to {other_name}" + ("" if j.enabled else " (disabled)")
            lab = QLabel(text)
            lab.setWordWrap(True)
            sec.add_widget(lab, search=f"joint {j.name} {j.id} {j.type}")
        if not joints:
            note = QLabel("No joints.")
            note.setObjectName("Muted")
            sec.add_widget(note, search="joint")
        btn = QPushButton(icons.icon("edit"), "Joints…")
        btn.setToolTip("Create and edit joints of this object")
        btn.clicked.connect(lambda: self.jointsRequested.emit(primary.id))
        sec.add_buttons(btn)

    def _build_coordinates(self, o: SceneObject) -> None:
        sec = self._section("coordinates", "Coordinate Systems")
        parent = QComboBox()
        parent.addItem("— None (scene root) —", None)
        for other in self.scene.objects.values():
            if other.id != o.id and self.scene.can_reparent(o.id, other.id):
                parent.addItem(f"{other.name} ({other.id})", other.id)
        parent.setCurrentIndex(max(parent.findData(o.parent_id), 0))

        def reparent(i):
            try:
                self.editor.reparent([o.id], parent.itemData(i))
            except SceneError as exc:
                self.errorOccurred.emit(str(exc))
        parent.activated.connect(reparent)
        sec.add_row("Parent", parent, search="parent hierarchy assembly")

        def world():
            return self.scene.world_pose(o.id)
        self._add_readout(sec, "Global position", lambda: world()[0], "length", 3, Origin.DERIVED, "position")
        self._add_readout(sec, "Global orientation", lambda: numerics.quat_to_euler(world()[1]), "angle", 3,
                          Origin.DERIVED, "orientation_euler", "Euler XYZ angles in the global frame")
        self._add_readout(sec, "Local position", lambda: o.position, "length", 3, Origin.DERIVED, "position",
                          "Position relative to the parent frame (global frame for root objects)")
        self._add_readout(sec, "Local orientation", lambda: o.euler(), "angle", 3, Origin.DERIVED,
                          "orientation_euler", "Euler XYZ angles relative to the parent frame")
        if "local_axes" in o.typedef.overlays:
            self._add_prop(sec, "overlay.local_axes", [o], label="Show local axes")
            self._add_prop(sec, "local_axes_size", [o])

    def _build_appearance(self, objs: list[SceneObject]) -> None:
        sec = self._section("appearance", "Materials & Appearance")
        if not objs[-1].typedef.uses_vertex_colors:
            self._add_prop(sec, "color", objs)
        for path in ("opacity", "material", "metallic", "roughness", "wireframe", "show_surface", "show_edges"):
            self._add_prop(sec, path, objs)

    def _build_display(self, objs: list[SceneObject]) -> None:
        sec = self._section("display", "Labels & Display")
        supported = [k for k in objs[-1].typedef.overlays if all(k in o.typedef.overlays for o in objs)]
        if objs[-1].typedef.physical is False:
            supported = [k for k in supported if k not in ("mass", "com")]
        grid = QWidget()
        gl = QGridLayout(grid)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setHorizontalSpacing(10)
        gl.setVerticalSpacing(3)
        primary = objs[-1]
        for k, key in enumerate(supported):
            cb = QCheckBox(OVERLAY_LABELS.get(key, key))
            path = f"overlay.{key}"
            cb.toggled.connect(lambda v, p=path: self._commit(objs, p, v))
            self._refreshers.append(lambda cb=cb, p=path: _set_blocked(cb, cb.setChecked, primary.get(p)))
            self._refreshers[-1]()
            gl.addWidget(cb, k // 2, k % 2)
        sec.add_widget(grid, search="overlay labels name id position velocity mass axes bounding box com arrows")
        for path in ("label_color", "label_font_size", "label_precision", "label_background", "label_placement"):
            self._add_prop(sec, path, objs)
        if len(objs) == 1:
            self._label_units(sec, primary)
        for q in ARROW_QUANTITIES:
            if f"{q}_arrow" in supported:
                self._add_prop(sec, f"arrow_scale.{q}", objs, tooltip=f"Arrow length per SI unit of {q.replace('_', ' ')}"
                               " (e.g. 0.25 m per m/s)")

    def _label_units(self, sec: Section, o: SceneObject) -> None:
        """Per-object display units of the values in this object's label."""
        row = QWidget()
        lay = QGridLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(6)
        for k, q in enumerate(LABEL_UNIT_QUANTITIES):
            combo = QComboBox()
            combo.addItem(f"project ({self.units.preferred_unit(q)})", "")
            for sym in self.units.units_for(q):
                combo.addItem(sym, sym)
            combo.setCurrentIndex(max(combo.findData(o.label_units.get(q, "")), 0))
            combo.setToolTip(f"Unit of {q.replace('_', ' ')} in this object's label")
            combo.activated.connect(lambda _i, c=combo, q=q: self._commit([o], f"label_unit.{q}", c.currentData()))
            lay.addWidget(QLabel(q.replace("_", " ").capitalize()), k, 0)
            lay.addWidget(combo, k, 1)
        sec.add_row("Label units", row, search="label display units")

    def _build_trajectory(self, objs: list[SceneObject]) -> None:
        sec = self._section("trajectory", "Trajectory")
        for key in ("enabled", "color", "width", "opacity", "point", "history", "recent_duration", "interval",
                    "max_points", "arrows", "arrow_spacing", "time_labels", "label_spacing", "preserve"):
            self._add_prop(sec, f"trajectory.{key}", objs)
        o = objs[-1]
        if len(objs) == 1:
            combo = QComboBox()
            for text, ref in reference_choices(self.scene, exclude=o.id):
                combo.addItem(text, ref or "")
            current = o.trajectory_setting("frame")
            if current and combo.findData(current) < 0:
                combo.addItem(f"⚠ missing {current}", current)
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.setToolTip("Frame in which the path is recorded. A relative path is drawn attached to the frame's "
                             "current pose (what an observer riding the frame sees)")
            combo.activated.connect(lambda _i: self._commit([o], "trajectory.frame", combo.currentData()))
            sec.add_row("Recorded in frame", combo, search="trajectory frame relative")
        buttons = QWidget()
        bl = QHBoxLayout(buttons)
        bl.setContentsMargins(0, 0, 0, 0)
        clear = QPushButton("Clear")
        clear.setToolTip("Hide the path drawn so far (new samples keep being drawn)")
        clear.clicked.connect(lambda: self.clearTrajectoryRequested.emit([x.id for x in objs]))
        export = QPushButton("Export…")
        export.setToolTip("Save the recorded path (in the selected frame) as CSV")
        export.setEnabled(len(objs) == 1)
        export.clicked.connect(lambda: self.exportTrajectoryRequested.emit(o.id))
        bl.addWidget(clear)
        bl.addWidget(export)
        bl.addStretch(1)
        sec.add_widget(buttons, search="trajectory clear export")

    def _build_custom(self, objs: list[SceneObject]) -> None:
        sec = self._section("custom", "Custom Properties")
        common = [k for k in objs[-1].custom if all(k in o.custom for o in objs)]
        primary = objs[-1]
        if self._custom_sort == "name":
            common.sort(key=lambda k: primary.custom[k].definition.label.lower())
        elif self._custom_sort == "key":
            common.sort()
        for key in common:
            cp = primary.custom[key]
            d = cp.definition
            self._add_prop(sec, f"custom.{key}", objs, label=d.label, origin=Origin.USER,
                           tooltip=f"key: {key}\n{d.description}".strip())
        if len(objs) == 1 and common:
            edit_combo = QComboBox()
            edit_combo.addItem("Edit or remove…", None)
            for key in common:
                edit_combo.addItem(f"Edit '{primary.custom[key].definition.label}'", ("edit", key))
                edit_combo.addItem(f"Remove '{primary.custom[key].definition.label}'", ("remove", key))
            edit_combo.activated.connect(lambda i: self._custom_action(primary, edit_combo.itemData(i)))
            sort = QComboBox()
            sort.addItem("Definition order", "definition")
            sort.addItem("Sort by name", "name")
            sort.addItem("Sort by key", "key")
            sort.setCurrentIndex(max(sort.findData(self._custom_sort), 0))
            sort.activated.connect(lambda i: (setattr(self, "_custom_sort", sort.itemData(i)), self._schedule()))
            sec.add_buttons(edit_combo, sort)
        if not common:
            note = QLabel("No custom properties." if len(objs) == 1 else "No custom properties shared by all selected objects.")
            note.setObjectName("Muted")
            sec.add_widget(note)
        add = QPushButton(icons.icon("plus"), "Add Property…")
        add.clicked.connect(lambda: self.add_custom_property(objs))
        sec.add_buttons(add)

    def _build_processes(self, o: SceneObject) -> None:
        from ..processes.process_base import ProcessError
        from ..processes.process_registry import process_registry
        sec = self._section("processes", "Assigned Processes")
        assigned = self.scene.assignments_for(o.id)
        for a in assigned:
            try:
                name = process_registry().get(a.process_id).name
            except ProcessError:
                name = f"{a.process_id} (unavailable)"
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox(f"{a.uid} · {name}" + (f" — {a.label}" if a.label else ""))
            cb.setChecked(a.enabled)
            cb.setToolTip(f"Targets: {a.targets.describe()}")
            cb.toggled.connect(lambda on, uid=a.uid: self.editor.update_assignment(uid, enabled=on))
            rl.addWidget(cb, 1)
            edit = QToolButton()
            edit.setIcon(icons.icon("edit"))
            edit.setToolTip("Edit assignment")
            edit.clicked.connect(lambda _=False, uid=a.uid: self.editAssignmentRequested.emit(uid))
            rl.addWidget(edit)
            sec.add_widget(row, search=f"process {name} {a.uid}")
        if not assigned:
            note = QLabel("No processes. Assign a process to simulate this object.")
            note.setObjectName("Muted")
            note.setWordWrap(True)
            sec.add_widget(note, search="process")
        add = QPushButton(icons.icon("plus"), "Assign Process…")
        add.clicked.connect(lambda: self.assignProcessRequested.emit([o.id]))
        sec.add_buttons(add)

    # ---- custom property actions ----------------------------------------------------------------------
    def add_custom_property(self, objs: list[SceneObject] | None = None) -> None:
        objs = objs or self._objs()
        if not objs:
            return
        keys = set().union(*[set(o.custom) for o in objs])
        dlg = CustomPropertyDialog(self.units, keys, parent=self)
        if dlg.exec() and dlg.result_data:
            try:
                self.editor.add_custom_property([o.id for o in objs], dlg.result_data)
            except PropertyError as exc:
                QMessageBox.warning(self, "Custom property", str(exc))

    def _custom_action(self, o: SceneObject, data) -> None:
        if not data:
            return
        action, key = data
        if action == "remove":
            label = o.custom[key].definition.label
            if QMessageBox.question(self, "Remove property", f"Remove custom property '{label}' from {o.name}?") \
                    == QMessageBox.StandardButton.Yes:
                self.editor.remove_custom_property([o.id], key)
            else:
                self._schedule()
            return
        dlg = CustomPropertyDialog(self.units, set(o.custom), data=o.custom[key].to_dict(), parent=self)
        if dlg.exec() and dlg.result_data:
            try:
                self.editor.replace_custom_property(o.id, key, dlg.result_data)
            except PropertyError as exc:
                QMessageBox.warning(self, "Custom property", str(exc))
        self._signature = None
        self._schedule()


def _set_blocked(widget: QWidget, setter, value) -> None:
    widget.blockSignals(True)
    try:
        setter(value)
    finally:
        widget.blockSignals(False)
