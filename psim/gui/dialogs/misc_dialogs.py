"""Smaller dialogs: shortcuts, preferences, units, mass properties report, help and about."""
from __future__ import annotations

import platform

import numpy as np

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import Qt, qVersion
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QHeaderView, QKeySequenceEdit, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ... import __version__, numerics
from ...plugins import loaded_plugins, plugin_errors
from ...scene.analysis import combined_mass_properties
from ...scene.scene_manager import Scene
from ...units.unit_expressions import si_unit_text
from ...units.unit_registry import UnitError, UnitRegistry
from ..actions import ActionRegistry


class ShortcutsDialog(QDialog):
    def __init__(self, actions: ActionRegistry, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(620, 560)
        self.actions = actions
        lay = QVBoxLayout(self)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Search commands…")
        self.filter.textChanged.connect(self._filter)
        lay.addWidget(self.filter)
        items = sorted(((aid, a) for aid, a in actions.items()), key=lambda x: x[0])
        self.table = QTableWidget(len(items), 3)
        self.table.setHorizontalHeaderLabels(["Command", "Shortcut", "Default"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.edits: dict[str, QKeySequenceEdit] = {}
        for r, (aid, a) in enumerate(items):
            name = QTableWidgetItem(a.text().replace("&", "").rstrip("…."))
            name.setToolTip(aid)
            name.setData(Qt.ItemDataRole.UserRole, aid)
            self.table.setItem(r, 0, name)
            ed = QKeySequenceEdit(a.shortcut())
            ed.setClearButtonEnabled(True)
            ed.setMaximumSequenceLength(1)
            ed.keySequenceChanged.connect(self._check)
            self.table.setCellWidget(r, 1, ed)
            self.table.setItem(r, 2, QTableWidgetItem(actions.default_shortcut(aid)))
            self.edits[aid] = ed
        lay.addWidget(self.table)
        self.status = QLabel()
        self.status.setStyleSheet("color: #d9a33a;")
        lay.addWidget(self.status)
        row = QHBoxLayout()
        reset = QPushButton("Reset All to Defaults")
        reset.clicked.connect(self._reset)
        row.addWidget(reset)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        lay.addLayout(row)
        self._check()

    def _filter(self, text: str) -> None:
        text = text.lower()
        for r in range(self.table.rowCount()):
            self.table.setRowHidden(r, bool(text) and text not in self.table.item(r, 0).text().lower())

    def _conflicts(self) -> dict[str, list[str]]:
        seen: dict[str, list[str]] = {}
        for aid, ed in self.edits.items():
            s = ed.keySequence().toString()
            if s:
                seen.setdefault(s, []).append(aid)
        return {k: v for k, v in seen.items() if len(v) > 1}

    def _check(self) -> bool:
        c = self._conflicts()
        self.status.setText("; ".join(f"{k} is assigned to {', '.join(v)}" for k, v in c.items()))
        return not c

    def _reset(self) -> None:
        for aid, ed in self.edits.items():
            ed.setKeySequence(QKeySequence(self.actions.default_shortcut(aid)))

    def _accept(self) -> None:
        if not self._check():
            QMessageBox.warning(self, "Shortcut conflict", "Resolve the conflicting shortcuts before saving.")
            return
        for aid, ed in self.edits.items():
            self.actions.set_shortcut(aid, ed.keySequence().toString())
        self.accept()


class PreferencesDialog(QDialog):
    def __init__(self, theme: str, autosave_s: int, render_fps: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.theme = QComboBox()
        self.theme.addItem("Graphite (dark)", "dark")
        self.theme.addItem("Light", "light")
        self.theme.setCurrentIndex(max(self.theme.findData(theme), 0))
        form.addRow("Theme", self.theme)
        self.autosave = QSpinBox()
        self.autosave.setRange(0, 3600)
        self.autosave.setSuffix(" s")
        self.autosave.setSpecialValueText("Disabled")
        self.autosave.setValue(autosave_s)
        self.autosave.setToolTip("Interval for writing the crash-recovery snapshot while there are unsaved changes")
        form.addRow("Autosave interval", self.autosave)
        self.fps = QSpinBox()
        self.fps.setRange(1, 480)
        self.fps.setSuffix(" FPS")
        self.fps.setValue(int(render_fps))
        self.fps.setToolTip("Redraw rate used when continuous rendering is enabled (View menu). "
                            "Rendering rate never affects simulation time steps.")
        form.addRow("Continuous render rate", self.fps)
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)


class UnitsDialog(QDialog):
    """Project display units and custom unit definitions."""

    def __init__(self, units: UnitRegistry, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Units")
        self.resize(560, 520)
        self.units = units
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        lay.addWidget(tabs)

        # project units
        page = QWidget()
        pl = QVBoxLayout(page)
        note = QLabel("Display units used throughout the application. Values are always stored in SI units; "
                      "changing a display unit never changes the underlying quantity.")
        note.setWordWrap(True)
        note.setObjectName("Muted")
        pl.addWidget(note)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Quantity", "Dimension", "SI base units", "Display unit"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        pl.addWidget(self.table)
        reset = QPushButton("Reset to SI")
        reset.setAutoDefault(False)
        reset.clicked.connect(self._reset)
        pl.addWidget(reset, 0, Qt.AlignmentFlag.AlignLeft)
        tabs.addTab(page, "Project Units")

        # custom units
        page2 = QWidget()
        cl = QVBoxLayout(page2)
        self.custom = QTableWidget(0, 4)
        self.custom.setHorizontalHeaderLabels(["Symbol", "Name", "Dimension", "SI factor"])
        self.custom.verticalHeader().setVisible(False)
        self.custom.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        cl.addWidget(self.custom)
        form = QFormLayout()
        self.sym = QLineEdit()
        self.sym.setPlaceholderText("e.g. furlong")
        self.name = QLineEdit()
        self.base = QComboBox()
        self.base.setEditable(True)
        self.factor = QLineEdit()
        self.factor.setPlaceholderText("e.g. 201.168")
        self.base.lineEdit().setPlaceholderText("a unit or an expression, e.g. m, kN·mm, furlong/fortnight")
        self.preview = QLabel()
        self.preview.setObjectName("Muted")
        self.preview.setWordWrap(True)
        form.addRow("Symbol", self.sym)
        form.addRow("Name", self.name)
        form.addRow("1 symbol =", self.factor)
        form.addRow("× unit or expression", self.base)
        form.addRow("", self.preview)
        self.base.editTextChanged.connect(self._preview)
        self.factor.textChanged.connect(self._preview)
        cl.addLayout(form)
        row = QHBoxLayout()
        add = QPushButton("Add Unit")
        add.setAutoDefault(False)
        add.clicked.connect(self._add)
        rem = QPushButton("Remove Selected")
        rem.clicked.connect(self._remove)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        cl.addLayout(row)
        self.error = QLabel()
        self.error.setStyleSheet("color: #e5534b;")
        cl.addWidget(self.error)
        tabs.addTab(page2, "Custom Units")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        lay.addWidget(buttons)
        self._populate()

    def _populate(self) -> None:
        qs = self.units.quantities()
        self.table.setRowCount(len(qs))
        for r, q in enumerate(qs):
            self.table.setItem(r, 0, QTableWidgetItem(q.label))
            self.table.setItem(r, 1, QTableWidgetItem(str(q.dimension)))
            self.table.setItem(r, 2, QTableWidgetItem(si_unit_text(q.dimension)))
            combo = QComboBox()
            combo.addItems(q.units)
            combo.setCurrentText(self.units.preferred_unit(q.key))
            combo.currentTextChanged.connect(lambda sym, k=q.key: self.units.set_preferred_unit(k, sym))
            self.table.setCellWidget(r, 3, combo)
        cu = self.units.custom_units()
        self.custom.setRowCount(len(cu))
        for r, u in enumerate(cu):
            for c, v in enumerate((u.symbol, u.name, str(u.dimension), f"{u.factor:.10g}")):
                self.custom.setItem(r, c, QTableWidgetItem(v))
        current = self.base.currentText()
        self.base.clear()
        for q in qs:
            for sym in q.units:
                if self.base.findText(sym) < 0:
                    self.base.addItem(sym)
        self.base.setCurrentText(current or "m")

    def _preview(self, *_a) -> None:
        text = self.base.currentText().strip()
        if not text:
            self.preview.setText("")
            return
        try:
            f, dim = self.units.parse_expression(text)
        except UnitError as exc:
            self.preview.setText(str(exc))
            return
        try:
            factor = float(self.factor.text()) if self.factor.text().strip() else 1.0
        except ValueError:
            factor = float("nan")
        quantities = [q.label for q in self.units.quantities_for_dimension(dim)]
        what = ", ".join(quantities) if quantities else "no known quantity — cannot be used"
        self.preview.setText(f"{self.units.describe_dimension(dim) if quantities else f'[{dim}]'} → {what}. "
                             f"1 unit = {factor * f:.10g} {si_unit_text(dim)}")

    def _reset(self) -> None:
        self.units.reset_preferences()
        self._populate()

    def _add(self) -> None:
        try:
            factor = float(self.factor.text())
            self.units.add_custom_unit(self.sym.text(), self.name.text(), self.base.currentText(), factor)
        except (ValueError, UnitError) as exc:
            self.error.setText(str(exc))
            return
        self.error.setText("")
        self.sym.clear()
        self.name.clear()
        self.factor.clear()
        self._populate()

    def _remove(self) -> None:
        rows = sorted({i.row() for i in self.custom.selectedIndexes()})
        for r in rows:
            try:
                self.units.remove_custom_unit(self.custom.item(r, 0).text())
            except UnitError as exc:
                self.error.setText(str(exc))
        self._populate()


class MassReportDialog(QDialog):
    def __init__(self, scene: Scene, ids: list[str], units: UnitRegistry, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mass Properties Report")
        self.resize(760, 460)
        res = combined_mass_properties(scene, ids)

        def clean(values, scale: float = 1.0):
            # suppress floating-point noise (e.g. -1.8e-17) relative to the magnitude being reported
            values = np.asarray(values, float)
            return np.where(np.abs(values) < 1e-10 * max(scale, 1e-300), 0.0, values)
        mu, lu, iu = units.preferred_unit("mass"), units.preferred_unit("length"), units.preferred_unit("inertia")
        lay = QVBoxLayout(self)
        table = QTableWidget(len(res["rows"]), 6)
        table.setHorizontalHeaderLabels(["ID", "Name", f"Mass [{mu}]", f"COM X [{lu}]", f"COM Y [{lu}]", f"COM Z [{lu}]"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        extent = max([float(np.abs(c).max()) for _o, _m, c, _i, _mp in res["rows"]] + [1.0])
        for r, (o, m, c, _i, _mp) in enumerate(res["rows"]):
            cc = units.from_si(clean(c, extent), lu)
            for col, v in enumerate((o.id, o.name, f"{units.from_si(m, mu):.6g}", f"{cc[0]:.5g}", f"{cc[1]:.5g}",
                                     f"{cc[2]:.5g}")):
                table.setItem(r, col, QTableWidgetItem(v))
        lay.addWidget(table)
        com = units.from_si(clean(res["com"], extent), lu)
        it = units.from_si(clean(res["inertia"], float(np.abs(res["inertia"]).max(initial=0.0))), iu)
        text = QLabel(
            f"<b>Total mass:</b> {units.from_si(res['mass'], mu):.6g} {mu}<br>"
            f"<b>Combined centre of mass:</b> ({com[0]:.5g}, {com[1]:.5g}, {com[2]:.5g}) {lu}<br>"
            f"<b>Inertia about combined COM (global axes) [{iu}]:</b><br>"
            + "<br>".join("&nbsp;&nbsp;[" + ", ".join(f"{v: .6g}" for v in row) + "]" for row in it)
            + "<br><span style='color:gray'>Computed from the faceted meshes; objects without mass are excluded.</span>")
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        lay.addWidget(buttons)


CONTROLS_HTML = """
<h3>Viewport</h3>
<table cellspacing="4">
<tr><td><b>Left click</b></td><td>Select object (Ctrl: toggle, Shift: add)</td></tr>
<tr><td><b>Left drag on empty space</b></td><td>Select tool (Q): box selection (Shift adds, Ctrl toggles);
other tools: orbit camera. Alt + left drag always orbits</td></tr>
<tr><td><b>Middle / right drag, Shift + left drag</b></td><td>Pan camera</td></tr>
<tr><td><b>Mouse wheel</b></td><td>Zoom towards the cursor</td></tr>
<tr><td><b>Double click</b></td><td>Frame the clicked object</td></tr>
<tr><td><b>Right click</b></td><td>Context menu</td></tr>
<tr><td><b>Drag gizmo handles</b></td><td>Move along an axis or plane, rotate about an axis, scale along a local axis.
 The centre handle moves in the view plane / scales uniformly. Esc cancels a drag.</td></tr>
<tr><td><b>Measuring between points</b></td><td>After Tools ▸ Measure … Between Points: left click picks a point on an
 object's surface (snapped to a nearby corner or edge; Ctrl: no snapping), left drag still orbits, Esc cancels</td></tr>
<tr><td><b>Drag from Object Library</b></td><td>Place the object on the surface under the cursor or on the work plane
 (with grid snapping)</td></tr>
</table>
<h3>Default shortcuts</h3>
<p>Q select · W move · E rotate · R scale · T toggle global/local gizmo · F frame selected · Home fit scene ·
1/3/7 front/right/top (Ctrl: opposite) · 0 isometric · 5 perspective/orthographic · G toggle XY grid ·
Ctrl+D duplicate · Ctrl+G group · Delete delete · F11 full-screen viewport.
All shortcuts can be changed in Tools ▸ Keyboard Shortcuts.</p>
"""


class HelpDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Controls & Shortcuts")
        self.resize(620, 520)
        lay = QVBoxLayout(self)
        b = QTextBrowser()
        b.setHtml(CONTROLS_HTML)
        lay.addWidget(b)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        lay.addWidget(buttons)


def about_html(gl_info: dict) -> str:
    plugins = ", ".join(loaded_plugins) or "none"
    errs = "<br>".join(plugin_errors)
    return (f"<h2>PSim Studio {__version__}</h2>"
            "<p>Professional 3D physics simulation studio — Phase 1: scene building, object library, "
            "property system and 3D viewport.</p>"
            f"<p><b>Numerical core:</b> {numerics.backend_description()}<br>"
            f"<b>OpenGL:</b> {gl_info.get('renderer', '?')} — {gl_info.get('version', '?')}<br>"
            f"<b>Qt:</b> {qVersion()} (PySide6 {pyside_version})<br>"
            f"<b>Python:</b> {platform.python_version()} on {platform.system()} {platform.release()}<br>"
            f"<b>Plugins:</b> {plugins}</p>" + (f"<p style='color:#e5534b'>{errs}</p>" if errs else ""))
