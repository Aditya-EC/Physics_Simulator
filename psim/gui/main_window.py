"""Main application window: menus, toolbar, docks, project lifecycle."""
from __future__ import annotations

import json
import os

import numpy as np
from PySide6.QtCore import QByteArray, QMimeData, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QGuiApplication, QUndoStack
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QFrame, QInputDialog, QLabel,
                               QMainWindow, QMenu, QMessageBox, QTabWidget, QToolButton, QUndoView, QVBoxLayout,
                               QWidget)

from .. import APP_NAME, __version__, numerics
from ..project import serialization as ser
from ..project.autosave import Autosaver
from ..project.templates import PROJECT_TEMPLATES
from ..properties.property_system import PropertyError
from ..rendering import picking
from ..rendering.view_settings import CameraState, ViewSettings
from ..rendering.viewport import Viewport
from ..scene import geometry as geo
from ..scene.builtin_types import mesh_assets
from ..scene.editing import SceneEditor
from ..scene.object_registry import CATEGORIES
from ..scene.scene_manager import Scene, SceneError
from ..units.unit_registry import default_registry
from . import icons, theme
from .actions import ActionRegistry
from .bottom_panels import ConsolePanel, LogBus, ObjectTablePanel, PerformancePanel, ProblemsPanel, install_logging
from .dialogs.misc_dialogs import (HelpDialog, MassReportDialog, PreferencesDialog, ShortcutsDialog, UnitsDialog,
                                   about_html)
from .dialogs.viewport_settings_dialog import ViewportSettingsDialog
from .library_items import Library
from .object_library import ObjectLibraryPanel
from .outliner import Outliner
from .properties_panel import PropertiesPanel

CLIP_MIME = "application/x-psim-objects"
FILE_FILTER = "PSim projects (*.psim);;JSON files (*.json);;All files (*)"
MAX_RECENT = 10


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings, check_recovery: bool = True) -> None:
        super().__init__()
        self.settings = settings
        self.setWindowTitle(APP_NAME)
        self.setDockNestingEnabled(True)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowTabbedDocks |
                            QMainWindow.DockOption.AllowNestedDocks)
        self.units = default_registry()
        self.scene = Scene(self)
        self.undo = QUndoStack(self)
        self.editor = SceneEditor(self.scene, self.undo)
        self.project = ser.ProjectData()
        self.project_path: str | None = None
        self._extra_dirty = False
        self._fullscreen_state: QByteArray | None = None
        self.bus = LogBus(self)
        install_logging(self.bus)
        self.library = Library()

        self._build_central()
        self._build_docks()
        self.actions_ = ActionRegistry(self)
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        self.actions_.load_overrides(settings)
        self._connect()
        self._default_state = self.saveState()
        self._restore_window()
        self.autosaver = Autosaver(self._snapshot, self.is_dirty, int(settings.value("autosave_s", 120)), self)
        self._update_title()
        self._update_actions()
        self.bus.info(f"{APP_NAME} {__version__} started · {numerics.backend_description()}")
        for msg in self.library.errors:
            self.bus.warning(msg, "library")
        if numerics.BACKEND != "native":
            self.bus.warning("Native C++ numerical core not loaded; using the NumPy fallback. "
                             "Run 'python build_native.py' to build it.", "numerics")
        from ..plugins import plugin_errors
        for msg in plugin_errors:
            self.bus.error(msg, "plugins")
        if check_recovery:
            QTimer.singleShot(0, self._offer_recovery)

    # ======================================================================================
    # construction
    # ======================================================================================
    def _build_central(self) -> None:
        self.viewport = Viewport(self.scene, self.editor, self.project.view, self.units)
        self.viewport.ghost_provider = lambda payload: self.library.preview_mesh(payload.get("key", ""))
        frame = QFrame()
        frame.setObjectName("ViewportFrame")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.viewport)
        self.setCentralWidget(frame)
        self.viewport.set_continuous(False, float(self.settings.value("render_fps", 60)))

    def _dock(self, title: str, name: str, widget: QWidget, area) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(name)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable |
                      QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, d)
        return d

    def _build_docks(self) -> None:
        L, R, B = Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea, \
            Qt.DockWidgetArea.BottomDockWidgetArea
        self.library_panel = ObjectLibraryPanel(self.library)
        self.dock_library = self._dock("Object Library", "dock_library", self.library_panel, L)
        self.outliner = Outliner(self.scene, self.editor)
        self.dock_outliner = self._dock("Scene Hierarchy", "dock_outliner", self.outliner, L)
        self.history = QUndoView(self.undo)
        self.history.setEmptyLabel("<initial state>")
        self.dock_history = self._dock("History", "dock_history", self.history, L)
        self.tabifyDockWidget(self.dock_outliner, self.dock_history)
        self.dock_outliner.raise_()
        self.splitDockWidget(self.dock_library, self.dock_outliner, Qt.Orientation.Vertical)
        self.inspector = PropertiesPanel(self.scene, self.editor, self.units)
        self.dock_inspector = self._dock("Properties", "dock_properties", self.inspector, R)

        self.console = ConsolePanel(self.bus)
        self.problems = ProblemsPanel(self.bus)
        self.object_table = ObjectTablePanel(self.scene, self.units)
        self.performance = PerformancePanel()
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setDocumentMode(True)
        self.bottom_tabs.addTab(self.console, icons.icon("terminal"), "Console")
        self.bottom_tabs.addTab(self.problems, icons.icon("warning"), "Errors && Warnings")
        self.bottom_tabs.addTab(self.object_table, icons.icon("table"), "Object Table")
        self.bottom_tabs.addTab(self.performance, icons.icon("activity"), "Performance")
        self.dock_bottom = self._dock("Panels", "dock_bottom", self.bottom_tabs, B)
        self.resizeDocks([self.dock_library, self.dock_inspector], [300, 340], Qt.Orientation.Horizontal)
        self.resizeDocks([self.dock_bottom], [190], Qt.Orientation.Vertical)
        self.resizeDocks([self.dock_library, self.dock_outliner], [520, 260], Qt.Orientation.Vertical)

    def _build_actions(self) -> None:
        A = self.actions_.add
        # file
        A("file.new", "&New Project", self.new_project, icon="new", shortcut="Ctrl+N")
        A("file.open", "&Open…", self.open_project_dialog, icon="open", shortcut="Ctrl+O")
        A("file.save", "&Save", self.save, icon="save", shortcut="Ctrl+S")
        A("file.save_as", "Save &As…", self.save_as, icon="save_as", shortcut="Ctrl+Shift+S")
        A("file.import_obj", "&Import Mesh (OBJ)…", self.import_obj, icon="import", shortcut="Ctrl+I")
        A("file.export_obj", "&Export Selected Mesh (OBJ)…", self.export_obj, icon="export")
        A("file.export_table", "Export Object &Table (CSV)…", lambda: self.object_table.export_csv(), icon="table")
        A("file.recover", "Recover &Autosave…", lambda: self._offer_recovery(explicit=True), icon="history")
        A("file.exit", "E&xit", self.close, shortcut="Ctrl+Q")
        # edit
        self.act_undo = self.actions_.register("edit.undo", self.undo.createUndoAction(self, "&Undo"), "Ctrl+Z", "undo")
        self.act_redo = self.actions_.register("edit.redo", self.undo.createRedoAction(self, "&Redo"), "Ctrl+Y", "redo")
        A("edit.cut", "Cu&t", self.cut, icon="cut", shortcut="Ctrl+X")
        A("edit.copy", "&Copy", self.copy, icon="copy", shortcut="Ctrl+C")
        A("edit.paste", "&Paste", self.paste, icon="paste", shortcut="Ctrl+V")
        A("edit.duplicate", "&Duplicate", self.duplicate, icon="duplicate", shortcut="Ctrl+D")
        A("edit.delete", "De&lete", self.delete_selection, icon="trash", shortcut="Del")
        A("edit.rename", "Re&name…", self.rename, icon="edit", shortcut="F2")
        A("edit.select_all", "Select &All", lambda: self.scene.set_selection(list(self.scene.objects)),
          shortcut="Ctrl+A")
        A("edit.deselect", "Deselect All", lambda: self.scene.set_selection([]), shortcut="Ctrl+Shift+A")
        A("edit.invert", "&Invert Selection", self.invert_selection, shortcut="Ctrl+Shift+I")
        # tools / transform
        self.tool_group = QActionGroup(self)
        for tool, text, ic, key in (("select", "&Select", "select", "Q"), ("translate", "&Move", "move", "W"),
                                    ("rotate", "&Rotate", "rotate", "E"), ("scale", "Sc&ale", "scale", "R")):
            a = A(f"tool.{tool}", text, lambda on, t=tool: on and self.viewport.set_tool(t), icon=ic, shortcut=key,
                  checkable=True, checked=tool == self.viewport.tool)
            self.tool_group.addAction(a)
        A("tool.global", "&Global Gizmo Orientation", self._set_gizmo_global, icon="globe", shortcut="T",
          checkable=True, checked=True, tip="Toggle gizmo orientation between global and object-local axes")
        # view
        A("view.ortho", "&Orthographic Projection", self.viewport.set_orthographic, icon="ortho", shortcut="5",
          checkable=True)
        for name, text, key in (("front", "&Front", "1"), ("back", "&Back", "Ctrl+1"), ("right", "&Right", "3"),
                                ("left", "&Left", "Ctrl+3"), ("top", "&Top", "7"), ("bottom", "Botto&m", "Ctrl+7"),
                                ("isometric", "&Isometric", "0")):
            A(f"view.{name}", text, lambda _=False, n=name: self.viewport.set_standard_view(n), shortcut=key)
        A("view.frame_selected", "&Frame Selected", lambda: self._frame_selection(), icon="frame", shortcut="F")
        A("view.fit_all", "Fit &Entire Scene", lambda: self.viewport.frame_objects(None), icon="fit", shortcut="Home")
        A("view.focus_origin", "Focus on &Origin", self.viewport.focus_origin, icon="origin", shortcut="Shift+C")
        self.follow_group = QActionGroup(self)
        for mode, text in (("none", "Free Camera"), ("look_at", "Look at Selected"),
                           ("follow", "Follow Selected (fixed offset)"), ("track_frame", "Track Selected Local Frame")):
            a = A(f"camera.{mode}", text, lambda on, m=mode: on and self._set_follow(m), checkable=True,
                  checked=mode == "none")
            self.follow_group.addAction(a)
        A("camera.add_bookmark", "&Add Camera Bookmark…", self.add_bookmark, icon="bookmark", shortcut="Ctrl+B")
        A("view.labels", "Object &Labels", lambda v: self._env("show_labels", v), icon="tag", checkable=True,
          checked=True)
        A("view.local_axes", "Show &All Local Axes", lambda v: self._env("show_local_axes", v), icon="frame_axes",
          checkable=True)
        A("view.shadows", "&Shadows", lambda v: self._env("shadows", v), icon="sun", checkable=True, checked=True)
        A("view.ground_shadow", "&Ground Shadow", lambda v: self._env("ground_shadow", v), checkable=True, checked=True)
        A("view.stats", "Viewport Status &Overlay", lambda v: self._env("show_stats", v), checkable=True, checked=True)
        A("view.continuous", "&Continuous Rendering", lambda v: self.viewport.set_continuous(v), icon="activity",
          checkable=True, tip="Redraw at the configured render rate instead of only when something changes")
        A("view.fullscreen", "Full-Screen &Viewport", self.toggle_fullscreen, icon="fullscreen", shortcut="F11")
        A("view.settings", "Viewport &Settings…", lambda: self.open_viewport_settings("environment"), icon="settings")
        # scene
        for key in ("XY", "YZ", "XZ"):
            A(f"grid.{key}", f"{key} Grid", lambda v, k=key: self._set_grid(k, v), icon="grid",
              shortcut="G" if key == "XY" else "", checkable=True, checked=self.project.view.grids[key].visible)
        A("scene.axes", "Global &Coordinate Axes", self._set_axes, icon="axes", checkable=True, checked=True)
        A("scene.grid_settings", "&Grid && Axes Settings…", lambda: self.open_viewport_settings("grids"), icon="grid")
        self.plane_group = QActionGroup(self)
        for key in ("XY", "YZ", "XZ"):
            a = A(f"plane.{key}", f"{key} Plane", lambda on, k=key: on and self._set_work_plane(k), checkable=True,
                  checked=key == "XY")
            self.plane_group.addAction(a)
        A("snap.grid", "&Grid Snapping", lambda v: self._snap("grid_snap", v), icon="magnet", shortcut="Shift+S",
          checkable=True, checked=True)
        A("snap.rotate", "&Rotation Snapping", lambda v: self._snap("rotate_snap", v), checkable=True)
        A("snap.scale", "S&cale Snapping", lambda v: self._snap("scale_snap", v), checkable=True)
        A("snap.surface", "&Surface Snapping", lambda v: self._snap("surface_snap", v), checkable=True, checked=True)
        A("snap.settings", "Snap &Settings…", lambda: self.open_viewport_settings("snap"), icon="settings")
        A("scene.group", "&Group", self.group, icon="group", shortcut="Ctrl+G")
        A("scene.ungroup", "&Ungroup", self.ungroup, icon="ungroup", shortcut="Ctrl+Shift+G")
        A("scene.unparent", "Move to Scene &Root", lambda: self._reparent(self.scene.selection, None))
        A("scene.stats", "Scene &Statistics…", self.scene_statistics, icon="info")
        # objects
        A("obj.lock", "&Lock / Unlock", self.toggle_lock, icon="lock", shortcut="Ctrl+L")
        A("obj.hide", "&Hide / Show", self.toggle_hide, icon="eye_off", shortcut="H")
        A("obj.unhide_all", "Show &All Objects", self.unhide_all, icon="eye", shortcut="Alt+H")
        A("obj.add_property", "Add &Custom Property…", lambda: self.inspector.add_custom_property(), icon="plus")
        A("obj.save_template", "Save as &Template…", self.save_template, icon="bookmark")
        A("obj.mass_report", "&Mass Properties Report…", self.mass_report, icon="mass")
        for axis in range(3):
            for mode in ("min", "center", "max"):
                A(f"align.{'xyz'[axis]}.{mode}", f"Align {'XYZ'[axis]} {mode.title()}",
                  lambda _=False, a=axis, m=mode: self._align(a, m))
            A(f"distribute.{'xyz'[axis]}", f"Distribute Along {'XYZ'[axis]}",
              lambda _=False, a=axis: self._distribute(a))
        # units / tools / window / help
        A("units.dialog", "&Units && Custom Units…", self.open_units, icon="ruler")
        A("units.si", "Reset Display Units to &SI", self.reset_units)
        A("tools.shortcuts", "&Keyboard Shortcuts…", self.open_shortcuts, icon="keyboard")
        A("tools.preferences", "&Preferences…", self.open_preferences, icon="settings", shortcut="Ctrl+,")
        A("window.save_layout", "&Save Layout As…", self.save_layout)
        A("window.reset_layout", "&Reset Layout", self.reset_layout)
        A("help.controls", "&Controls && Shortcuts", lambda: HelpDialog(self).exec(), icon="keyboard",
          shortcut="F1")
        A("help.about", "&About PSim Studio", self.about, icon="info")
        A("help.about_qt", "About &Qt", lambda: QMessageBox.aboutQt(self))

    def _build_menus(self) -> None:
        a = self.actions_
        mb = self.menuBar()

        m = mb.addMenu("&File")
        for k in ("file.new",):
            m.addAction(a[k])
        tm = m.addMenu(icons.icon("layers"), "New from &Template")
        for key, (label, _fn) in PROJECT_TEMPLATES.items():
            tm.addAction(label, lambda k=key: self.new_project(template=k))
        m.addAction(a["file.open"])
        self.recent_menu = m.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        m.addSeparator()
        m.addActions([a["file.save"], a["file.save_as"]])
        m.addSeparator()
        m.addActions([a["file.import_obj"], a["file.export_obj"], a["file.export_table"]])
        m.addSeparator()
        m.addAction(a["file.recover"])
        m.addSeparator()
        m.addAction(a["file.exit"])

        m = mb.addMenu("&Edit")
        m.addActions([self.act_undo, self.act_redo])
        m.addSeparator()
        m.addActions([a["edit.cut"], a["edit.copy"], a["edit.paste"], a["edit.duplicate"], a["edit.delete"],
                      a["edit.rename"]])
        m.addSeparator()
        m.addActions([a["edit.select_all"], a["edit.deselect"], a["edit.invert"]])

        m = mb.addMenu("&View")
        m.addAction(a["view.ortho"])
        vm = m.addMenu("Standard &Views")
        vm.addActions([a[f"view.{n}"] for n in ("front", "back", "right", "left", "top", "bottom", "isometric")])
        m.addActions([a["view.frame_selected"], a["view.fit_all"], a["view.focus_origin"]])
        m.addSeparator()
        cm = m.addMenu(icons.icon("camera"), "&Camera Mode")
        cm.addActions(self.follow_group.actions())
        self.bookmark_menu = m.addMenu(icons.icon("bookmark"), "Camera &Bookmarks")
        self.bookmark_menu.aboutToShow.connect(self._fill_bookmarks)
        m.addSeparator()
        m.addActions([a["view.labels"], a["view.local_axes"], a["view.shadows"], a["view.ground_shadow"],
                      a["view.stats"]])
        m.addSeparator()
        m.addActions([a["view.continuous"], a["view.fullscreen"], a["view.settings"]])

        m = mb.addMenu("&Scene")
        gm = m.addMenu(icons.icon("grid"), "&Grids")
        gm.addActions([a["grid.XY"], a["grid.YZ"], a["grid.XZ"]])
        m.addActions([a["scene.axes"], a["scene.grid_settings"]])
        m.addSeparator()
        pm = m.addMenu("Placement &Plane")
        pm.addActions(self.plane_group.actions())
        sm = m.addMenu(icons.icon("magnet"), "S&napping")
        sm.addActions([a["snap.grid"], a["snap.rotate"], a["snap.scale"], a["snap.surface"]])
        sm.addSeparator()
        sm.addAction(a["snap.settings"])
        m.addSeparator()
        m.addActions([a["scene.group"], a["scene.ungroup"], a["scene.unparent"]])
        m.addSeparator()
        m.addAction(a["scene.stats"])

        m = mb.addMenu("&Objects")
        self.add_menu = m.addMenu(icons.icon("add"), "&Add Object")
        self._fill_add_menu(self.add_menu)
        m.addSeparator()
        m.addActions([a["tool.select"], a["tool.translate"], a["tool.rotate"], a["tool.scale"], a["tool.global"]])
        m.addSeparator()
        al = m.addMenu(icons.icon("align"), "A&lign")
        for axis in "xyz":
            for mode in ("min", "center", "max"):
                al.addAction(a[f"align.{axis}.{mode}"])
            al.addSeparator()
        dm = m.addMenu("&Distribute")
        dm.addActions([a[f"distribute.{x}"] for x in "xyz"])
        m.addSeparator()
        m.addActions([a["obj.lock"], a["obj.hide"], a["obj.unhide_all"]])
        m.addSeparator()
        m.addActions([a["obj.add_property"], a["obj.save_template"]])

        m = mb.addMenu("&Physics")
        m.addAction(a["obj.mass_report"])
        m.addSeparator()
        self._planned(m, "Rigid-body dynamics, collisions and joints — Phase 7")

        m = mb.addMenu("P&rocesses")
        self._planned(m, "Formula-based process engine and authoring — Phase 4")

        m = mb.addMenu("&Coordinates")
        m.addAction(a["tool.global"])
        m.addAction(a["view.local_axes"])
        m.addAction(a["scene.axes"])
        m.addSeparator()
        self._planned(m, "Named reference frames and frame-relative kinematics — Phase 5")

        m = mb.addMenu("Pl&ots")
        self._planned(m, "2D/3D scientific plotting workspace — Phase 6")

        m = mb.addMenu("&Units")
        m.addActions([a["units.dialog"], a["units.si"]])

        m = mb.addMenu("Si&mulation")
        self._planned(m, "Simulation engine (fixed Δt, update-rate scheduler) — Phase 3")

        m = mb.addMenu("&Tools")
        m.addActions([a["tools.shortcuts"], a["tools.preferences"], a["view.settings"]])

        m = mb.addMenu("&Window")
        for d in (self.dock_library, self.dock_outliner, self.dock_history, self.dock_inspector, self.dock_bottom):
            m.addAction(d.toggleViewAction())
        m.addSeparator()
        self.layout_menu = m.addMenu("&Layouts")
        self.layout_menu.aboutToShow.connect(self._fill_layouts)
        m.addActions([a["window.save_layout"], a["window.reset_layout"]])
        m.addSeparator()
        th = m.addMenu("&Theme")
        self.theme_group = QActionGroup(self)
        for key, label in (("dark", "Graphite (Dark)"), ("light", "Light")):
            act = th.addAction(label, lambda k=key: self.set_theme(k))
            act.setCheckable(True)
            act.setChecked(theme.current().name == key)
            act.setData(key)
            self.theme_group.addAction(act)

        m = mb.addMenu("&Help")
        m.addActions([a["help.controls"], a["help.about"], a["help.about_qt"]])

    def _planned(self, menu: QMenu, text: str) -> None:
        act = menu.addAction(icons.icon("info"), f"Planned: {text}")
        act.setEnabled(False)

    def _fill_add_menu(self, menu: QMenu) -> None:
        menu.clear()
        for cat in CATEGORIES:
            items = [it for it in self.library.items() if it.category == cat]
            if not items:
                continue
            sub = menu.addMenu(cat)
            for it in items:
                sub.addAction(self.library_panel.thumbnail(it.key), it.name, lambda k=it.key: self.add_library_item(k))
        menu.addSeparator()
        menu.addAction(self.actions_["file.import_obj"])

    def _build_toolbar(self) -> None:
        a = self.actions_
        tb = self.addToolBar("Main Toolbar")
        tb.setObjectName("toolbar_main")
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(True)
        tb.addActions([a["file.new"], a["file.open"], a["file.save"], a["file.save_as"]])
        tb.addSeparator()
        tb.addActions([self.act_undo, self.act_redo])
        tb.addSeparator()
        add_btn = QToolButton()
        add_btn.setIcon(icons.icon("add"))
        add_btn.setToolTip("Add Object")
        add_btn.setMenu(self.add_menu)
        add_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tb.addWidget(add_btn)
        tb.addSeparator()
        tb.addActions([a["tool.select"], a["tool.translate"], a["tool.rotate"], a["tool.scale"], a["tool.global"]])
        tb.addSeparator()
        tb.addActions([a["scene.axes"], a["grid.XY"], a["snap.grid"]])
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["XY", "YZ", "XZ"])
        self.plane_combo.setToolTip("Placement (work) plane for drag-and-drop")
        self.plane_combo.currentTextChanged.connect(self._set_work_plane)
        tb.addWidget(self.plane_combo)
        tb.addSeparator()
        props = self.dock_inspector.toggleViewAction()
        props.setIcon(icons.icon("properties"))
        props.setToolTip("Show or hide the Properties inspector")
        tb.addAction(props)
        tb.addSeparator()
        tb.addActions([a["view.ortho"], a["view.frame_selected"], a["view.fit_all"], a["view.focus_origin"]])
        views_btn = QToolButton()
        views_btn.setIcon(icons.icon("camera"))
        views_btn.setToolTip("Standard views and camera modes")
        vm = QMenu(views_btn)
        vm.addActions([a[f"view.{n}"] for n in ("front", "back", "right", "left", "top", "bottom", "isometric")])
        vm.addSeparator()
        vm.addActions(self.follow_group.actions())
        views_btn.setMenu(vm)
        views_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tb.addWidget(views_btn)
        tb.addAction(a["view.fullscreen"])
        self.toolbar = tb

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.sb_selection = QLabel("")
        self.sb_frame = QLabel("")
        backend = QLabel("C++ core" if numerics.BACKEND == "native" else "NumPy fallback")
        backend.setToolTip(numerics.backend_description())
        if numerics.BACKEND != "native":
            backend.setStyleSheet(f"color: {theme.current().warning};")
        self.sb_units = QLabel("")
        for w in (self.sb_selection, self.sb_units, self.sb_frame, backend):
            sb.addPermanentWidget(w)
        self._update_units_label()

    def _connect(self) -> None:
        sc = self.scene
        sc.selectionChanged.connect(lambda _ids: self._update_actions())
        sc.objectsAdded.connect(lambda _ids: self._update_actions())
        sc.objectsRemoved.connect(lambda _ids: self._update_actions())
        # bound methods (not lambdas) so Qt disconnects them automatically when the window is destroyed
        self.undo.cleanChanged.connect(self._on_undo_changed)
        self.undo.indexChanged.connect(self._on_undo_changed)
        self.viewport.dropRequested.connect(self._on_drop)
        self.viewport.contextMenuRequested.connect(self.show_context_menu)
        self.viewport.toolChanged.connect(lambda t: self.actions_[f"tool.{t}"].setChecked(True))
        self.viewport.statsUpdated.connect(self._on_stats)
        self.viewport.message.connect(lambda msg: self.bus.error(msg, "viewport"))
        self.library_panel.addRequested.connect(self.add_library_item)
        self.library_panel.importMeshRequested.connect(self.import_obj)
        self.library_panel.deleteTemplateRequested.connect(self.delete_template)
        self.outliner.reparentRequested.connect(self._reparent)
        self.outliner.contextMenuRequested.connect(self.show_context_menu)
        self.outliner.frameRequested.connect(lambda oid: self.viewport.frame_objects([oid]))
        self.object_table.frameRequested.connect(lambda oid: self.viewport.frame_objects([oid]))
        self.inspector.errorOccurred.connect(lambda msg: self.bus.error(msg, "inspector"))
        self.inspector.statusMessage.connect(lambda msg: self.statusBar().showMessage(msg, 3000))
        self.problems.countChanged.connect(self._problem_count)
        self._last_undo_index = 0
        QGuiApplication.clipboard().dataChanged.connect(self._update_actions)
        QApplication.instance().aboutToQuit.connect(self._disconnect_for_shutdown)

    def _disconnect_for_shutdown(self) -> None:
        """Detach observers whose targets are destroyed in arbitrary order during teardown."""
        for sig in (self.undo.indexChanged, self.undo.cleanChanged, QGuiApplication.clipboard().dataChanged):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass

    # ======================================================================================
    # state helpers
    # ======================================================================================
    def is_dirty(self) -> bool:
        return not self.undo.isClean() or self._extra_dirty

    def _mark_dirty(self) -> None:
        self._extra_dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = os.path.basename(self.project_path) if self.project_path else "Untitled"
        self.setWindowTitle(f"{name}[*] — {APP_NAME}")
        self.setWindowModified(self.is_dirty())

    def _update_actions(self, *args) -> None:
        sel = self.scene.selection
        has = bool(sel)
        a = self.actions_
        for k in ("edit.cut", "edit.copy", "edit.duplicate", "edit.delete", "obj.lock", "obj.hide",
                  "obj.add_property", "obj.save_template", "view.frame_selected", "scene.group"):
            a[k].setEnabled(has)
        a["edit.rename"].setEnabled(len(sel) == 1)
        a["file.export_obj"].setEnabled(len(sel) == 1)
        a["scene.ungroup"].setEnabled(any(self.scene.children(i) for i in sel))
        a["scene.unparent"].setEnabled(any(self.scene.get(i).parent_id for i in sel))
        for axis in "xyz":
            for mode in ("min", "center", "max"):
                a[f"align.{axis}.{mode}"].setEnabled(len(sel) >= 2)
            a[f"distribute.{axis}"].setEnabled(len(sel) >= 3)
        mime = QGuiApplication.clipboard().mimeData()
        a["edit.paste"].setEnabled(mime is not None and mime.hasFormat(CLIP_MIME))
        a["file.recover"].setEnabled(Autosaver.pending_recovery() is not None)
        n = len(sel)
        self.sb_selection.setText(f"{n} selected" if n else f"{len(self.scene)} objects")

    def _update_units_label(self) -> None:
        u = self.units
        self.sb_units.setText(f"Units: {u.preferred_unit('length')}, {u.preferred_unit('mass')}, "
                              f"{u.preferred_unit('time')}, {u.preferred_unit('angle')}")

    def _on_stats(self, stats) -> None:
        fps = self.viewport.measured_fps
        if self.viewport.continuous:
            self.sb_frame.setText(f"{fps:.0f} FPS · {stats.cpu_ms:.1f} ms")
        else:
            self.sb_frame.setText(f"frame {stats.cpu_ms:.1f} ms")
        if self.bottom_tabs.currentWidget() is self.performance:
            self.performance.update_stats(stats, fps, self.viewport.continuous)
        if not getattr(self, "_gl_logged", False) and self.viewport.renderer.gl_info:
            self._gl_logged = True
            info = self.viewport.renderer.gl_info
            self.performance.set_gl_info(info)
            self.bus.info(f"OpenGL {info.get('version')} · {info.get('renderer')}", "viewport")

    def _problem_count(self, errors: int, warnings: int) -> None:
        idx = self.bottom_tabs.indexOf(self.problems)
        text = "Errors && Warnings" + (f" ({errors + warnings})" if errors + warnings else "")
        self.bottom_tabs.setTabText(idx, text)

    def _on_undo_changed(self, *_args) -> None:
        self._update_title()
        self._log_undo()

    def _log_undo(self) -> None:
        idx = self.undo.index()
        if idx > self._last_undo_index and idx > 0:
            self.bus.info(self.undo.text(idx - 1), "edit")
        self._last_undo_index = idx

    # ======================================================================================
    # project lifecycle
    # ======================================================================================
    def _confirm_discard(self) -> bool:
        if not self.is_dirty() or not self.scene.objects and self.project_path is None:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "The project has unsaved changes. Save them first?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                 QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Save:
            return self.save()
        return r == QMessageBox.StandardButton.Discard

    def _reset_project(self) -> None:
        self.scene.clear()
        self.undo.clear()
        mesh_assets.clear()
        self.library.reload_meshes()
        self.library_panel.rebuild()
        self.project = ser.ProjectData()
        self.units.load_dict({})
        self._apply_view_settings(self.project.view)
        self.project_path = None
        self._extra_dirty = False

    def new_project(self, template: str | None = None) -> None:
        if not self._confirm_discard():
            return
        self._reset_project()
        if template and PROJECT_TEMPLATES.get(template, (None, None))[1]:
            label, fn = PROJECT_TEMPLATES[template]
            fn(self.scene, self.editor)
            self.undo.clear()
            self.viewport.frame_objects(None)
            self.bus.info(f"New project from template '{label}'")
        else:
            self.bus.info("New project")
        self.undo.setClean()
        self._update_title()
        self._update_actions()

    def open_project_dialog(self) -> None:
        if not self._confirm_discard():
            return
        start = os.path.dirname(self.project_path) if self.project_path else str(self.settings.value("last_dir", ""))
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", start, FILE_FILTER)
        if path:
            self.open_project(path, confirm=False)

    def open_project(self, path: str, confirm: bool = True) -> bool:
        if confirm and not self._confirm_discard():
            return False
        try:
            raw = ser.read_project_file(path)
            self._load_raw(raw)
        except ser.ProjectError as exc:
            QMessageBox.critical(self, "Open project", f"Could not open '{os.path.basename(path)}':\n\n{exc}")
            self.bus.error(f"Open failed: {exc}", "project")
            return False
        self.project_path = path
        self._add_recent(path)
        self.settings.setValue("last_dir", os.path.dirname(path))
        self.bus.info(f"Opened {path} ({len(self.scene)} objects)", "project")
        self._update_title()
        return True

    def _load_raw(self, raw: dict) -> None:
        scratch_scene = Scene()
        # validate fully on a scratch scene first so a bad file never clobbers the open project
        ser.load_project_dict(raw, scratch_scene, type(self.units)())
        self.scene.clear()
        self.undo.clear()
        data = ser.load_project_dict(raw, self.scene, self.units)
        self.project = data
        self._apply_view_settings(data.view)
        self.library.reload_meshes()
        self.library_panel.rebuild()
        if data.ui.get("layout"):
            self.restoreState(QByteArray.fromBase64(data.ui["layout"].encode()))
        for w in data.warnings:
            self.bus.warning(w, "project")
        self._extra_dirty = False
        self.undo.setClean()
        self._update_units_label()
        self._update_actions()
        self.object_table.rebuild()

    def _apply_view_settings(self, view: ViewSettings) -> None:
        self.project.view = view
        self.viewport.set_view_settings(view)
        a = self.actions_
        for key in ("XY", "YZ", "XZ"):
            _set_checked(a[f"grid.{key}"], view.grids[key].visible)
        _set_checked(a["scene.axes"], view.axes.visible)
        _set_checked(a["view.ortho"], view.camera.orthographic)
        env = view.environment
        for k, attr in (("view.labels", "show_labels"), ("view.local_axes", "show_local_axes"),
                        ("view.shadows", "shadows"), ("view.ground_shadow", "ground_shadow"),
                        ("view.stats", "show_stats")):
            _set_checked(a[k], getattr(env, attr))
        sn = view.snap
        for k, attr in (("snap.grid", "grid_snap"), ("snap.rotate", "rotate_snap"), ("snap.scale", "scale_snap"),
                        ("snap.surface", "surface_snap")):
            _set_checked(a[k], getattr(sn, attr))
        _set_checked(a[f"plane.{sn.work_plane}"], True)
        self.plane_combo.blockSignals(True)
        self.plane_combo.setCurrentText(sn.work_plane)
        self.plane_combo.blockSignals(False)

    def _snapshot(self) -> dict:
        self.project.ui["layout"] = bytes(self.saveState().toBase64()).decode()
        if self.project_path:
            self.project.metadata["path"] = self.project_path
        return ser.project_to_dict(self.scene, self.units, self.project)

    def save(self) -> bool:
        if not self.project_path:
            return self.save_as()
        return self._save_to(self.project_path)

    def save_as(self) -> bool:
        start = self.project_path or os.path.join(str(self.settings.value("last_dir", "")), "untitled.psim")
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", start, FILE_FILTER)
        if not path:
            return False
        if not os.path.splitext(path)[1]:
            path += ".psim"
        return self._save_to(path)

    def _save_to(self, path: str) -> bool:
        try:
            self.project.ui["layout"] = bytes(self.saveState().toBase64()).decode()
            self.project.metadata.setdefault("name", os.path.splitext(os.path.basename(path))[0])
            ser.save_project(path, self.scene, self.units, self.project)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Save project", f"Could not save the project:\n\n{exc}")
            self.bus.error(f"Save failed: {exc}", "project")
            return False
        self.project_path = path
        self.undo.setClean()
        self._extra_dirty = False
        self._add_recent(path)
        self.settings.setValue("last_dir", os.path.dirname(path))
        Autosaver.discard()
        self.bus.info(f"Saved {path}", "project")
        self.statusBar().showMessage(f"Saved {os.path.basename(path)}", 3000)
        self._update_title()
        self._update_actions()
        return True

    def _offer_recovery(self, explicit: bool = False) -> None:
        data = Autosaver.pending_recovery()
        if data is None:
            if explicit:
                QMessageBox.information(self, "Recover", "No autosaved data is available.")
            return
        r = QMessageBox.question(
            self, "Recover unsaved work",
            "PSim Studio found autosaved work from a session that did not close normally.\n\n"
            "Do you want to recover it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Discard)
        if r == QMessageBox.StandardButton.Yes:
            try:
                self._load_raw(data)
                self.project_path = None
                self._extra_dirty = True
                self.bus.info("Recovered autosaved project (not yet saved)", "project")
            except ser.ProjectError as exc:
                QMessageBox.critical(self, "Recover", f"The recovery file could not be loaded:\n\n{exc}")
            self._update_title()
        elif r == QMessageBox.StandardButton.Discard:
            Autosaver.discard()
        self._update_actions()

    def _recent(self) -> list[str]:
        v = self.settings.value("recent", [])
        if isinstance(v, str):
            v = [v]
        return [p for p in (v or []) if isinstance(p, str)]

    def _add_recent(self, path: str) -> None:
        rec = [p for p in self._recent() if os.path.abspath(p) != os.path.abspath(path)]
        self.settings.setValue("recent", [path, *rec][:MAX_RECENT])

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        rec = self._recent()
        for p in rec:
            act = self.recent_menu.addAction(os.path.basename(p), lambda p=p: self.open_project(p))
            act.setToolTip(p)
            act.setEnabled(os.path.exists(p))
        if not rec:
            self.recent_menu.addAction("No recent projects").setEnabled(False)
        else:
            self.recent_menu.addSeparator()
            self.recent_menu.addAction("Clear List", lambda: self.settings.setValue("recent", []))

    # ======================================================================================
    # object creation / library
    # ======================================================================================
    def _default_placement(self, key: str) -> np.ndarray:
        vp = self.viewport
        mesh = self.library.preview_mesh(key)
        lo, hi = mesh.bounds() if mesh is not None else (np.zeros(3), np.zeros(3))
        o, d = vp.camera.screen_ray(vp.width() / 2, vp.height() / 2, vp.width(), vp.height())
        sn = self.project.view.snap
        pos, _ = picking.placement_position(self.scene, o, d, lo, hi, sn.work_plane, sn.work_plane_offset,
                                            sn.grid_snap, sn.translate_increment, sn.surface_snap,
                                            fallback=vp.camera.target)
        return pos

    def add_library_item(self, key: str, position=None) -> list:
        item = self.library.get(key)
        if item is None:
            return []
        if position is None:
            position = self._default_placement(key)
        try:
            if item.kind in ("type", "mesh"):
                obj = self.editor.add_new(item.type_id, position=position, params=item.params,
                                          name=None if item.kind == "type" else self.scene.unique_name(item.name))
                return [obj]
            return self.editor.instantiate(item.states, offset=position, text=f"Add {item.name}:")
        except (PropertyError, SceneError) as exc:
            self.bus.error(str(exc), "library")
            return []

    def _on_drop(self, payload: str, position: list, surface_id) -> None:
        try:
            key = json.loads(payload).get("key", "")
        except ValueError:
            return
        objs = self.add_library_item(key, np.asarray(position))
        if objs:
            where = f" on {self.scene.get(surface_id).name}" if surface_id and surface_id in self.scene else ""
            self.statusBar().showMessage(f"Placed {objs[0].name} ({objs[0].id}){where}", 4000)
            self.viewport.setFocus()

    def import_obj(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Mesh", str(self.settings.value("last_dir", "")),
                                              "Wavefront OBJ (*.obj);;All files (*)")
        if path:
            self.import_obj_file(path)

    def import_obj_file(self, path: str):
        try:
            if os.path.getsize(path) > 200 * 1024 * 1024:
                raise geo.MeshImportError("File is larger than 200 MB")
            with open(path, encoding="utf-8", errors="replace") as fh:
                mesh = geo.load_obj(fh.read())
        except (OSError, geo.MeshImportError) as exc:
            QMessageBox.warning(self, "Import mesh", f"Could not import '{os.path.basename(path)}':\n\n{exc}")
            self.bus.error(f"Mesh import failed: {exc}", "import")
            return None
        name = os.path.splitext(os.path.basename(path))[0]
        key = mesh_assets.add(mesh, name)
        self.library.reload_meshes()
        self.library_panel.rebuild()
        self._fill_add_menu(self.add_menu)
        objs = self.add_library_item(f"mesh:{key}")
        closed = "closed" if mesh.solid else "open (treated as a thin shell for mass properties)"
        self.bus.info(f"Imported {name}: {mesh.triangle_count:,} triangles, {closed}", "import")
        return objs[0] if objs else None

    def export_obj(self) -> None:
        o = self.scene.primary
        if o is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Mesh", f"{o.name}.obj", "Wavefront OBJ (*.obj)")
        if not path:
            return
        mesh = o.mesh().transformed(self.scene.world_matrix(o.id))
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(geo.mesh_to_obj(mesh))
        except OSError as exc:
            QMessageBox.critical(self, "Export mesh", str(exc))
            return
        self.bus.info(f"Exported {o.name} to {path} (global coordinates)", "export")

    def save_template(self) -> None:
        sel = self.scene.selection
        if not sel:
            return
        name, ok = QInputDialog.getText(self, "Save as Template", "Template name:",
                                        text=self.scene.get(sel[-1]).name)
        if not ok or not name.strip():
            return
        try:
            self.library.save_template(name.strip(), self.editor.serialize(sel))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Save template", str(exc))
            return
        self.library_panel.rebuild()
        self._fill_add_menu(self.add_menu)
        self.bus.info(f"Saved template '{name.strip()}'", "library")

    def delete_template(self, key: str) -> None:
        it = self.library.get(key)
        if it and QMessageBox.question(self, "Delete template", f"Delete template '{it.name}'?") == \
                QMessageBox.StandardButton.Yes:
            self.library.delete_template(key)
            self.library_panel.rebuild()
            self._fill_add_menu(self.add_menu)

    # ======================================================================================
    # editing commands
    # ======================================================================================
    def copy(self) -> None:
        states = self.editor.serialize(self.scene.selection)
        if not states:
            return
        mime = QMimeData()
        mime.setData(CLIP_MIME, json.dumps(states).encode())
        mime.setText("\n".join(f"{s['id']}\t{s['name']}\t{s['type']}" for s in states))
        QGuiApplication.clipboard().setMimeData(mime)
        self.statusBar().showMessage(f"Copied {len(states)} object(s)", 2500)

    def cut(self) -> None:
        self.copy()
        self.delete_selection()

    def paste(self) -> None:
        mime = QGuiApplication.clipboard().mimeData()
        if mime is None or not mime.hasFormat(CLIP_MIME):
            return
        try:
            states = json.loads(bytes(mime.data(CLIP_MIME)).decode())
            if not isinstance(states, list):
                raise ValueError("clipboard data is not a list")
            self.editor.instantiate(states, offset=(0.5, 0.5, 0.0))
        except (ValueError, PropertyError, SceneError, KeyError) as exc:
            self.bus.error(f"Paste failed: {exc}", "edit")

    def duplicate(self) -> None:
        self.editor.duplicate(self.scene.selection)

    def delete_selection(self) -> None:
        sel = self.scene.selection
        locked = [i for i in sel if self.scene.is_effectively_locked(i)]
        if locked:
            self.statusBar().showMessage(f"{len(locked)} locked object(s) were not deleted", 4000)
        self.editor.delete([i for i in sel if i not in locked])

    def rename(self) -> None:
        o = self.scene.primary
        if o is None:
            return
        name, ok = QInputDialog.getText(self, "Rename", f"New name for {o.id}:", text=o.name)
        if ok and name.strip():
            self.editor.set_property([o.id], "name", name.strip(), merge=False)

    def invert_selection(self) -> None:
        sel = set(self.scene.selection)
        self.scene.set_selection([i for i in self.scene.objects if i not in sel])

    def group(self) -> None:
        g = self.editor.group(self.scene.selection)
        if g is not None:
            self.bus.info(f"Grouped into {g.name} ({g.id})", "edit")

    def ungroup(self) -> None:
        self.editor.ungroup(self.scene.selection)

    def _reparent(self, ids, parent_id) -> None:
        try:
            self.editor.reparent(ids, parent_id)
        except SceneError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            self.bus.warning(str(exc), "hierarchy")

    def toggle_lock(self) -> None:
        sel = self.scene.selection
        if sel:
            self.editor.set_property(sel, "locked", not self.scene.get(sel[-1]).locked, merge=False)

    def toggle_hide(self) -> None:
        sel = self.scene.selection
        if sel:
            self.editor.set_property(sel, "visible", not self.scene.get(sel[-1]).visible, merge=False)

    def unhide_all(self) -> None:
        hidden = [o.id for o in self.scene.objects.values() if not o.visible]
        if hidden:
            self.editor.set_property(hidden, "visible", True, merge=False)

    def _align(self, axis: int, mode: str) -> None:
        self.editor.align(self.scene.selection, axis, mode)

    def _distribute(self, axis: int) -> None:
        self.editor.distribute(self.scene.selection, axis)

    def _frame_selection(self) -> None:
        sel = self.scene.selection
        self.viewport.frame_objects(sel if sel else None)

    def mass_report(self) -> None:
        ids = self.scene.selection or list(self.scene.objects)
        MassReportDialog(self.scene, ids, self.units, self).exec()

    def scene_statistics(self) -> None:
        counts: dict[str, int] = {}
        tris = 0
        for o in self.scene.objects.values():
            counts[o.typedef.name] = counts.get(o.typedef.name, 0) + 1
            tris += o.mesh().triangle_count
        b = self.scene.world_bounds()
        lu = self.units.preferred_unit("length")
        size = "—" if b is None else " × ".join(f"{v:.3g}" for v in self.units.from_si(b[1] - b[0], lu)) + f" {lu}"
        lines = "".join(f"<tr><td>{k}</td><td align='right'>{v}</td></tr>" for k, v in sorted(counts.items()))
        QMessageBox.information(self, "Scene Statistics",
                                f"<b>{len(self.scene)}</b> objects · <b>{tris:,}</b> triangles<br>"
                                f"Scene extent: {size}<br>Next object ID: OBJ-{self.scene.ids.next_value:06d}"
                                f"<table cellspacing='6'>{lines}</table>")

    # ======================================================================================
    # context menu
    # ======================================================================================
    def show_context_menu(self, global_pos, oid) -> None:
        a = self.actions_
        m = QMenu(self)
        if oid:
            o = self.scene.get(oid)
            title = m.addAction(f"{o.name}  ·  {o.id}")
            title.setEnabled(False)
            m.addSeparator()
            m.addActions([a["view.frame_selected"], a["edit.rename"], a["edit.duplicate"], a["edit.copy"],
                          a["edit.cut"], a["edit.delete"]])
            m.addSeparator()
            m.addActions([a["scene.group"], a["scene.ungroup"], a["scene.unparent"]])
            m.addSeparator()
            m.addActions([a["obj.hide"], a["obj.lock"]])
            m.addSeparator()
            m.addActions([a["obj.add_property"], a["obj.save_template"], a["obj.mass_report"]])
            cam = m.addMenu(icons.icon("camera"), "Camera")
            cam.addActions(self.follow_group.actions())
        else:
            m.addMenu(self.add_menu)
            m.addAction(a["edit.paste"])
            m.addSeparator()
            m.addActions([a["edit.select_all"], a["view.fit_all"], a["view.focus_origin"]])
            m.addSeparator()
            pm = m.addMenu("Placement Plane")
            pm.addActions(self.plane_group.actions())
            m.addAction(a["obj.unhide_all"])
        m.exec(global_pos)

    # ======================================================================================
    # view / settings
    # ======================================================================================
    def _set_gizmo_global(self, on: bool) -> None:
        self.viewport.set_gizmo_global(on)
        self.actions_["tool.global"].setIcon(icons.icon("globe" if on else "local"))
        self.actions_["tool.global"].setText("&Global Gizmo Orientation" if on else "&Local Gizmo Orientation")

    def _set_follow(self, mode: str) -> None:
        prim = self.scene.primary
        if mode != "none" and prim is None:
            self.statusBar().showMessage("Select an object to follow first", 3000)
            _set_checked(self.actions_["camera.none"], True)
            mode = "none"
        self.viewport.set_follow(mode, prim.id if prim and mode != "none" else None)

    def _env(self, attr: str, value) -> None:
        setattr(self.project.view.environment, attr, bool(value))
        self._mark_dirty()
        self.viewport.update()

    def _snap(self, attr: str, value) -> None:
        setattr(self.project.view.snap, attr, bool(value))
        self._mark_dirty()

    def _set_grid(self, key: str, on: bool) -> None:
        self.project.view.grids[key].visible = bool(on)
        self._mark_dirty()
        self.viewport.update()

    def _set_axes(self, on: bool) -> None:
        self.project.view.axes.visible = bool(on)
        self._mark_dirty()
        self.viewport.update()

    def _set_work_plane(self, key: str) -> None:
        if key not in ("XY", "YZ", "XZ"):
            return
        self.project.view.snap.work_plane = key
        _set_checked(self.actions_[f"plane.{key}"], True)
        if self.plane_combo.currentText() != key:
            self.plane_combo.blockSignals(True)
            self.plane_combo.setCurrentText(key)
            self.plane_combo.blockSignals(False)
        self._mark_dirty()

    def open_viewport_settings(self, tab: str) -> None:
        dlg = ViewportSettingsDialog(self.project.view, self.units, tab, self)

        def changed():
            self._mark_dirty()
            self.viewport.update()
        dlg.changed.connect(changed)
        dlg.exec()
        self._apply_view_settings(self.project.view)

    def add_bookmark(self) -> None:
        name, ok = QInputDialog.getText(self, "Camera Bookmark", "Bookmark name:",
                                        text=f"View {len(self.project.view.bookmarks) + 1}")
        if ok and name.strip():
            from dataclasses import asdict
            self.project.view.bookmarks[name.strip()] = CameraState(**asdict(self.viewport.camera.state))
            self._mark_dirty()

    def _fill_bookmarks(self) -> None:
        m = self.bookmark_menu
        m.clear()
        m.addAction(self.actions_["camera.add_bookmark"])
        bms = self.project.view.bookmarks
        if bms:
            m.addSeparator()
        for name in bms:
            m.addAction(name, lambda n=name: self._restore_bookmark(n))
        if bms:
            m.addSeparator()
            dm = m.addMenu("Delete Bookmark")
            for name in bms:
                dm.addAction(name, lambda n=name: (self.project.view.bookmarks.pop(n, None), self._mark_dirty()))

    def _restore_bookmark(self, name: str) -> None:
        from dataclasses import asdict
        bm = self.project.view.bookmarks.get(name)
        if bm is None:
            return
        for k, v in asdict(bm).items():
            setattr(self.viewport.camera.state, k, v)
        _set_checked(self.actions_["view.ortho"], bm.orthographic)
        self.viewport.update()

    def toggle_fullscreen(self) -> None:
        if self._fullscreen_state is None:
            self._fullscreen_state = self.saveState()
            self._was_maximized = self.isMaximized()
            for d in self.findChildren(QDockWidget):
                d.hide()
            self.toolbar.hide()
            self.menuBar().hide()
            self.statusBar().hide()
            self.showFullScreen()
            self.statusBar().showMessage("Press F11 to leave full-screen", 3000)
        else:
            self.menuBar().show()
            self.statusBar().show()
            self.showMaximized() if self._was_maximized else self.showNormal()
            self.restoreState(self._fullscreen_state)
            self._fullscreen_state = None

    def open_units(self) -> None:
        UnitsDialog(self.units, self).exec()
        self._units_changed()

    def reset_units(self) -> None:
        self.units.reset_preferences()
        self._units_changed()

    def _units_changed(self) -> None:
        self._mark_dirty()
        self._update_units_label()
        self.inspector._signature = None
        self.inspector._schedule()
        self.object_table.rebuild()
        self.viewport.update()

    def open_shortcuts(self) -> None:
        if ShortcutsDialog(self.actions_, self).exec():
            self.actions_.save_overrides(self.settings)

    def open_preferences(self) -> None:
        dlg = PreferencesDialog(theme.current().name, self.autosaver.interval_s, self.viewport.render_fps, self)
        if dlg.exec():
            self.set_theme(dlg.theme.currentData())
            self.autosaver.set_interval(dlg.autosave.value())
            self.settings.setValue("autosave_s", dlg.autosave.value())
            self.settings.setValue("render_fps", dlg.fps.value())
            self.viewport.set_continuous(self.viewport.continuous, float(dlg.fps.value()))

    def set_theme(self, name: str) -> None:
        theme.apply(QApplication.instance(), name)
        self.settings.setValue("theme", name)
        for aid, act in self.actions_.items():
            ic = act.property("icon_name")
            if ic:
                act.setIcon(icons.icon(ic))
        for act in self.theme_group.actions():
            act.setChecked(act.data() == name)
        self.library_panel.rebuild()
        self.inspector._signature = None
        self.inspector._schedule()
        self.outliner.rebuild()

    def save_layout(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Layout", "Layout name:")
        if ok and name.strip():
            self.settings.setValue(f"layouts/{name.strip()}", self.saveState())

    def _fill_layouts(self) -> None:
        m = self.layout_menu
        m.clear()
        self.settings.beginGroup("layouts")
        names = self.settings.childKeys()
        self.settings.endGroup()
        for n in names:
            m.addAction(n, lambda n=n: self.restoreState(self.settings.value(f"layouts/{n}")))
        if names:
            m.addSeparator()
            dm = m.addMenu("Delete")
            for n in names:
                dm.addAction(n, lambda n=n: self.settings.remove(f"layouts/{n}"))
        else:
            m.addAction("No saved layouts").setEnabled(False)

    def reset_layout(self) -> None:
        self.restoreState(self._default_state)
        for d in self.findChildren(QDockWidget):
            d.show()
        self.dock_history.hide()

    def about(self) -> None:
        QMessageBox.about(self, f"About {APP_NAME}", about_html(self.viewport.renderer.gl_info))

    # ======================================================================================
    # window state
    # ======================================================================================
    def _restore_window(self) -> None:
        geom = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        if geom is not None:
            self.restoreGeometry(geom)
        else:
            screen = QGuiApplication.primaryScreen().availableGeometry()
            self.resize(int(screen.width() * 0.86), int(screen.height() * 0.86))
        if state is not None:
            self.restoreState(state)
        else:
            self.dock_history.hide()
            QTimer.singleShot(0, self._initial_dock_sizes)

    def _initial_dock_sizes(self) -> None:
        self.resizeDocks([self.dock_library, self.dock_inspector], [300, 380], Qt.Orientation.Horizontal)
        self.resizeDocks([self.dock_bottom], [max(150, self.height() // 6)], Qt.Orientation.Vertical)
        self.resizeDocks([self.dock_library, self.dock_outliner], [self.height() // 2, self.height() // 4],
                         Qt.Orientation.Vertical)

    def closeEvent(self, e: QCloseEvent) -> None:  # noqa: N802
        if not self._confirm_discard():
            e.ignore()
            return
        if self._fullscreen_state is not None:
            self.toggle_fullscreen()
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.actions_.save_overrides(self.settings)
        Autosaver.discard()
        e.accept()


def _set_checked(action: QAction, value: bool) -> None:
    action.blockSignals(True)
    action.setChecked(bool(value))
    action.blockSignals(False)
