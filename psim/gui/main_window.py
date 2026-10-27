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
from ..processes.process_base import ProcessError
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
from .simulation_controller import SimulationController
from ..rendering.trajectories import TrajectoryBuilder
from .plots.workspace import QUICK_PLOTS, PlotWorkspace
from .recording_panel import RecordingPanel
from .relative_motion_panel import RelativeMotionPanel
from .measurements_panel import MeasurementsPanel
from .simulation_panels import ProcessesPanel, SimulationFields
from .dialogs.process_dialogs import AssignProcessDialog, DependencyDialog, ProcessDocumentationDialog
from .dialogs.simulation_settings_dialog import SimulationSettingsDialog
from ..simulation.settings import PacingMode

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
        self.sim = SimulationController(self.scene, self.editor, lambda: self.project.simulation, self)
        self.developer_token = None

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
        self.processes_panel = ProcessesPanel(self.scene, self.editor)
        self.bottom_tabs.addTab(self.processes_panel, icons.icon("function"), "Processes")
        self.relative_motion = RelativeMotionPanel(self.scene, self.units)
        self.bottom_tabs.addTab(self.relative_motion, icons.icon("frame_axes"), "Relative Motion")
        self.measurements_panel = MeasurementsPanel(self.scene, self.editor, self.units)
        self.measurements_panel.errorOccurred.connect(lambda m: self.bus.error(m, "measurements"))
        self.measurements_panel.pickRequested.connect(self.measure_points)
        self.bottom_tabs.addTab(self.measurements_panel, icons.icon("ruler"), "Measurements")
        self.recording_panel = RecordingPanel(self.scene, self.units, lambda: self.sim.recording,
                                              lambda: self.project.simulation)
        self.bottom_tabs.addTab(self.recording_panel, icons.icon("history"), "Recording")
        self.bottom_tabs.addTab(self.performance, icons.icon("activity"), "Performance")
        self.dock_bottom = self._dock("Panels", "dock_bottom", self.bottom_tabs, B)
        self.plot_workspace = PlotWorkspace(self.scene, self.units, lambda: self.sim.recording)
        self.dock_plots = self._dock("Plot Workspace", "dock_plots", self.plot_workspace, B)
        self.tabifyDockWidget(self.dock_bottom, self.dock_plots)
        self.dock_bottom.raise_()
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
        # own undo/redo actions so they can be disabled while a simulation session owns the scene state
        self.act_undo = A("edit.undo", "&Undo", self._undo, icon="undo", shortcut="Ctrl+Z")
        self.act_redo = A("edit.redo", "&Redo", self._redo, icon="redo", shortcut="Ctrl+Y")
        # simulation
        A("sim.start", "&Start / Resume", lambda: self._sim_start(), icon="play", shortcut="F5")
        A("sim.pause", "&Pause", self.sim.pause, icon="pause", shortcut="F6")
        A("sim.stop", "S&top", lambda: self.sim.stop(), icon="stop", shortcut="Shift+F5")
        A("sim.reset", "&Reset", self.sim.reset, icon="reset", shortcut="Ctrl+F5")
        A("sim.step", "Single S&tep", lambda: self._sim_start(step=True), icon="step", shortcut="F10")
        A("sim.afap", "Run as &Fast as Possible", self._set_afap, icon="fast", checkable=True,
          tip="Scientific calculation mode: no real-time pacing (Δt and duration still apply)")
        A("sim.settings", "Simulation &Settings…", self.open_simulation_settings, icon="timer")
        A("sim.save_state", "Save Simulation S&tate…", self.save_simulation_state, icon="save",
          tip="Save the current simulation state (time, every body's motion) to continue from it later")
        A("sim.load_state", "Restore Simulation St&ate…", self.restore_simulation_state, icon="open",
          tip="Continue from a saved simulation state of this project")
        A("sim.conditions", "Stop &Conditions…", self.open_stop_conditions, icon="stop",
          tip="Pause or end the run when a formula becomes true (e.g. an object reaches the ground)")
        # processes
        A("proc.assign", "&Assign Process…", lambda: self.assign_process(self.scene.selection), icon="function",
          shortcut="Ctrl+Shift+P")
        A("proc.manager", "Process &Table", self._show_processes, icon="table")
        A("proc.validate", "&Validate Processes", self.validate_processes, icon="check")
        A("proc.deps", "&Dependencies && Diagnostics…", self.show_dependencies, icon="tree")
        A("proc.globals", "&Global Parameters…", self.edit_global_parameters, icon="properties")
        A("tools.developer", "&Developer Mode…", self.toggle_developer_mode, icon="lock",
          tip="Unlock process authoring (requires developer credentials)")
        A("dev.new", "&New Formula Process…", lambda: self.open_process_editor(None), icon="function")
        A("dev.edit", "&Edit Formula Process (new version)…", self.edit_formula_process, icon="edit")
        A("dev.withdraw", "&Withdraw Published Process…", self.withdraw_process, icon="trash")
        A("proc.docs", "Process &Documentation…", lambda: ProcessDocumentationDialog(self.units, self).exec(),
          icon="help")
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
        A("physics.joints", "&Joints…", lambda: self.open_joints(), icon="target",
          tip="Create and edit joints (fixed, ball, revolute, prismatic, distance)")
        A("physics.show_colliders", "Show All &Colliders", lambda v: self._env("show_colliders", v), icon="cube",
          checkable=True, tip="Draw every collision shape (green: exact, yellow: approximate)")
        A("physics.show_joints", "Show &Joints", lambda v: self._env("show_joints", v), checkable=True, checked=True)
        A("physics.show_contacts", "Show C&ontacts", lambda v: self._env("show_contacts", v), checkable=True,
          checked=True, tip="Draw contact points and normals (length ∝ contact force) while simulating")
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
        A("obj.place_relative", "&Place Relative To…", self.place_relative, icon="target",
          tip="Put the selection at a numeric offset from another object, a reference frame or the origin")
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
        A("coord.frames", "&Reference Frames…", self.open_reference_frames, icon="frame_axes",
          tip="Create and edit named reference frames (fixed or attached to objects)")
        A("coord.add_frame", "&Add Reference Frame at Selection", self.add_frame_at_selection, icon="target",
          tip="Attach a new named frame to the selected object, at its origin")
        A("tools.measure_distance", "Measure &Distance", lambda: self.measure("distance"), icon="ruler",
          tip="Add a live distance measurement between the two selected objects")
        A("tools.measure_angle", "Measure &Angle", lambda: self.measure("angle"),
          tip="Add a live angle measurement at the second of three selected objects")
        A("tools.measure_point_distance", "Measure Distance Between &Points", lambda: self.measure_points("distance"),
          tip="Click two points on object surfaces in the viewport; the distance follows the objects")
        A("tools.measure_point_angle", "Measure Angle Between P&oints", lambda: self.measure_points("angle"),
          tip="Click three points on object surfaces (A, the vertex, B); the angle follows the objects")
        A("tools.measurements", "&Measurements Panel", self.show_measurements, icon="ruler")
        A("coord.relative", "Relative &Motion Panel", self.show_relative_motion, icon="activity",
          tip="Position, velocity and acceleration of an object expressed in / relative to any frame")
        A("plots.workspace", "Plot &Workspace", self.show_plots, icon="chart", shortcut="Ctrl+Shift+P")
        A("plots.new2d", "New &2D Plot…", lambda: (self.show_plots(), self.plot_workspace.new_plot("2d")),
          icon="chart")
        A("plots.new3d", "New &3D Plot…", lambda: (self.show_plots(), self.plot_workspace.new_plot("3d")),
          icon="cube")
        A("data.panel", "Data &Recording Panel", self.show_recording_panel, icon="history")
        A("data.csv", "Export Recorded Data (&CSV)…", self.export_recorded_csv, icon="table")
        A("data.json", "Export Recorded Data (&JSON)…", self.export_recorded_json, icon="export")
        A("data.report", "Simulation &Report…", self.generate_report, icon="info")
        A("data.clear", "Clear Recording", self.sim_clear_recording, icon="trash")
        A("units.dialog", "&Units && Custom Units…", self.open_units, icon="ruler")
        A("units.si", "Reset Display Units to &SI", self.reset_units)
        A("units.inspector", "&Formula Unit Inspector…", self.open_unit_inspector, icon="function",
          tip="Unit requirements of processes and dimensions of formula expressions")
        A("tools.shortcuts", "&Keyboard Shortcuts…", self.open_shortcuts, icon="keyboard")
        A("tools.preferences", "&Preferences…", self.open_preferences, icon="settings", shortcut="Ctrl+,")
        A("window.save_layout", "&Save Layout As…", self.save_layout)
        A("window.reset_layout", "&Reset Layout", self.reset_layout)
        A("help.controls", "&Controls && Shortcuts", lambda: HelpDialog(self).exec(), icon="keyboard",
          shortcut="F1")
        A("help.about", "&About PSim Studio", self.about, icon="info")
        A("help.sysinfo", "&System Information…", self.show_system_info, icon="terminal",
          tip="Versions, numerical core, OpenGL driver and the log folder (for problem reports)")
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
        m.addAction(a["obj.place_relative"])
        m.addSeparator()
        m.addActions([a["obj.lock"], a["obj.hide"], a["obj.unhide_all"]])
        m.addSeparator()
        m.addActions([a["obj.add_property"], a["obj.save_template"]])

        m = mb.addMenu("&Physics")
        m.addAction(a["obj.mass_report"])
        m.addAction(a["physics.joints"])
        m.addSeparator()
        m.addActions([a["physics.show_colliders"], a["physics.show_joints"], a["physics.show_contacts"]])
        m.addSeparator()
        m.addAction(a["sim.settings"])

        m = mb.addMenu("P&rocesses")
        m.addActions([a["proc.assign"], a["proc.manager"], a["proc.globals"], a["proc.validate"], a["proc.deps"],
                      a["proc.docs"]])

        m = mb.addMenu("&Coordinates")
        m.addAction(a["tool.global"])
        m.addAction(a["view.local_axes"])
        m.addAction(a["scene.axes"])
        m.addSeparator()
        m.addActions([a["coord.frames"], a["coord.add_frame"], a["coord.relative"]])

        m = mb.addMenu("Pl&ots")
        m.addActions([a["plots.workspace"], a["plots.new2d"], a["plots.new3d"]])
        quick = m.addMenu("&Quick Plot of Selection")
        for key, text in QUICK_PLOTS:
            quick.addAction(text, lambda k=key: (self.show_plots(), self.plot_workspace.add_quick(k)))
        m.addSeparator()
        m.addActions([a["data.panel"], a["data.csv"], a["data.json"], a["data.report"], a["data.clear"]])

        m = mb.addMenu("&Units")
        m.addActions([a["units.dialog"], a["units.si"]])
        m.addSeparator()
        m.addAction(a["units.inspector"])

        m = mb.addMenu("Si&mulation")
        m.addActions([a["sim.start"], a["sim.pause"], a["sim.stop"], a["sim.reset"], a["sim.step"]])
        m.addSeparator()
        m.addActions([a["sim.afap"], a["sim.settings"], a["sim.conditions"]])
        m.addSeparator()
        m.addActions([a["sim.save_state"], a["sim.load_state"]])

        m = mb.addMenu("&Tools")
        m.addActions([a["tools.measure_distance"], a["tools.measure_angle"], a["tools.measure_point_distance"],
                      a["tools.measure_point_angle"], a["tools.measurements"]])
        m.addSeparator()
        m.addActions([a["tools.shortcuts"], a["tools.preferences"], a["view.settings"]])
        m.addSeparator()
        m.addAction(a["tools.developer"])
        self.developer_menu = mb.addMenu("&Developer")
        self.developer_menu.addActions([a["dev.new"], a["dev.edit"], a["dev.withdraw"]])
        self.developer_menu.menuAction().setVisible(False)  # shown only while developer mode is unlocked

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
        m.addActions([a["help.controls"], a["help.sysinfo"], a["help.about"], a["help.about_qt"]])

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
        self.addToolBarBreak()
        st = self.addToolBar("Simulation")
        st.setObjectName("toolbar_simulation")
        st.setIconSize(QSize(18, 18))
        st.addActions([a["sim.start"], a["sim.pause"], a["sim.stop"], a["sim.reset"], a["sim.step"], a["sim.afap"]])
        st.addSeparator()
        self.sim_fields = SimulationFields()
        self.sim_fields.deltaTimeChanged.connect(lambda v: self._sim_setting(self.sim.set_delta_time, v))
        self.sim_fields.updateRateChanged.connect(lambda v: self._sim_setting(self.sim.set_update_rate, v))
        self.sim_fields.durationChanged.connect(lambda v: self._sim_setting(self.sim.set_duration, v))
        st.addWidget(self.sim_fields)
        st.addAction(a["sim.settings"])
        st.addSeparator()
        st.addActions([a["proc.assign"], a["proc.deps"]])
        self.sim_toolbar = st

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
        self.viewport.hint.connect(lambda msg: self.statusBar().showMessage(msg) if msg else
                                   self.statusBar().clearMessage())
        self.viewport.pointsPicked.connect(self._points_picked)
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
        self.undo.canUndoChanged.connect(self._update_undo_actions)
        self.undo.canRedoChanged.connect(self._update_undo_actions)
        self.undo.undoTextChanged.connect(self._update_undo_actions)
        self.undo.redoTextChanged.connect(self._update_undo_actions)
        self.sim.stateChanged.connect(self._on_sim_state)
        self.trajectories = TrajectoryBuilder()
        self.viewport.trajectory_provider = self._trajectory_paths
        self.viewport.contacts_provider = lambda: self.sim.contacts
        self.sim.recordingChanged.connect(self._on_recording_changed)
        self.plot_workspace.changed.connect(self._mark_dirty)
        self.recording_panel.intervalChanged.connect(self.sim.set_record_interval)
        self.recording_panel.exportCsvRequested.connect(self.export_recorded_csv)
        self.recording_panel.exportJsonRequested.connect(self.export_recorded_json)
        self.recording_panel.reportRequested.connect(self.generate_report)
        self.recording_panel.clearRequested.connect(self.sim_clear_recording)
        self.sim.statsUpdated.connect(self._on_sim_stats)
        self.sim.info.connect(lambda m: self.bus.info(m, "simulation"))
        self.sim.warning.connect(lambda m: self.bus.warning(m, "simulation"))
        self.sim.error.connect(self._on_sim_error)
        self.processes_panel.addRequested.connect(lambda: self.assign_process(self.scene.selection))
        self.processes_panel.editRequested.connect(self.edit_assignment)
        self.processes_panel.validateRequested.connect(self.validate_processes)
        self.processes_panel.dependenciesRequested.connect(self.show_dependencies)
        self.processes_panel.documentationRequested.connect(
            lambda: ProcessDocumentationDialog(self.units, self).exec())
        self.inspector.assignProcessRequested.connect(self.assign_process)
        self.inspector.editAssignmentRequested.connect(self.edit_assignment)
        self.inspector.clearTrajectoryRequested.connect(self.clear_trajectory)
        self.inspector.exportTrajectoryRequested.connect(self.export_trajectory)
        self.inspector.jointsRequested.connect(self.open_joints)
        self.scene.processesChanged.connect(self._mark_dirty_processes)
        self.viewport.status_provider = self._viewport_sim_status
        self._sync_sim_fields()
        self._update_undo_actions()
        self._on_sim_state("idle")
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
        a["coord.add_frame"].setEnabled(has)
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
        self.sim.abandon()
        self.scene.clear()
        self.undo.clear()
        mesh_assets.clear()
        self.library.reload_meshes()
        self.library_panel.rebuild()
        self.project = ser.ProjectData()
        self.units.load_dict({})
        self._apply_view_settings(self.project.view)
        self._sync_sim_fields()
        self.plot_workspace.clear_plots(emit=False)
        self.recording_panel.reload_settings()
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
        self.sim.abandon()
        self.scene.clear()
        self.undo.clear()
        data = ser.load_project_dict(raw, self.scene, self.units)
        self.project = data
        self._apply_view_settings(data.view)
        self._sync_sim_fields()
        self.library.reload_meshes()
        self.library_panel.rebuild()
        if data.ui.get("layout"):
            self.restoreState(QByteArray.fromBase64(data.ui["layout"].encode()))
        for w in data.warnings + self.plot_workspace.load_specs(data.plots):
            self.bus.warning(w, "project")
        self.recording_panel.reload_settings()
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
                        ("view.stats", "show_stats"), ("physics.show_colliders", "show_colliders"),
                        ("physics.show_joints", "show_joints"), ("physics.show_contacts", "show_contacts")):
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
        self.project.plots = self.plot_workspace.specs()
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
            self.project.plots = self.plot_workspace.specs()
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
        if item is None or not self._require_idle("Adding objects"):
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
        if not self._require_idle("Import"):
            return
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
    # simulation & processes
    # ======================================================================================
    def _require_idle(self, what: str) -> bool:
        """Structural edits are blocked while a simulation session owns the scene state."""
        if self.sim.session_active:
            msg = f"{what} is not available during a simulation. Stop or reset the simulation first."
            self.statusBar().showMessage(msg, 5000)
            self.bus.warning(msg, "simulation")
            return False
        return True

    def _undo(self) -> None:
        if self._require_idle("Undo"):
            self.undo.undo()

    def _redo(self) -> None:
        if self._require_idle("Redo"):
            self.undo.redo()

    def _update_undo_actions(self, *_args) -> None:
        idle = not self.sim.session_active
        self.act_undo.setEnabled(idle and self.undo.canUndo())
        self.act_redo.setEnabled(idle and self.undo.canRedo())
        self.act_undo.setText(f"&Undo {self.undo.undoText()}".strip())
        self.act_redo.setText(f"&Redo {self.undo.redoText()}".strip())

    def _sim_start(self, step: bool = False) -> None:
        """Start/resume (or single-step); validation errors are collected and shown together."""
        self._start_errors: list[str] = []
        self._collecting_start_errors = True
        try:
            if step:
                self.sim.step()
                ok = self.sim.session_active
            else:
                ok = self.sim.start()
        finally:
            self._collecting_start_errors = False
        if not ok and self._start_errors:
            QMessageBox.warning(self, "Cannot start simulation", "\n\n".join(self._start_errors[-8:]))

    def _on_sim_error(self, msg: str) -> None:
        self.bus.error(msg, "simulation")
        if getattr(self, "_collecting_start_errors", False):
            self._start_errors.append(msg)
        elif self.sim.state == "error":
            QMessageBox.critical(self, "Simulation error", msg)

    def _sim_setting(self, setter, value: float) -> None:
        try:
            setter(value)
            errors = self.project.simulation.validate()
            if errors:
                raise ValueError("; ".join(errors))
        except ValueError as exc:
            self.bus.error(f"Invalid simulation setting: {exc}", "simulation")
        self._mark_dirty()
        self._sync_sim_fields()

    def _sync_sim_fields(self) -> None:
        s = self.project.simulation
        self.sim_fields.set_values(s.delta_time, s.update_rate_hz, s.duration)
        _set_checked(self.actions_["sim.afap"], s.pacing == PacingMode.AS_FAST_AS_POSSIBLE)

    def _set_afap(self, on: bool) -> None:
        self.sim.set_pacing(PacingMode.AS_FAST_AS_POSSIBLE if on else PacingMode.REAL_TIME)
        self._mark_dirty()

    def open_simulation_settings(self) -> None:
        dlg = SimulationSettingsDialog(self.project.simulation, self.sim.session_active, self)
        if dlg.exec() and dlg.result_settings is not None:
            new = dlg.result_settings
            old = self.project.simulation
            self.project.simulation = new
            self.recording_panel.reload_settings()
            if self.sim.session_active:
                for attr, setter in (("delta_time", self.sim.set_delta_time),
                                     ("update_rate_hz", self.sim.set_update_rate),
                                     ("duration", self.sim.set_duration), ("render_fps", self.sim.set_render_fps)):
                    if getattr(new, attr) != getattr(old, attr):
                        setter(getattr(new, attr))
                if new.pacing != old.pacing:
                    self.sim.set_pacing(new.pacing)
            self.viewport.set_continuous(self.viewport.continuous, new.render_fps)
            self._mark_dirty()
            self._sync_sim_fields()

    def save_simulation_state(self) -> None:
        from ..simulation.simulation_engine import SimulationError
        start = os.path.splitext(self.project_path or "simulation")[0] + ".psimstate"
        path, _ = QFileDialog.getSaveFileName(self, "Save Simulation State", start,
                                              "PSim simulation states (*.psimstate);;All files (*)")
        if not path:
            return
        try:
            self.sim.save_state(path)
        except (SimulationError, OSError, TimeoutError) as exc:
            QMessageBox.warning(self, "Save Simulation State", str(exc))

    def restore_simulation_state(self) -> None:
        from ..simulation.simulation_engine import SimulationError
        path, _ = QFileDialog.getOpenFileName(self, "Restore Simulation State",
                                              os.path.dirname(self.project_path or ""),
                                              "PSim simulation states (*.psimstate);;All files (*)")
        if not path:
            return
        try:
            self.sim.restore_state(path)
        except (SimulationError, OSError, TimeoutError) as exc:
            QMessageBox.warning(self, "Restore Simulation State", str(exc))

    def open_stop_conditions(self) -> None:
        from .dialogs.stop_conditions_dialog import StopConditionsDialog
        dlg = StopConditionsDialog(self.scene, self.project.simulation.stop_conditions, self)
        if dlg.exec() and dlg.result_conditions is not None \
                and dlg.result_conditions != self.project.simulation.stop_conditions:
            self.project.simulation.stop_conditions = dlg.result_conditions
            self._mark_dirty()
            self.sim.refresh_plan()   # a running simulation uses the new conditions from the next step

    def _on_sim_state(self, state: str) -> None:
        a = self.actions_
        active = self.sim.session_active
        running = state == "running"
        a["sim.start"].setEnabled(not running)
        a["sim.start"].setText("&Resume" if state in ("paused",) else "&Start")
        a["sim.pause"].setEnabled(running)
        a["sim.stop"].setEnabled(active)
        a["sim.reset"].setEnabled(self.sim.can_reset)
        a["sim.step"].setEnabled(state in ("idle", "paused"))
        a["sim.save_state"].setEnabled(active)
        for k in ("edit.cut", "edit.paste", "edit.duplicate", "edit.delete", "scene.group", "scene.ungroup",
                  "scene.unparent", "file.import_obj"):
            if active:
                a[k].setEnabled(False)
        if not active:
            self._update_actions()
        self._update_undo_actions()
        self.inspector.set_simulated_ids(self.sim.body_ids)
        if running:
            self.viewport.set_continuous(True, self.project.simulation.render_fps)
        elif not self.actions_["view.continuous"].isChecked():
            self.viewport.set_continuous(False)
        if state == "idle":
            self.sim_fields.show_idle()
        self.viewport.update()

    def _on_sim_stats(self, st) -> None:
        self.sim_fields.show_stats(st)
        if self.bottom_tabs.currentWidget() is self.performance:
            self.performance.update_sim_stats(st)

    def _viewport_sim_status(self) -> str:
        st = self.sim.last_stats
        if not self.sim.session_active or st is None:
            return ""
        return f"Simulation {st.state}  ·  t = {st.time:.4f} s  ·  step {st.steps:,}"

    # ---- recording & trajectories ------------------------------------------------------------------------------
    def _trajectory_paths(self) -> list:
        return self.trajectories.build(self.scene, self.sim.recording)

    def _on_recording_changed(self) -> None:
        self.trajectories.reset()
        self.viewport.update()
        self.plot_workspace.refresh(force=True)
        self.recording_panel.refresh(force=True)

    def show_plots(self) -> None:
        self.dock_plots.show()
        self.dock_plots.raise_()

    def show_recording_panel(self) -> None:
        self.dock_bottom.show()
        self.dock_bottom.raise_()
        self.bottom_tabs.setCurrentWidget(self.recording_panel)

    def sim_clear_recording(self) -> None:
        self.sim.clear_recording()

    def _require_recording(self, title: str):
        rec = self.sim.recording
        if rec is None or not rec.n:
            QMessageBox.information(self, title, "Nothing recorded yet: run the simulation with data recording "
                                    "enabled (Plots ▸ Data Recording Panel).")
            return None
        return rec

    def _object_names(self) -> dict:
        rec = self.sim.recording
        base = dict(rec.meta.get("names", {})) if rec is not None else {}
        return {**base, **{o.id: o.name for o in self.scene.objects.values()}}

    def export_recorded_csv(self) -> None:
        from ..recording.export import export_csv
        from ..recording.variables import Evaluator, Variable, VariableError, info, object_quantities
        rec = self._require_recording("Export CSV")
        if rec is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Recorded Data", "recording.csv", "CSV (*.csv)")
        if not path:
            return
        variables = []
        for oid in [k[0] for k in rec.keys() if len(k) == 2 and k[1] == "origin"]:
            for q in object_quantities(rec, oid):
                if q in ("mass", "distance", "displacement", "speed", "momentum", "angular_momentum"):
                    continue
                var = Variable(oid, q)
                comps = ("x", "y", "z") if info(var).vector else ("",)
                variables += [Variable(oid, q, c) for c in comps]
        try:
            n = export_csv(path, Evaluator(rec, self.scene), variables, self.units, self._object_names())
        except (OSError, VariableError) as exc:
            QMessageBox.warning(self, "Export CSV", str(exc))
            return
        self.bus.info(f"Exported {n} samples × {len(variables) + 1} columns to {path}", "recording")

    def export_recorded_json(self) -> None:
        from ..recording.export import export_json
        rec = self._require_recording("Export JSON")
        if rec is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Recorded Data", "recording.json", "JSON (*.json)")
        if not path:
            return
        try:
            export_json(path, rec, self._object_names())
        except OSError as exc:
            QMessageBox.warning(self, "Export JSON", str(exc))
            return
        self.bus.info(f"Exported the recording ({rec.n} samples, SI units) to {path}", "recording")

    def generate_report(self) -> None:
        from ..recording.export import generate_report
        rec = self._require_recording("Simulation Report")
        if rec is None:
            return
        name = self.project.metadata.get("name") or (os.path.splitext(os.path.basename(self.project_path))[0]
                                                     if self.project_path else "Simulation")
        path, _ = QFileDialog.getSaveFileName(self, "Save Simulation Report", f"{name} report.html", "HTML (*.html)")
        if not path:
            return
        try:
            generate_report(path, rec, self.scene, self.units, f"{name} — simulation report",
                            self.plot_workspace.images())
        except OSError as exc:
            QMessageBox.warning(self, "Simulation Report", str(exc))
            return
        self.bus.info(f"Report written to {path}", "recording")

    def clear_trajectory(self, ids: list[str] | None = None) -> None:
        for oid in ids if ids is not None else self.scene.selection:
            self.trajectories.clear(oid, self.sim.recording)
        self.viewport.update()

    def export_trajectory(self, oid: str) -> None:
        from ..recording.export import export_trajectory_csv
        from ..recording.variables import Evaluator, VariableError
        rec = self.sim.recording
        if rec is None or not rec.has((oid, "origin")):
            QMessageBox.information(self, "Export Trajectory", "No recorded path for this object yet: run the "
                                    "simulation with data recording enabled (Simulation ▸ Simulation Settings).")
            return
        obj = self.scene.get(oid)
        path, _ = QFileDialog.getSaveFileName(self, "Export Trajectory", f"{obj.name}_trajectory.csv", "CSV (*.csv)")
        if not path:
            return
        point = "position" if obj.trajectory_setting("point") == "origin" else "com"
        try:
            n = export_trajectory_csv(path, Evaluator(rec, self.scene), oid, obj.trajectory_setting("frame") or None,
                                      point, self.units)
        except (OSError, VariableError) as exc:
            QMessageBox.warning(self, "Export Trajectory", str(exc))
            return
        self.bus.info(f"Exported {n} trajectory points of {obj.name} to {path}", "recording")

    def _mark_dirty_processes(self) -> None:
        self._update_title()

    def _show_processes(self) -> None:
        self.dock_bottom.show()
        self.bottom_tabs.setCurrentWidget(self.processes_panel)

    def assign_process(self, ids) -> None:
        dlg = AssignProcessDialog(self.scene, self.units, preset_ids=list(ids or []), parent=self)
        if dlg.exec() and dlg.result:
            r = dlg.result
            a = self.editor.assign_process(r["process_id"], r["targets"], r["params"], r["label"])
            self.bus.info(f"Assigned {a.uid} ({a.process_id}) to {a.targets.describe()}", "processes")
            self._report_plan(quiet=True)

    def edit_assignment(self, uid: str) -> None:
        a = self.scene.assignment(uid)
        if a is None:
            return
        dlg = AssignProcessDialog(self.scene, self.units, assignment=a, parent=self)
        if dlg.exec() and dlg.result:
            r = dlg.result
            self.editor.update_assignment(uid, targets=r["targets"], params=r["params"], label=r["label"])
            self._report_plan(quiet=True)

    def _report_plan(self, quiet: bool = False):
        plan = self.sim.validate()
        for d in plan.diagnostics:
            {"error": self.bus.error, "warning": self.bus.warning}.get(d.level, self.bus.info)(d.message, "processes")
        if not quiet and not plan.diagnostics:
            self.bus.info(f"Process plan valid: {len(plan.body_ids)} bodies, "
                          f"{len(plan.contributions) + len(plan.prescribed)} active assignments", "processes")
        return plan

    def validate_processes(self) -> None:
        plan = self._report_plan()
        if plan.errors:
            QMessageBox.warning(self, "Process validation", "\n\n".join(d.message for d in plan.errors))
        else:
            self.statusBar().showMessage("Process plan is valid", 3000)

    def show_dependencies(self) -> None:
        DependencyDialog(self.sim.validate(), self.scene, self).exec()

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
        if not self._require_idle("Cut"):
            return
        self.copy()
        self.delete_selection()

    def paste(self) -> None:
        if not self._require_idle("Paste"):
            return
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
        if not self._require_idle("Duplicate"):
            return
        self.editor.duplicate(self.scene.selection)

    def delete_selection(self) -> None:
        if not self._require_idle("Delete"):
            return
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
        if not self._require_idle("Group"):
            return
        g = self.editor.group(self.scene.selection)
        if g is not None:
            self.bus.info(f"Grouped into {g.name} ({g.id})", "edit")

    def ungroup(self) -> None:
        if not self._require_idle("Ungroup"):
            return
        self.editor.ungroup(self.scene.selection)

    def _reparent(self, ids, parent_id) -> None:
        if not self._require_idle("Reparenting"):
            return
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

    def place_relative(self) -> None:
        from .dialogs.placement_dialog import PlaceRelativeDialog
        if not self._require_idle("Placing objects"):
            return
        dlg = PlaceRelativeDialog(self.scene, self.editor, self.units, self.scene.selection, self)
        dlg.exec()

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
            m.addActions([a["obj.place_relative"], a["obj.hide"], a["obj.lock"]])
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

    # ---- developer mode & formula processes ----------------------------------------------------------------
    def toggle_developer_mode(self) -> None:
        from ..processes.permissions import access_controller
        from .dialogs.developer_dialogs import DeveloperModeDialog
        access = access_controller()
        if self.developer_token is not None:
            access.lock()
            self.developer_token = None
            self.bus.info("Developer mode locked", "permissions")
        else:
            dlg = DeveloperModeDialog(access, self)
            if dlg.exec() and dlg.token is not None:
                self.developer_token = dlg.token
                self.bus.info(f"Developer mode unlocked (signing key {dlg.token.key_id})", "permissions")
        on = self.developer_token is not None
        self.developer_menu.menuAction().setVisible(on)
        self.actions_["tools.developer"].setText("Lock &Developer Mode" if on else "&Developer Mode…")
        self.actions_["tools.developer"].setIcon(icons.icon("unlock" if on else "lock"))

    def open_process_editor(self, spec) -> None:
        from ..processes.permissions import access_controller
        from ..processes.process_registry import process_registry
        from .dialogs.developer_dialogs import ProcessEditorDialog
        try:
            dlg = ProcessEditorDialog(process_registry(), access_controller(), self.developer_token, self.units, spec,
                                      self)
        except PermissionError as exc:
            QMessageBox.warning(self, "Developer mode", str(exc))
            return
        if dlg.exec() and dlg.published is not None:
            p = dlg.published
            self.bus.info(f"Published {p.name} {p.version} ({p.process_id})", "processes")
            self.processes_panel.rebuild()
            self.scene.processesChanged.emit()

    def edit_formula_process(self) -> None:
        from ..processes.formula_process import FormulaProcessSpec
        from ..processes.process_registry import process_registry
        formulas = [d for d in process_registry().definitions() if d.is_formula]
        if not formulas:
            return
        labels = [f"{d.name} ({d.process_id} {d.version})" for d in formulas]
        choice, ok = QInputDialog.getItem(self, "Edit Formula Process", "Start a new version from:", labels, 0, False)
        if not ok:
            return
        d = formulas[labels.index(choice)]
        spec = FormulaProcessSpec.from_dict(d.formula_spec.to_dict())
        parts = [int(x) for x in spec.version.split(".")] + [0] * (3 - len(spec.version.split(".")))
        parts[-1] += 1
        spec.version = ".".join(str(x) for x in parts)
        if process_registry().sources.get(d.process_id) == "library":
            spec.process_id = "user." + spec.process_id.split(".", 1)[1]  # library ids are reserved
        self.open_process_editor(spec)

    def withdraw_process(self) -> None:
        from ..processes.process_registry import process_registry
        reg = process_registry()
        pkgs = [d for d in reg.definitions() if reg.sources.get(d.process_id) not in ("built-in", "library")]
        if not pkgs:
            QMessageBox.information(self, "Withdraw", "No published packages are installed.")
            return
        labels = [f"{d.name} ({d.process_id} {d.version})" for d in pkgs]
        choice, ok = QInputDialog.getItem(self, "Withdraw Process", "Process:", labels, 0, False)
        if not ok:
            return
        d = pkgs[labels.index(choice)]
        try:
            reg.withdraw(d.process_id, self.developer_token)
        except (PermissionError, ProcessError) as exc:
            QMessageBox.warning(self, "Withdraw", str(exc))
            return
        self.bus.info(f"Withdrew {d.process_id}; assignments that use it are reported as errors", "processes")
        self.processes_panel.rebuild()

    # ---- reference frames ------------------------------------------------------------------------------------
    def open_joints(self, body: str | None = None) -> None:
        from .dialogs.joints_dialog import JointsDialog
        JointsDialog(self.scene, self.editor, self.units, body=body if isinstance(body, str) else None,
                     parent=self).exec()

    def open_reference_frames(self, select: str | None = None) -> None:
        from .dialogs.frames_dialog import ReferenceFramesDialog
        ReferenceFramesDialog(self.scene, self.editor, self.units, select if isinstance(select, str) else None,
                              self).exec()

    def add_frame_at_selection(self) -> None:
        o = self.scene.primary
        if o is None:
            return
        names = {f.name.lower() for f in self.scene.frames}
        base = f"{o.name} frame"
        name, k = base, 2
        while name.lower() in names:
            name, k = f"{base} {k}", k + 1
        fid = self.editor.add_frame(name, o.id)
        self.statusBar().showMessage(f"Added reference frame '{name}' ({fid}) attached to {o.name}", 4000)
        self.open_reference_frames(fid)

    def open_unit_inspector(self) -> None:
        from ..processes.process_registry import process_registry
        from .dialogs.unit_inspector_dialog import FormulaUnitInspector
        FormulaUnitInspector(process_registry(), self.units, None, self).exec()

    def show_measurements(self) -> None:
        self.dock_bottom.show()
        self.bottom_tabs.setCurrentWidget(self.measurements_panel)

    def measure(self, kind: str) -> None:
        from ..scene.measurements import MeasurementError
        if self.measurements_panel.at.currentData() == "point":
            self.measure_points(kind)
            return
        ids = [i for i in self.scene.selection if i in self.scene.objects]
        try:
            self.editor.add_measurement(kind, ids, self.measurements_panel.at.currentData())
        except (MeasurementError, SceneError) as exc:
            self.statusBar().showMessage(str(exc), 5000)
            self.bus.error(str(exc), "measurements")
            return
        self.show_measurements()

    def measure_points(self, kind: str) -> None:
        """Start picking surface points in the viewport for a new measurement."""
        self.viewport.begin_point_pick(kind)

    def _points_picked(self, kind: str, points: list) -> None:
        from ..scene.measurements import MeasurementError
        try:
            mid = self.editor.add_measurement(kind, [], points=points)
        except (MeasurementError, SceneError) as exc:
            self.bus.error(str(exc), "measurements")
            return
        m = next(x for x in self.scene.measurements if x.id == mid)
        self.statusBar().showMessage(f"Added {m.name}", 4000)
        self.show_measurements()

    def show_relative_motion(self) -> None:
        self.dock_bottom.show()
        self.bottom_tabs.setCurrentWidget(self.relative_motion)

    def edit_global_parameters(self) -> None:
        from .dialogs.process_dialogs import GlobalParametersDialog
        dlg = GlobalParametersDialog(self.scene, self.units, self)
        if dlg.exec() and dlg.result_params is not None:
            self.editor.set_global_parameters(dlg.result_params)

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

    def show_system_info(self) -> None:
        from .dialogs.system_info_dialog import SystemInfoDialog
        SystemInfoDialog(self.viewport.renderer.gl_info, self).exec()

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
        if self.sim.session_active:
            self.sim.stop()
        if self.developer_token is not None:
            from ..processes.permissions import access_controller
            access_controller().lock()
            self.developer_token = None
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
