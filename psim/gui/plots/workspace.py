"""Plot Workspace: tabs of 2D/3D plots of recorded data, live updates, synchronized time cursors, export."""
from __future__ import annotations

import logging

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (QFileDialog, QLabel, QMenu, QMessageBox, QTabWidget, QToolBar, QToolButton,
                               QVBoxLayout, QWidget)

from ...recording.export import export_csv
from ...recording.plot_spec import PlotSpec, PlotSpecError, axis_title, quick_plot, series_data
from ...recording.recording import Recording
from ...recording.variables import Evaluator, VariableError
from ...scene.scene_manager import Scene
from ...units.unit_registry import UnitRegistry
from .. import icons
from .plot_editor import PlotEditorDialog
from .plot_views import Plot2DView, Plot3DView, PlotView

log = logging.getLogger(__name__)

QUICK_PLOTS = (("position", "Position vs time"), ("velocity", "Velocity vs time"),
               ("acceleration", "Acceleration vs time"), ("angular", "Angular velocity vs time"),
               ("speed", "Speed vs time"), ("energy", "Kinetic energy vs time"),
               ("force_displacement", "Net force vs displacement"), ("trajectory3d", "3D trajectories"))


class PlotWorkspace(QWidget):
    changed = Signal()   # plot definitions changed (project becomes dirty)

    def __init__(self, scene: Scene, units: UnitRegistry, recording_getter, parent=None) -> None:
        super().__init__(parent)
        self.scene, self.units, self._recording = scene, units, recording_getter
        self.plots: list[tuple[PlotSpec, PlotView]] = []
        self._version = None
        self._cursor_time: float | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        tb = QToolBar()
        tb.addAction(icons.icon("chart"), "New 2D Plot", lambda: self.new_plot("2d"))
        tb.addAction(icons.icon("cube"), "New 3D Plot", lambda: self.new_plot("3d"))
        quick = QToolButton()
        quick.setText("Quick Plot")
        quick.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(quick)
        for key, text in QUICK_PLOTS:
            menu.addAction(text, lambda k=key: self.add_quick(k))
        quick.setMenu(menu)
        quick.setToolTip("Ready-made plots of the selected objects")
        tb.addWidget(quick)
        tb.addSeparator()
        tb.addAction(icons.icon("edit"), "Edit Plot…", self.edit_current)
        tb.addAction(icons.icon("export"), "Export Image…", self.export_image)
        tb.addAction(icons.icon("table"), "Export Data…", self.export_data)
        tb.addSeparator()
        self.live = tb.addAction(icons.icon("activity"), "Live Update")
        self.live.setCheckable(True)
        self.live.setChecked(True)
        self.live.setToolTip("Refresh the plots while the simulation records new samples")
        self.sync = tb.addAction(icons.icon("target"), "Sync Cursors")
        self.sync.setCheckable(True)
        self.sync.setChecked(True)
        self.sync.setToolTip("Move the time cursor of every time plot together (3D plots mark the same instant)")
        lay.addWidget(tb)
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self.remove_plot)
        self.tabs.tabBarDoubleClicked.connect(lambda _i: self.edit_current())
        lay.addWidget(self.tabs, 1)
        self.status = QLabel()
        self.status.setObjectName("Muted")
        lay.addWidget(self.status)
        self.empty = QLabel("No plots yet. Run a simulation (data recording is on by default), select objects and "
                            "use Quick Plot, or create a plot with New 2D / 3D Plot.")
        self.empty.setWordWrap(True)
        self.empty.setObjectName("Muted")
        lay.addWidget(self.empty)
        self._timer = QTimer(self)
        self._timer.setInterval(200)  # 5 Hz live updates
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self._update_empty()

    @property
    def recording(self) -> Recording | None:
        return self._recording()

    # ---- definitions -----------------------------------------------------------------------------------
    def specs(self) -> list[dict]:
        order = [self.tabs.widget(i) for i in range(self.tabs.count())]
        by_view = {id(v): s for s, v in self.plots}
        return [by_view[id(v)].to_dict() for v in order if id(v) in by_view]

    def load_specs(self, data: list) -> list[str]:
        """Replace all plots. Returns warnings for definitions that could not be read."""
        warnings = []
        self.clear_plots(emit=False)
        for d in data or []:
            try:
                self.add_plot(PlotSpec.from_dict(d), emit=False)
            except PlotSpecError as exc:
                warnings.append(f"Plot skipped: {exc}")
        return warnings

    def clear_plots(self, emit: bool = True) -> None:
        while self.plots:
            self._remove_index(0)
        if emit:
            self.changed.emit()

    def add_plot(self, spec: PlotSpec, emit: bool = True) -> PlotView:
        view: PlotView = Plot3DView(spec) if spec.kind == "3d" else Plot2DView(spec)
        view.cursorMoved.connect(self._cursor_moved)
        self.plots.append((spec, view))
        self.tabs.addTab(view, spec.title)
        self.tabs.setCurrentWidget(view)
        self._update_empty()
        self.refresh(force=True)
        if emit:
            self.changed.emit()
        return view

    def new_plot(self, kind: str) -> None:
        dlg = PlotEditorDialog(self.scene, self.recording, PlotSpec("Plot", kind, []), self)
        dlg._add()
        if dlg.exec() and dlg.result_spec is not None:
            self.add_plot(dlg.result_spec)

    def add_quick(self, kind: str) -> None:
        ids = [i for i in self.scene.selection if i in self.scene.objects]
        rec = self.recording
        if not ids and rec is not None:
            ids = [k[0] for k in rec.keys() if len(k) == 2 and k[1] == "origin"]
        if not ids:
            QMessageBox.information(self, "Quick Plot", "Select one or more objects first.")
            return
        spec = quick_plot(kind, ids, {o.id: o.name for o in self.scene.objects.values()})
        if len(ids) == 1:
            spec.title = f"{self.scene.get(ids[0]).name}: {spec.title}"
        self.add_plot(spec)

    def _current_index(self) -> int:
        w = self.tabs.currentWidget()
        return next((i for i, (_s, v) in enumerate(self.plots) if v is w), -1)

    def edit_current(self) -> None:
        i = self._current_index()
        if i < 0:
            return
        spec, view = self.plots[i]
        dlg = PlotEditorDialog(self.scene, self.recording, spec, self)
        if not dlg.exec() or dlg.result_spec is None:
            return
        new = dlg.result_spec
        if new.kind != spec.kind:
            tab = self.tabs.indexOf(view)
            self._remove_index(i)
            self.add_plot(new, emit=False)
            self.tabs.tabBar().moveTab(self.tabs.indexOf(self.plots[-1][1]), tab)
        else:
            self.plots[i] = (new, view)
            view.apply_spec(new)
            self.tabs.setTabText(self.tabs.indexOf(view), new.title)
            self.refresh(force=True)
        self.changed.emit()

    def remove_plot(self, tab_index: int) -> None:
        w = self.tabs.widget(tab_index)
        i = next((k for k, (_s, v) in enumerate(self.plots) if v is w), -1)
        if i >= 0:
            self._remove_index(i)
            self.changed.emit()

    def _remove_index(self, i: int) -> None:
        _spec, view = self.plots.pop(i)
        self.tabs.removeTab(self.tabs.indexOf(view))
        view.deleteLater()
        self._update_empty()

    def _update_empty(self) -> None:
        self.empty.setVisible(not self.plots)
        self.tabs.setVisible(bool(self.plots))

    # ---- data ---------------------------------------------------------------------------------------------
    def _context(self) -> dict:
        names = {o.id: o.name for o in self.scene.objects.values()}
        rec = self.recording
        if rec is not None:
            names = {**rec.meta.get("names", {}), **names}
        frame_names = {f.id: f.name for f in self.scene.frames}
        frame_names.update({o.id: o.name for o in self.scene.objects.values()})
        custom = {}
        for o in self.scene.objects.values():
            for k, cp in o.custom.items():
                custom[(o.id, k)] = (cp.definition.quantity, cp.definition.vtype.value == "vec3")
        return {"names": names, "frame_names": frame_names, "custom_quantities": custom}

    def refresh(self, force: bool = False) -> None:
        rec = self.recording
        version = (id(rec), rec.version if rec is not None else None)
        if not force and (version == self._version or not self.live.isChecked() or not self.isVisible()):
            return
        self._version = version
        ctx = self._context()
        ev = Evaluator(rec, self.scene) if rec is not None and rec.n else None
        for spec, view in self.plots:
            titles = {a: axis_title(spec, a, self.units, **ctx) for a in ("x", "y", "z")}
            try:
                data = series_data(ev, spec, self.units, **ctx) if ev is not None else []
            except VariableError as exc:  # pragma: no cover - series_data reports per series
                log.warning("Plot '%s': %s", spec.title, exc)
                data = []
            view.set_data(data, titles)
            view.set_cursor(self._cursor_time)
        self.status.setText(rec.status_text() if rec is not None else
                            "No recording: run the simulation with data recording enabled")

    def _cursor_moved(self, t: float) -> None:
        self._cursor_time = t
        if self.sync.isChecked():
            sender = self.sender()
            for _spec, view in self.plots:
                if view is not sender:
                    view.set_cursor(t)

    # ---- export ---------------------------------------------------------------------------------------------
    def export_image(self) -> None:
        i = self._current_index()
        if i < 0:
            return
        spec, view = self.plots[i]
        path, _ = QFileDialog.getSaveFileName(self, "Export Plot Image", f"{spec.title}.png", "PNG image (*.png)")
        if path:
            view.export_image(path)

    def export_data(self) -> None:
        i = self._current_index()
        rec = self.recording
        if i < 0 or rec is None or not rec.n:
            return
        spec, _view = self.plots[i]
        path, _ = QFileDialog.getSaveFileName(self, "Export Plot Data", f"{spec.title}.csv", "CSV (*.csv)")
        if not path:
            return
        variables = []
        for s in spec.series:
            for v in (s.x, s.y, s.z):
                if v is not None and v not in variables:
                    variables.append(v)
        ctx = self._context()
        try:
            export_csv(path, Evaluator(rec, self.scene), variables, self.units, ctx["names"], ctx["frame_names"],
                       ctx["custom_quantities"])
        except (OSError, VariableError) as exc:
            QMessageBox.warning(self, "Export Plot Data", str(exc))

    def images(self) -> list[tuple[str, bytes]]:
        """PNG snapshots of every plot (for the simulation report)."""
        self.refresh(force=True)
        return [(spec.title, view.grab_png()) for spec, view in self.plots]
