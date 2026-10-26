"""Plot definitions, the plot workspace (2D), the plot editor and the recording panel."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from psim.processes.process_base import TargetFilter
from psim.recording.plot_spec import (PlotSpec, PlotSpecError, SeriesSpec, axis_title, quick_plot, series_data,
                                      values_at_time)
from psim.recording.recording import Recording
from psim.recording.variables import Evaluator, Variable
from psim.simulation.settings import SimulationSettings
from psim.units.unit_registry import UnitRegistry

from test_recording import make_engine, run


@pytest.fixture
def owned(qapp):
    import shiboken6
    widgets = []
    yield widgets.append
    for w in widgets:
        if shiboken6.isValid(w):
            shiboken6.delete(w)


@pytest.fixture
def recorded(scene):
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 10])
    ball.name = "Ball"
    ball.linear_velocity = [2.0, 0, 5.0]
    e.assign_process("dyn.gravity", TargetFilter(ids=[ball.id]))
    rec = Recording(interval_steps=2)
    run(make_engine(scene, rec, delta_time=0.01, duration=1.0, integrator="rk4"))
    return s, e, s.get(ball.id), rec


def test_plot_spec_round_trip_and_validation():
    spec = quick_plot("velocity", ["OBJ-000001", "OBJ-000002"])
    assert len(spec.series) == 6 and spec.x_is_time
    again = PlotSpec.from_dict(spec.to_dict())
    assert again.to_dict() == spec.to_dict()
    traj = quick_plot("trajectory3d", ["OBJ-000001"], {"OBJ-000001": "Ball"})
    assert traj.kind == "3d" and traj.series[0].label == "Ball" and not traj.x_is_time
    bad = spec.to_dict()
    bad["series"][0]["style"] = "wiggly"
    with pytest.raises(PlotSpecError):
        PlotSpec.from_dict(bad)
    with pytest.raises(PlotSpecError, match="3D plots need"):
        PlotSpec.from_dict({**spec.to_dict(), "kind": "3d"})
    with pytest.raises(PlotSpecError, match="limits"):
        PlotSpec.from_dict({**spec.to_dict(), "x_limits": [2, 1]})


def test_series_data_in_display_units(recorded):
    s, e, ball, rec = recorded
    units = UnitRegistry()
    units.set_preferred_unit("velocity", "km/h")
    spec = PlotSpec("v", "2d", [SeriesSpec(Variable.time(), Variable(ball.id, "velocity", "x")),
                                SeriesSpec(Variable.time(), Variable("OBJ-999999", "velocity", "x"))])
    data = series_data(Evaluator(rec, s), spec, units, {ball.id: "Ball"})
    np.testing.assert_allclose(data[0].y, 7.2)
    assert data[0].label == "Ball: Velocity x" and len(data[0].x) == rec.n
    assert data[1].error and "not recorded" in data[1].error
    assert axis_title(spec, "y", units) == "Velocity x [km/h]"
    assert axis_title(spec, "x", units) == "Time [s]"
    spec.series = spec.series[:1]
    assert axis_title(spec, "y", units, {ball.id: "Ball"}) == "Ball: Velocity x [km/h]"
    (name, x, y), = values_at_time(data, 0.503)
    assert x == pytest.approx(0.5) and y == pytest.approx(7.2)


def test_plot_editor_builds_specs(recorded, owned):
    from psim.gui.plots.plot_editor import PlotEditorDialog
    s, e, ball, rec = recorded
    s.set_selection([ball.id])
    dlg = PlotEditorDialog(s, rec)
    owned(dlg)
    dlg._add()
    ed = dlg.axis_editors["y"]
    ed.quantity.setCurrentIndex(ed.quantity.findData("acceleration"))
    ed.component.setCurrentIndex(ed.component.findData("z"))
    dlg.title.setText("Fall")
    dlg._accept()
    spec = dlg.result_spec
    assert spec.title == "Fall" and spec.series[0].y == Variable(ball.id, "acceleration", "z")
    assert spec.series[0].x == Variable.time()
    dlg.kind.setCurrentIndex(dlg.kind.findData("3d"))
    dlg._accept()
    assert dlg.result_spec.kind == "3d" and dlg.result_spec.series[0].z is not None
    empty = PlotEditorDialog(s, None)
    owned(empty)
    empty._accept()
    assert empty.result_spec is None and "at least one series" in empty.error.text()


def test_workspace_live_plots_cursors_and_export(recorded, owned, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from psim.gui.plots.workspace import PlotWorkspace
    s, e, ball, rec = recorded
    current = {"rec": rec}
    ws = PlotWorkspace(s, UnitRegistry(), lambda: current["rec"])
    owned(ws)
    ws.show()
    s.set_selection([ball.id])
    ws.add_quick("position")
    ws.add_quick("speed")
    (spec1, view1), (spec2, view2) = ws.plots
    xs, ys = view1.curves[2].getData()
    assert len(xs) == rec.n and ys[0] == pytest.approx(10.0)
    # synchronized cursors
    view1.cursor.setValue(0.5)
    assert view2.cursor.value() == pytest.approx(0.5)
    assert "t = 0.5" in view2.readout.toPlainText()
    # definitions round-trip (project save / load)
    saved = ws.specs()
    ws.load_specs(saved)
    assert [p[0].to_dict() for p in ws.plots] == saved
    assert ws.load_specs([{"kind": "bogus"}]) and ws.plots == []
    ws.load_specs(saved)
    # export the current plot's data
    out = tmp_path / "plot.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    ws.tabs.setCurrentIndex(0)
    ws.export_data()
    rows = list(csv.reader(open(out, encoding="utf-8")))
    assert rows[0][0] == "time [s]" and len(rows) == rec.n + 1
    pngs = ws.images()
    assert len(pngs) == 2 and pngs[0][1].startswith(b"\x89PNG")
    # live update: new recording -> new data
    current["rec"] = None
    ws.refresh(force=True)
    assert "No recording" in ws.status.text()


def test_recording_panel(recorded, owned):
    from psim.gui.recording_panel import RecordingPanel
    s, e, ball, rec = recorded
    rec.log(0.2, 20, "warning", "watch out")
    settings = SimulationSettings()
    panel = RecordingPanel(s, UnitRegistry(), lambda: rec, lambda: settings)
    owned(panel)
    panel.show()
    intervals = []
    panel.intervalChanged.connect(intervals.append)
    panel.every.setValue(25)
    panel.memory.setValue(64)
    panel.overflow.setCurrentIndex(panel.overflow.findData("window"))
    panel.process_outputs.setChecked(False)
    assert intervals == [25] and settings.record_max_mb == 64 and settings.record_overflow == "window"
    assert settings.record_process_outputs is False
    panel.quantity.setCurrentIndex(panel.quantity.findData("speed"))
    panel.refresh(force=True)
    assert panel.stats.rowCount() == 1 and panel.stats.item(0, 0).text() == "Ball"
    assert float(panel.stats.item(0, 2).text()) == pytest.approx(np.hypot(2, 5))   # initial speed
    assert panel.events.rowCount() == 1 and panel.events.item(0, 3).text() == "watch out"


def test_plots_saved_in_project(scene):
    import json
    from psim.project import serialization as ser
    from psim.scene.scene_manager import Scene
    s, e = scene
    data = ser.ProjectData(plots=[quick_plot("speed", ["OBJ-000001"]).to_dict()])
    d = json.loads(json.dumps(ser.project_to_dict(s, UnitRegistry(), data)))
    back = ser.load_project_dict(d, Scene(), UnitRegistry())
    assert back.plots == data.plots and back.warnings == []
    d["plots"].append({"kind": "nope"})
    back = ser.load_project_dict(d, Scene(), UnitRegistry())
    assert len(back.plots) == 1 and "Plot #2 skipped" in back.warnings[0]
