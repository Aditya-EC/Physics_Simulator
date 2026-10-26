"""Phase 8 GUI: stop-conditions dialog, simulation-state actions, measurements, performance profile."""
from __future__ import annotations

from PySide6.QtCore import Qt


from test_coordinates_gui import owned  # noqa: F401  (fixture)


def test_stop_conditions_dialog_checks_and_returns_conditions(scene, owned):  # noqa: F811
    from psim.gui.dialogs.stop_conditions_dialog import StopConditionsDialog
    s, e = scene
    ball = e.add_new("body.point_mass", position=[0, 0, 3])
    dlg = StopConditionsDialog(s, [{"name": "Ground", "expression": f'object("{ball.id}").position.z < 0[m]',
                                    "action": "stop", "enabled": True}])
    owned(dlg)
    assert dlg.check() and "1 enabled condition checked" in dlg.status.text()
    dlg.add_row()
    dlg.table.item(1, 2).setText(f'object("{ball.id}").position.z < 2[s]')
    assert not dlg.check() and "Incompatible dimensions" in dlg.status.text()
    dlg.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)         # disabled: not compiled
    assert dlg.check()
    dlg.table.item(1, 1).setText("ground")                               # duplicate name (case-insensitive)
    assert not dlg.check() and "Duplicate" in dlg.status.text()
    dlg.table.selectRow(1)
    dlg._remove()
    dlg._accept()
    assert dlg.result_conditions == [{"name": "Ground", "expression": f'object("{ball.id}").position.z < 0[m]',
                                      "action": "stop", "enabled": True}]


def test_performance_panel_shows_the_stage_profile(qapp, owned):  # noqa: F811
    from psim.gui.bottom_panels import PerformancePanel
    from psim.simulation.runner import RunStats
    p = PerformancePanel()
    owned(p)
    prof = {"updates": 0.0, "processes_and_integration": 0.25, "prescribed": 0.0, "constraints": 1.5,
            "recording": 0.1, "conditions": 0.01, "total": 1.86, "steps": 10}
    p.update_sim_stats(RunStats("running", 0.1, 10, 0.01, 1.0, "afap", 100, 900, 9, 0.01, 0, False, 1.9, prof))
    text = p.labels["profile"].text()
    assert "constraints 1.500" in text and "total 1.860" in text and "updates" not in text
