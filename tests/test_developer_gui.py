"""Developer-mode dialogs, the process editor and the global-parameters dialog."""
from __future__ import annotations

import os

import pytest

from psim.processes import built_in_processes
from psim.processes.formula_process import FormulaProcessSpec
from psim.processes.permissions import AccessController, PermissionDenied
from psim.processes.process_registry import ProcessRegistry
from psim.units.unit_registry import UnitRegistry

GOOD = {"process_id": "lab.spring_anchor", "name": "Anchor Spring", "version": "1.0.0", "kind": "contribution",
        "requires_mass": True,
        "parameters": [{"key": "k", "type": "float", "quantity": "stiffness", "default": 10.0}],
        "script": "force = -param.k * self.position",
        "tests": [{"name": "restoring", "objects": {"self": {"position": [0.5, 0, 0]}}, "expect": {"force": [-5, 0, 0]}}]}


@pytest.fixture
def owned(qapp):
    """Destroy dialogs while the QApplication is alive (not during interpreter shutdown)."""
    import shiboken6
    widgets = []
    yield widgets.append
    for w in widgets:
        if shiboken6.isValid(w):
            shiboken6.delete(w)


@pytest.fixture
def dev(qapp, tmp_path):
    access = AccessController(str(tmp_path))
    access.create_credentials("correct horse battery", "Test Dev")
    reg = ProcessRegistry()
    built_in_processes.register_all(reg)
    reg.seal(access)
    return access, reg


def test_developer_mode_dialog_unlocks_only_with_passphrase(qapp, tmp_path, dev, owned):
    from psim.gui.dialogs.developer_dialogs import DeveloperModeDialog
    first = DeveloperModeDialog(AccessController(str(tmp_path / "none")))
    owned(first)
    assert not hasattr(first, "passphrase")
    access, _ = dev
    dlg = DeveloperModeDialog(access)
    owned(dlg)
    dlg.passphrase.setText("nope nope nope")
    dlg._unlock()
    assert dlg.token is None and "Incorrect" in dlg.error.text()
    dlg.passphrase.setText("correct horse battery")
    dlg._unlock()
    assert access.is_valid(dlg.token)


def test_process_editor_requires_token(dev):
    from psim.gui.dialogs.developer_dialogs import ProcessEditorDialog
    access, reg = dev
    with pytest.raises(PermissionDenied):
        ProcessEditorDialog(reg, access, None, UnitRegistry())


def test_process_editor_validate_test_and_publish(dev, monkeypatch, owned):
    from PySide6.QtWidgets import QMessageBox
    from psim.gui.dialogs.developer_dialogs import ProcessEditorDialog
    access, reg = dev
    token = access.unlock("correct horse battery")
    dlg = ProcessEditorDialog(reg, access, token, UnitRegistry(), FormulaProcessSpec.from_dict(GOOD))
    owned(dlg)
    got, want = dlg.collect().to_dict(), FormulaProcessSpec.from_dict({**GOOD, "author": "Test Dev"}).to_dict()
    assert {k: v for k, v in got.items() if k != "parameters"} == {k: v for k, v in want.items() if k != "parameters"}
    assert {k: v for k, v in got["parameters"][0].items() if v not in (None, "")} == {**GOOD["parameters"][0], "label": "k"}
    assert dlg.validate() and dlg.run_tests()
    dlg.script.setPlainText("let a = self.mass\nforce = a")
    assert not dlg.validate()
    assert "line 2" in dlg.results.toPlainText() and dlg.script.extraSelections()
    dlg.load_spec(FormulaProcessSpec.from_dict(GOOD))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    dlg.publish()
    assert dlg.published is reg.get("lab.spring_anchor")
    assert os.listdir(os.path.join(access.base, "processes"))


def test_global_parameters_dialog_converts_units(qapp, scene, owned):
    from psim.gui.dialogs.process_dialogs import GlobalParametersDialog
    s, e = scene
    units = UnitRegistry()
    e.set_global_parameters([{"name": "wind", "type": "vec3", "quantity": "velocity", "value": [1.0, 0.0, 0.0]}])
    dlg = GlobalParametersDialog(s, units)
    owned(dlg)
    assert dlg.table.rowCount() == 1
    dlg._add_row({"name": "rho", "type": "float", "quantity": "density", "value": 1.2})
    dlg._accept()
    assert dlg.result_params[0]["value"] == pytest.approx([1.0, 0.0, 0.0])
    assert dlg.result_params[1]["value"] == pytest.approx(1.2)
    dlg.table.item(1, 0).setText("bad name!")
    dlg.result_params = None
    dlg._accept()
    assert dlg.result_params is None and dlg.error.text()
