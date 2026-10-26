"""Developer-only tools: unlocking developer mode and the formula process editor.

These dialogs are only reachable with a valid developer token, and every
mutating operation goes through ProcessRegistry methods that verify the token
themselves (hiding the GUI is not the protection).
"""
from __future__ import annotations

import html
import json

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor, QTextFormat
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QTextEdit,
                               QVBoxLayout, QWidget)

from ...processes.dimensional_analysis import FUNCTION_DOCS
from ...processes.formula_process import FormulaProcessSpec, compile_spec, run_tests
from ...processes.packages import packages_dir
from ...processes.permissions import AccessController, DeveloperToken, PermissionDenied
from ...processes.process_base import ProcessError
from ...processes.process_registry import ProcessRegistry
from ...units.unit_registry import UnitRegistry


class DeveloperModeDialog(QDialog):
    def __init__(self, access: AccessController, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Developer Mode")
        self.setMinimumWidth(460)
        self.access = access
        self.token: DeveloperToken | None = None
        lay = QVBoxLayout(self)
        if not access.has_credentials():
            msg = QLabel("Developer mode lets an authorized developer create, validate, test and publish formula "
                         "processes. No developer credentials are installed on this computer.<br><br>"
                         "Create them from a terminal with:<br><code>python -m psim.devtools init-developer</code>")
            msg.setWordWrap(True)
            lay.addWidget(msg)
            b = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            b.rejected.connect(self.reject)
            lay.addWidget(b)
            return
        lay.addWidget(QLabel("Enter the developer passphrase to unlock process authoring for this session."))
        self.passphrase = QLineEdit()
        self.passphrase.setEchoMode(QLineEdit.EchoMode.Password)
        lay.addWidget(self.passphrase)
        self.error = QLabel()
        self.error.setStyleSheet("color: #e5534b;")
        lay.addWidget(self.error)
        b = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        b.accepted.connect(self._unlock)
        b.rejected.connect(self.reject)
        lay.addWidget(b)

    def _unlock(self) -> None:
        try:
            self.token = self.access.unlock(self.passphrase.text())
        except PermissionDenied as exc:
            self.error.setText(str(exc))
            self.passphrase.selectAll()
            return
        self.accept()


PARAM_COLUMNS = ["key", "label", "type", "quantity", "default", "minimum", "maximum", "description"]


class ProcessEditorDialog(QDialog):
    def __init__(self, registry: ProcessRegistry, access: AccessController, token: DeveloperToken,
                 units: UnitRegistry, spec: FormulaProcessSpec | None = None, parent=None) -> None:
        access.require(token, "The process editor")  # before any Qt object exists
        super().__init__(parent)
        self.setWindowTitle("Process Editor (developer)")
        self.resize(1180, 780)
        self.registry, self.access, self.token, self.units = registry, access, token, units
        self.published = None
        lay = QVBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        lay.addWidget(split, 1)

        # ---- left: definition -----------------------------------------------------------------------------
        tabs = QTabWidget()
        meta = QWidget()
        form = QFormLayout(meta)
        self.pid = QLineEdit()
        self.pid.setPlaceholderText("lab.my_process")
        self.name = QLineEdit()
        self.version = QLineEdit("1.0.0")
        self.category = QLineEdit("Custom")
        self.kind = QComboBox()
        self.kind.addItems(["contribution", "prescribed", "update"])
        self.mode = QComboBox()
        self.mode.addItems(["per_object", "global"])
        self.description = QLineEdit()
        self.requires_mass = QCheckBox("Targets must have mass")
        self.eligible = QLineEdit()
        self.eligible.setPlaceholderText("any type (or comma-separated type ids)")
        self.min_t = QSpinBox()
        self.min_t.setRange(1, 1000)
        self.max_t = QSpinBox()
        self.max_t.setRange(0, 1000)
        self.max_t.setSpecialValueText("unlimited")
        self.after = QLineEdit()
        self.after.setPlaceholderText("process ids this update process runs after")
        self.policy = QComboBox()
        self.policy.addItems(["stop", "skip"])
        self.author = QLineEdit(token.author)
        for label, w in (("Process id", self.pid), ("Name", self.name), ("Version", self.version),
                         ("Category", self.category), ("Kind", self.kind), ("Mode", self.mode),
                         ("Description", self.description), ("", self.requires_mass), ("Eligible types", self.eligible),
                         ("Min targets", self.min_t), ("Max targets", self.max_t), ("Runs after", self.after),
                         ("On runtime error", self.policy), ("Author", self.author)):
            form.addRow(label, w)
        tabs.addTab(meta, "Metadata")

        ptab = QWidget()
        pl = QVBoxLayout(ptab)
        self.params = QTableWidget(0, len(PARAM_COLUMNS))
        self.params.setHorizontalHeaderLabels(PARAM_COLUMNS)
        self.params.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        pl.addWidget(self.params)
        prow = QHBoxLayout()
        add = QPushButton("Add Parameter")
        add.clicked.connect(lambda: self._add_param({"key": "", "type": "float"}))
        rem = QPushButton("Remove Selected")
        rem.clicked.connect(lambda: [self.params.removeRow(r) for r in sorted({i.row() for i in
                                                                              self.params.selectedIndexes()},
                                                                             reverse=True)])
        prow.addWidget(add)
        prow.addWidget(rem)
        prow.addStretch(1)
        pl.addLayout(prow)
        hint = QLabel("type: float | int | bool | vec2 | vec3 · quantity: a unit-registry quantity (length, force, "
                      "density, …) · defaults, minimum and maximum in SI · vectors as x, y, z")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        pl.addWidget(hint)
        tabs.addTab(ptab, "Parameters")

        self.globals_edit = self._json_tab(tabs, "Required Globals",
                                           '[{"name": "air_density", "type": "float", "quantity": "density"}]')
        self.custom_edit = self._json_tab(tabs, "Required Custom Properties",
                                          '[{"key": "drag_area", "type": "float", "quantity": "area"}]')
        self.tests_edit = self._json_tab(tabs, "Test Cases",
                                         '[{"name": "…", "objects": {"self": {"velocity": [1, 0, 0]}}, '
                                         '"params": {}, "expect": {"force": [0, 0, 0]}, "tolerance": 1e-9}]')
        self.doc_edit = QPlainTextEdit()
        tabs.addTab(self.doc_edit, "Documentation")
        split.addWidget(tabs)

        # ---- right: script + reference + results ----------------------------------------------------------------
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("<b>Script</b> — <code>let x = …</code>, <code>require cond : \"msg\"</code>, "
                            "<code>&lt;output&gt; = …</code>"))
        self.script = QPlainTextEdit()
        mono = QFont("Cascadia Mono")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.script.setFont(mono)
        self.script.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        rl.addWidget(self.script, 3)
        ref = QTextBrowser()
        ref.setHtml("<b>Language reference</b><br>" + "<br>".join(
            f"<b>{k}:</b> {html.escape(v)}" for k, v in FUNCTION_DOCS.items()) +
            "<br><b>Outputs:</b> contribution → force, torque, acceleration, angular_acceleration; prescribed → "
            "position, velocity, orientation, angular_velocity; update → position, velocity, orientation, "
            "angular_velocity, custom.&lt;key&gt;. Global processes write targets[i].&lt;output&gt;.")
        ref.setMaximumHeight(170)
        rl.addWidget(ref)
        self.results = QTextBrowser()
        rl.addWidget(self.results, 2)
        split.addWidget(right)
        split.setSizes([480, 700])

        row = QHBoxLayout()
        for text, slot in (("Validate", self.validate), ("Run Tests", self.run_tests), ("Load Draft…", self.load_draft),
                           ("Save Draft…", self.save_draft)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        pub = QPushButton("Sign && Publish")
        pub.setDefault(True)
        pub.clicked.connect(self.publish)
        row.addWidget(pub)
        close = QPushButton("Close")
        close.setAutoDefault(False)
        close.clicked.connect(self.reject)
        row.addWidget(close)
        lay.addLayout(row)
        if spec is not None:
            self.load_spec(spec)

    def _json_tab(self, tabs: QTabWidget, title: str, placeholder: str) -> QPlainTextEdit:
        e = QPlainTextEdit()
        e.setPlaceholderText(placeholder)
        e.setFont(QFont("Cascadia Mono", 9))
        tabs.addTab(e, title)
        return e

    def _add_param(self, p: dict) -> None:
        r = self.params.rowCount()
        self.params.insertRow(r)
        for c, key in enumerate(PARAM_COLUMNS):
            v = p.get(key)
            if isinstance(v, list):
                text = ", ".join(f"{x:g}" for x in v)
            else:
                text = "" if v is None else str(v)
            self.params.setItem(r, c, QTableWidgetItem(text))

    # ---- spec <-> widgets ------------------------------------------------------------------------------------
    def load_spec(self, spec: FormulaProcessSpec) -> None:
        self.pid.setText(spec.process_id)
        self.name.setText(spec.name)
        self.version.setText(spec.version)
        self.category.setText(spec.category)
        self.kind.setCurrentText(spec.kind)
        self.mode.setCurrentText(spec.mode)
        self.description.setText(spec.description)
        self.requires_mass.setChecked(spec.requires_mass)
        self.eligible.setText(", ".join(spec.eligible_types))
        self.min_t.setValue(spec.min_targets)
        self.max_t.setValue(spec.max_targets or 0)
        self.after.setText(", ".join(spec.after))
        self.policy.setCurrentText(spec.error_policy)
        self.author.setText(spec.author or self.token.author)
        self.params.setRowCount(0)
        for p in spec.parameters:
            self._add_param(p)
        self.globals_edit.setPlainText(json.dumps(spec.globals, indent=1) if spec.globals else "")
        self.custom_edit.setPlainText(json.dumps(spec.custom, indent=1) if spec.custom else "")
        self.tests_edit.setPlainText(json.dumps(spec.tests, indent=1) if spec.tests else "")
        self.doc_edit.setPlainText(spec.documentation)
        self.script.setPlainText(spec.script)

    def collect(self) -> FormulaProcessSpec:
        def jlist(edit: QPlainTextEdit, what: str) -> list:
            text = edit.toPlainText().strip()
            if not text:
                return []
            try:
                v = json.loads(text)
            except ValueError as exc:
                raise ProcessError(f"{what}: invalid JSON ({exc})") from None
            if not isinstance(v, list):
                raise ProcessError(f"{what}: expected a JSON list")
            return v

        params = []
        for r in range(self.params.rowCount()):
            row = {k: (self.params.item(r, c).text().strip() if self.params.item(r, c) else "")
                   for c, k in enumerate(PARAM_COLUMNS)}
            if not row["key"]:
                continue
            p = {"key": row["key"], "label": row["label"] or row["key"], "type": row["type"] or "float"}
            if row["quantity"]:
                p["quantity"] = row["quantity"]
            for k in ("default", "minimum", "maximum"):
                if row[k]:
                    try:
                        vals = [float(x) for x in row[k].split(",")]
                    except ValueError:
                        if k == "default" and row["type"] == "bool":
                            p[k] = row[k].lower() in ("true", "1", "yes")
                            continue
                        raise ProcessError(f"Parameter '{row['key']}': {k} must be numeric") from None
                    p[k] = vals if len(vals) > 1 else vals[0]
            if row["description"]:
                p["description"] = row["description"]
            params.append(p)
        split = lambda t: [x.strip() for x in t.split(",") if x.strip()]  # noqa: E731
        return FormulaProcessSpec(
            process_id=self.pid.text().strip(), name=self.name.text().strip(), version=self.version.text().strip(),
            category=self.category.text().strip() or "Custom", description=self.description.text().strip(),
            kind=self.kind.currentText(), mode=self.mode.currentText(), parameters=params,
            globals=jlist(self.globals_edit, "Required globals"), custom=jlist(self.custom_edit, "Required custom"),
            script=self.script.toPlainText(), requires_mass=self.requires_mass.isChecked(),
            eligible_types=split(self.eligible.text()), min_targets=self.min_t.value(),
            max_targets=self.max_t.value() or None, after=split(self.after.text()), error_policy=self.policy.currentText(),
            documentation=self.doc_edit.toPlainText().strip(), tests=jlist(self.tests_edit, "Test cases"),
            author=self.author.text().strip())

    # ---- actions -------------------------------------------------------------------------------------------------
    def _show(self, html_text: str) -> None:
        self.results.setHtml(html_text)

    def validate(self) -> bool:
        try:
            spec = self.collect()
        except ProcessError as exc:
            self._show(f"<span style='color:#e5534b'>{html.escape(str(exc))}</span>")
            return False
        cp, issues = compile_spec(spec)
        sels = []
        for i in issues:
            if i.line > 0:
                block = self.script.document().findBlockByLineNumber(i.line - 1)
                sel = QTextEdit.ExtraSelection()
                sel.format.setBackground(Qt.GlobalColor.darkRed)
                sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
                sel.cursor = QTextCursor(block)
                sels.append(sel)
        self.script.setExtraSelections(sels)
        if cp is None:
            self._show("<b>Validation failed</b><br>" + "<br>".join(
                f"<span style='color:#e5534b'>✗</span> {html.escape(str(i))}" for i in issues))
            return False
        self._show(f"<span style='color:#4fb36b'>✓ Valid {spec.kind} process</span> — writes "
                   f"{html.escape(', '.join(sorted(cp.writes)))}; reads {html.escape(', '.join(sorted(cp.reads)) or '—')}"
                   + (f"; updated(): {html.escape(', '.join(sorted(cp.updated_refs)))}" if cp.updated_refs else ""))
        return True

    def run_tests(self) -> bool:
        if not self.validate():
            return False
        cp, _ = compile_spec(self.collect())
        results = run_tests(cp)
        if not results:
            self._show("<span style='color:#d9a33a'>No test cases defined. Publishing requires none, but tests "
                       "are strongly recommended.</span>")
            return True
        ok = all(r.passed for r in results)
        self._show(f"<b>{sum(r.passed for r in results)}/{len(results)} tests passed</b><br>" + "<br>".join(
            f"<span style='color:{'#4fb36b' if r.passed else '#e5534b'}'>{'✓' if r.passed else '✗'}</span> "
            f"{html.escape(r.name)} — {html.escape(r.message)}" for r in results))
        return ok

    def publish(self) -> None:
        if not self.run_tests():
            return
        spec = self.collect()
        try:
            pdef, path = self.registry.publish(spec, self.token, packages_dir(self.access.base))
        except (ProcessError, PermissionDenied) as exc:
            QMessageBox.warning(self, "Publish", str(exc))
            return
        self.published = pdef
        QMessageBox.information(self, "Published", f"{spec.name} {spec.version} was signed and published.\n\n{path}")
        self.accept()

    def save_draft(self) -> None:
        try:
            spec = self.collect()
        except ProcessError as exc:
            QMessageBox.warning(self, "Save draft", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Draft", f"{spec.process_id or 'process'}.json",
                                              "Process drafts (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(spec.to_dict(), fh, indent=1, ensure_ascii=False)

    def load_draft(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Draft", "", "Process drafts (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                self.load_spec(FormulaProcessSpec.from_dict(json.load(fh)))
        except (OSError, ValueError, ProcessError) as exc:
            QMessageBox.warning(self, "Load draft", str(exc))
