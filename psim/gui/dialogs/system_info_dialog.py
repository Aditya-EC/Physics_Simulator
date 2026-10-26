"""Help ▸ System Information: versions, numerical core, OpenGL and the log folder (for support requests)."""
from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout

from ... import crash


def info_text(gl_info: dict | None = None) -> str:
    info = crash.system_info()
    if gl_info:
        info["OpenGL"] = f"{gl_info.get('renderer', '?')} · {gl_info.get('version', '?')}"
    info["log folder"] = crash.log_dir() or "(file logging not active)"
    last = crash.last_report()
    if last:
        info["last crash report"] = last
    return "\n".join(f"{k}: {v}" for k, v in info.items())


class SystemInfoDialog(QDialog):
    def __init__(self, gl_info: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("System Information")
        self.resize(640, 360)
        lay = QVBoxLayout(self)
        note = QLabel("Include this information when reporting a problem. Unexpected errors are saved as "
                      "crash reports in the log folder.")
        note.setWordWrap(True)
        note.setObjectName("Muted")
        lay.addWidget(note)
        self.text = QPlainTextEdit(info_text(gl_info))
        self.text.setReadOnly(True)
        lay.addWidget(self.text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy = QPushButton("Copy")
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.text.toPlainText()))
        self.open_logs = QPushButton("Open Log Folder")
        self.open_logs.setEnabled(crash.log_dir() is not None)
        self.open_logs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(crash.log_dir())))
        buttons.addButton(copy, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self.open_logs, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
