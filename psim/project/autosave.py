"""Periodic autosave and crash recovery.

While a project has unsaved changes, a recovery snapshot is written every
`interval_s` seconds to the per-user data directory. A clean shutdown removes
it; if the file is present at start-up, the previous session ended
unexpectedly and the user is offered recovery.
"""
from __future__ import annotations

import json
import os
from typing import Callable

from PySide6.QtCore import QObject, QStandardPaths, QTimer

from .serialization import write_json_atomic


def data_dir() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".psim_studio")
    os.makedirs(base, exist_ok=True)
    return base


def recovery_path() -> str:
    return os.path.join(data_dir(), "recovery.psim")


class Autosaver(QObject):
    def __init__(self, snapshot: Callable[[], dict], is_dirty: Callable[[], bool], interval_s: int = 120,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._snapshot = snapshot
        self._dirty = is_dirty
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.save_now)
        self.last_error = ""
        self.set_interval(interval_s)

    def set_interval(self, seconds: int) -> None:
        self.interval_s = max(0, int(seconds))
        if self.interval_s:
            self._timer.start(self.interval_s * 1000)
        else:
            self._timer.stop()

    def save_now(self) -> bool:
        if not self._dirty():
            return False
        try:
            payload = self._snapshot()
            write_json_atomic(recovery_path(), payload)
            self.last_error = ""
            return True
        except (OSError, TypeError, ValueError) as exc:
            self.last_error = str(exc)
            return False

    @staticmethod
    def discard() -> None:
        try:
            os.remove(recovery_path())
        except OSError:
            pass

    @staticmethod
    def pending_recovery() -> dict | None:
        path = recovery_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None
