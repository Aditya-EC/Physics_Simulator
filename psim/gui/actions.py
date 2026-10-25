"""Central registry of QActions with stable ids and user-configurable shortcuts."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QSettings
from PySide6.QtGui import QAction, QKeySequence

from . import icons


class ActionRegistry(QObject):
    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self._actions: dict[str, QAction] = {}
        self._defaults: dict[str, str] = {}

    def add(self, aid: str, text: str, slot: Callable | None = None, *, icon: str | None = None,
            shortcut: str = "", tip: str = "", checkable: bool = False, checked: bool = False) -> QAction:
        if aid in self._actions:
            raise KeyError(f"duplicate action id '{aid}'")
        a = QAction(text, self.parent())
        a.setObjectName(aid)
        if icon:
            a.setIcon(icons.icon(icon))
            a.setProperty("icon_name", icon)
        if tip or shortcut:
            a.setToolTip(f"{tip or text.replace('&', '').rstrip('.…')}" + (f"  ({shortcut})" if shortcut else ""))
            a.setStatusTip(tip or text.replace("&", ""))
        if checkable:
            a.setCheckable(True)
            a.setChecked(checked)
        if slot is not None:
            (a.toggled if checkable else a.triggered).connect(slot)
        self._defaults[aid] = shortcut
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        self._actions[aid] = a
        return a

    def register(self, aid: str, action: QAction, shortcut: str = "", icon: str | None = None) -> QAction:
        """Add an externally created action (e.g. QUndoStack actions) to the registry."""
        if aid in self._actions:
            raise KeyError(f"duplicate action id '{aid}'")
        action.setObjectName(aid)
        if icon:
            action.setIcon(icons.icon(icon))
            action.setProperty("icon_name", icon)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        self._defaults[aid] = shortcut
        self._actions[aid] = action
        return action

    def __getitem__(self, aid: str) -> QAction:
        return self._actions[aid]

    def get(self, aid: str) -> QAction | None:
        return self._actions.get(aid)

    def items(self):
        return self._actions.items()

    def default_shortcut(self, aid: str) -> str:
        return self._defaults.get(aid, "")

    def set_shortcut(self, aid: str, seq: str) -> None:
        a = self._actions[aid]
        a.setShortcut(QKeySequence(seq))
        tip = a.statusTip() or a.text().replace("&", "")
        a.setToolTip(tip + (f"  ({seq})" if seq else ""))

    def conflicts(self) -> dict[str, list[str]]:
        seen: dict[str, list[str]] = {}
        for aid, a in self._actions.items():
            s = a.shortcut().toString()
            if s:
                seen.setdefault(s, []).append(aid)
        return {k: v for k, v in seen.items() if len(v) > 1}

    def load_overrides(self, settings: QSettings) -> None:
        settings.beginGroup("shortcuts")
        for aid in settings.childKeys():
            if aid in self._actions:
                self.set_shortcut(aid, str(settings.value(aid, "")))
        settings.endGroup()

    def save_overrides(self, settings: QSettings) -> None:
        settings.beginGroup("shortcuts")
        settings.remove("")
        for aid, a in self._actions.items():
            cur = a.shortcut().toString()
            if cur != self._defaults.get(aid, ""):
                settings.setValue(aid, cur)
        settings.endGroup()
