"""Collapsible inspector section with labelled, searchable rows."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from ...properties.property_system import Origin
from .. import icons
from .inputs import OriginBadge


class Section(QWidget):
    collapsed_state: dict[str, bool] = {}  # remembered per section key for the session

    def __init__(self, key: str, title: str, parent=None) -> None:
        super().__init__(parent)
        self.key = key
        self._rows: list[tuple[str, list[QWidget]]] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = QToolButton()
        self.header.setObjectName("SectionHeader")
        self.header.setText(title.replace("&", "&&"))   # "&" is literal, not a mnemonic
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.clicked.connect(self.toggle)
        lay.addWidget(self.header)
        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(10, 6, 8, 8)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(5)
        self.grid.setColumnStretch(2, 1)
        self.grid.setColumnMinimumWidth(1, 70)
        lay.addWidget(self.body)
        self._set_collapsed(self.collapsed_state.get(key, False))

    def toggle(self) -> None:
        self._set_collapsed(self.body.isVisible())

    def _set_collapsed(self, collapsed: bool) -> None:
        self.body.setVisible(not collapsed)
        self.collapsed_state[self.key] = collapsed
        self.header.setIcon(icons.icon("chevron_right" if collapsed else "chevron_down"))

    def add_row(self, label: str, editor: QWidget, unit: QWidget | None = None, origin: Origin | None = None,
                tooltip: str = "", search: str = "") -> None:
        r = self.grid.rowCount()
        widgets: list[QWidget] = []
        if origin is not None:
            b = OriginBadge(origin)
            self.grid.addWidget(b, r, 0, Qt.AlignmentFlag.AlignVCenter)
            widgets.append(b)
        lab = QLabel(label)
        lab.setObjectName("Muted")
        lab.setWordWrap(True)
        lab.setMaximumWidth(112)
        lab.setToolTip(tooltip or label)
        self.grid.addWidget(lab, r, 1)
        widgets.append(lab)
        if unit is None:
            self.grid.addWidget(editor, r, 2, 1, 2)
        else:
            self.grid.addWidget(editor, r, 2)
            self.grid.addWidget(unit, r, 3)
            widgets.append(unit)
        widgets.append(editor)
        self._rows.append(((label + " " + search).lower(), widgets))

    def add_widget(self, widget: QWidget, search: str = "") -> None:
        r = self.grid.rowCount()
        self.grid.addWidget(widget, r, 0, 1, 4)
        self._rows.append((search.lower(), [widget]))

    def add_buttons(self, *buttons: QWidget) -> None:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(4)
        for b in buttons:
            lay.addWidget(b)
        lay.addStretch(1)
        self.add_widget(row, search="")

    def apply_filter(self, text: str) -> bool:
        """Show only rows matching `text`; returns True if anything remains visible."""
        text = text.lower().strip()
        any_visible = False
        for hay, widgets in self._rows:
            show = not text or text in hay or not hay
            if text and not hay:
                show = False
            for w in widgets:
                w.setVisible(show)
            any_visible |= show
        self.setVisible(any_visible or not text)
        if text and any_visible and not self.body.isVisible():
            self.body.setVisible(True)
        return any_visible
