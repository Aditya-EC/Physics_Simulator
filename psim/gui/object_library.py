"""Object Library panel: searchable, categorised cards with 3D thumbnails; drag into the viewport."""
from __future__ import annotations

import json

from PySide6.QtCore import QMimeData, QSize, Qt, Signal
from PySide6.QtGui import QDrag, QIcon, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem, QMenu,
                               QPushButton, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from ..rendering.thumbnails import render_thumbnail
from ..rendering.viewport import LIBRARY_MIME
from . import icons
from .library_items import Library

THUMB = 64


class _CardList(QListWidget):
    """Icon-mode list that sizes itself to its contents and starts library drags."""

    def __init__(self, panel: "ObjectLibraryPanel") -> None:
        super().__init__()
        self.panel = panel
        self.setObjectName("LibraryList")
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setIconSize(QSize(THUMB, THUMB))
        self.setGridSize(QSize(THUMB + 30, THUMB + 38))
        self.setWordWrap(True)
        self.setUniformItemSizes(True)
        self.setSpacing(2)
        self.setDragEnabled(True)
        self.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QListWidget.Shape.NoFrame)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def startDrag(self, actions) -> None:  # noqa: N802
        item = self.currentItem()
        if item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        mime = QMimeData()
        mime.setData(LIBRARY_MIME, json.dumps({"key": key}).encode())
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(item.icon().pixmap(QSize(THUMB, THUMB)))
        drag.setHotSpot(drag.pixmap().rect().center() / drag.pixmap().devicePixelRatio())
        drag.exec(Qt.DropAction.CopyAction)

    def relayout(self) -> None:
        visible = sum(1 for i in range(self.count()) if not self.item(i).isHidden())
        width = max(self.viewport().width(), self.gridSize().width())
        per_row = max(1, width // self.gridSize().width())
        rows = (visible + per_row - 1) // per_row
        self.setFixedHeight(rows * self.gridSize().height() + 6 if rows else 0)

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self.relayout()

    def _menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.addAction(icons.icon("plus"), "Add to Scene", lambda: self.panel.addRequested.emit(key))
        if key.startswith("template:"):
            menu.addAction(icons.icon("trash"), "Delete Template", lambda: self.panel.deleteTemplateRequested.emit(key))
        menu.exec(self.mapToGlobal(pos))


class ObjectLibraryPanel(QWidget):
    addRequested = Signal(str)             # library key (double-click / context menu)
    importMeshRequested = Signal()
    deleteTemplateRequested = Signal(str)

    def __init__(self, library: Library, parent=None) -> None:
        super().__init__(parent)
        self.library = library
        self._thumbs: dict[str, QIcon] = {}
        self._sections: dict[str, tuple[QToolButton, _CardList, QLabel]] = {}
        self._collapsed: set[str] = set()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 4)
        lay.setSpacing(6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search objects…")
        self.search.setClearButtonEnabled(True)
        self.search.addAction(icons.icon("search"), QLineEdit.ActionPosition.LeadingPosition)
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 0, 0, 0)
        self.body_lay.setSpacing(0)
        self.scroll.setWidget(self.body)
        lay.addWidget(self.scroll, 1)
        hint = QLabel("Drag objects into the viewport, or double-click to add.")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.rebuild()

    def thumbnail(self, key: str) -> QIcon:
        if key not in self._thumbs:
            mesh, color, vcol, view = self.library.thumbnail_spec(key)
            dpr = max(self.devicePixelRatioF(), 1.0)
            img = render_thumbnail(mesh, THUMB, color, view[0], view[1], vertex_colors=vcol, dpr=dpr) if mesh else None
            self._thumbs[key] = QIcon(QPixmap.fromImage(img)) if img is not None else icons.icon("cube")
        return self._thumbs[key]

    def rebuild(self) -> None:
        while self.body_lay.count():
            w = self.body_lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._sections.clear()
        for cat in self.library.categories():
            header = QToolButton()
            header.setObjectName("CategoryHeader")
            header.setText(cat.upper())
            header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            header.setIconSize(QSize(12, 12))
            header.clicked.connect(lambda _=False, c=cat: self._toggle(c))
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 4, 0, 0)
            rl.addWidget(header, 1)
            if cat == "User-Defined Objects":
                b = QPushButton(icons.icon("import"), "Import OBJ…")
                b.setProperty("flat", True)
                b.clicked.connect(self.importMeshRequested.emit)
                rl.addWidget(b)
            lst = _CardList(self)
            lst.itemDoubleClicked.connect(lambda it: self.addRequested.emit(it.data(Qt.ItemDataRole.UserRole)))
            for it in self.library.items():
                if it.category != cat:
                    continue
                item = QListWidgetItem(self.thumbnail(it.key), it.name)
                item.setData(Qt.ItemDataRole.UserRole, it.key)
                item.setToolTip(f"<b>{it.name}</b><br>{it.description}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
                lst.addItem(item)
            empty = QLabel(self._empty_text(cat))
            empty.setObjectName("Muted")
            empty.setWordWrap(True)
            empty.setContentsMargins(6, 2, 6, 6)
            empty.setVisible(lst.count() == 0)
            self.body_lay.addWidget(row)
            self.body_lay.addWidget(lst)
            self.body_lay.addWidget(empty)
            self._sections[cat] = (header, lst, empty)
        self.body_lay.addStretch(1)
        self._filter(self.search.text())

    @staticmethod
    def _empty_text(cat: str) -> str:
        if cat == "Saved Templates":
            return "Select objects and use Objects ▸ Save as Template to create reusable templates."
        if cat == "User-Defined Objects":
            return "Import a Wavefront OBJ mesh to create a custom polygonal object."
        return "No objects."

    def _toggle(self, cat: str) -> None:
        if cat in self._collapsed:
            self._collapsed.remove(cat)
        else:
            self._collapsed.add(cat)
        self._filter(self.search.text())

    def _filter(self, text: str) -> None:
        matches = {it.key for it in self.library.search(text)} if text.strip() else None
        for cat, (header, lst, empty) in self._sections.items():
            n = 0
            for i in range(lst.count()):
                item = lst.item(i)
                show = matches is None or item.data(Qt.ItemDataRole.UserRole) in matches
                item.setHidden(not show)
                n += show
            collapsed = cat in self._collapsed and matches is None
            header.setIcon(icons.icon("chevron_right" if collapsed else "chevron_down"))
            header.parentWidget().setVisible(n > 0 or matches is None)
            lst.setVisible(n > 0 and not collapsed)
            empty.setVisible(n == 0 and matches is None and not collapsed)
            lst.relayout()
