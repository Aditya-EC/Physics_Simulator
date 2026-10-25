"""Scene outliner: object hierarchy with selection sync, visibility/lock toggles and drag-to-reparent."""
from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QLineEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from ..scene.editing import SceneEditor
from ..scene.scene_manager import Scene, SceneError
from . import icons

ID_ROLE = Qt.ItemDataRole.UserRole


class _Tree(QTreeWidget):
    def __init__(self, outliner: "Outliner") -> None:
        super().__init__()
        self.outliner = outliner
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    def dropEvent(self, e) -> None:  # noqa: N802
        target = self.itemAt(e.position().toPoint())
        pos = self.dropIndicatorPosition()
        ids = [it.data(0, ID_ROLE) for it in self.selectedItems()]
        if target is None or pos == QAbstractItemView.DropIndicatorPosition.OnViewport:
            new_parent = None
        elif pos == QAbstractItemView.DropIndicatorPosition.OnItem:
            new_parent = target.data(0, ID_ROLE)
        else:
            p = target.parent()
            new_parent = p.data(0, ID_ROLE) if p is not None else None
        e.ignore()  # the scene, not Qt, owns the hierarchy
        self.outliner.reparentRequested.emit(ids, new_parent)


class Outliner(QWidget):
    reparentRequested = Signal(list, object)
    contextMenuRequested = Signal(QPoint, object)
    frameRequested = Signal(str)

    COLUMNS = ("Name", "ID", "", "")

    def __init__(self, scene: Scene, editor: SceneEditor, parent=None) -> None:
        super().__init__(parent)
        self.scene, self.editor = scene, editor
        self._syncing = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by name, ID, type or tag…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        lay.addWidget(self.filter)
        self.tree = _Tree(self)
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(list(self.COLUMNS))
        h = self.tree.header()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        h.setStretchLastSection(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context)
        self.tree.itemSelectionChanged.connect(self._tree_selection)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.itemDoubleClicked.connect(lambda it, col: col in (0, 1) and self.frameRequested.emit(it.data(0, ID_ROLE)))
        lay.addWidget(self.tree)
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(self.rebuild)
        for sig in (scene.objectsAdded, scene.objectsRemoved):
            sig.connect(lambda _ids: self._rebuild_timer.start(0))
        scene.objectsChanged.connect(lambda _ids: self._rebuild_timer.start(120))
        scene.hierarchyChanged.connect(lambda: self._rebuild_timer.start(0))
        scene.sceneReset.connect(lambda: self._rebuild_timer.start(0))
        scene.selectionChanged.connect(self._scene_selection)
        self.rebuild()

    def rebuild(self) -> None:
        expanded = {self.tree.topLevelItem(i).data(0, ID_ROLE) for i in range(self.tree.topLevelItemCount())
                    if self.tree.topLevelItem(i).isExpanded()}
        stack = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            it = stack.pop()
            if it.isExpanded():
                expanded.add(it.data(0, ID_ROLE))
            stack.extend(it.child(k) for k in range(it.childCount()))
        known = getattr(self, "_known", None)
        self._syncing = True
        self.tree.clear()
        items: dict[str, QTreeWidgetItem] = {}

        def add(obj, parent_item):
            it = QTreeWidgetItem([obj.name, obj.id, "", ""])
            it.setData(0, ID_ROLE, obj.id)
            it.setIcon(0, icons.icon(_type_icon(obj.type_id)))
            it.setIcon(2, icons.icon("eye" if obj.visible else "eye_off"))
            it.setToolTip(2, "Toggle visibility")
            it.setIcon(3, icons.icon("lock" if obj.locked else "unlock"))
            it.setToolTip(3, "Toggle lock")
            it.setToolTip(0, f"{obj.typedef.name}" + (f"\nTags: {', '.join(obj.tags)}" if obj.tags else ""))
            if not obj.visible:
                it.setForeground(0, self.palette().placeholderText())
            if parent_item is None:
                self.tree.addTopLevelItem(it)
            else:
                parent_item.addChild(it)
            items[obj.id] = it
            for c in self.scene.children(obj.id):
                add(c, it)
            # expand on first build, keep user expansion state, and open newly created items
            if known is None or obj.id in expanded or obj.id not in known:
                it.setExpanded(True)

        for root in self.scene.roots():
            add(root, None)
        self._items = items
        self._known = set(items)
        self._syncing = False
        self._scene_selection(self.scene.selection)
        self._apply_filter(self.filter.text())

    def _apply_filter(self, text: str) -> None:
        text = text.lower().strip()

        def visit(it: QTreeWidgetItem) -> bool:
            obj = self.scene.find(it.data(0, ID_ROLE))
            hay = f"{obj.name} {obj.id} {obj.typedef.name} {' '.join(obj.tags)}".lower() if obj else ""
            child_match = any([visit(it.child(k)) for k in range(it.childCount())])
            match = not text or text in hay or child_match
            it.setHidden(not match)
            return match

        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))

    def _scene_selection(self, ids) -> None:
        if not hasattr(self, "_items"):
            return
        self._syncing = True
        self.tree.clearSelection()
        for i in ids:
            it = self._items.get(i)
            if it is not None:
                it.setSelected(True)
                p = it.parent()
                while p is not None:
                    p.setExpanded(True)
                    p = p.parent()
        if ids and ids[-1] in self._items:
            self.tree.scrollToItem(self._items[ids[-1]])
        self._syncing = False

    def _tree_selection(self) -> None:
        if self._syncing:
            return
        ids = [it.data(0, ID_ROLE) for it in self.tree.selectedItems()]
        cur = self.tree.currentItem()
        if cur is not None and cur.data(0, ID_ROLE) in ids:  # primary = current item
            ids.remove(cur.data(0, ID_ROLE))
            ids.append(cur.data(0, ID_ROLE))
        self.scene.set_selection(ids)

    def _clicked(self, item: QTreeWidgetItem, column: int) -> None:
        oid = item.data(0, ID_ROLE)
        obj = self.scene.find(oid)
        if obj is None:
            return
        try:
            if column == 2:
                self.editor.set_property([oid], "visible", not obj.visible, merge=False)
            elif column == 3:
                self.editor.set_property([oid], "locked", not obj.locked, merge=False)
        except SceneError:
            pass

    def _context(self, pos) -> None:
        it = self.tree.itemAt(pos)
        oid = it.data(0, ID_ROLE) if it else None
        if oid and oid not in self.scene.selection:
            self.scene.set_selection([oid])
        self.contextMenuRequested.emit(self.tree.viewport().mapToGlobal(pos), oid)


def _type_icon(type_id: str) -> str:
    if type_id.startswith("frame.") or type_id == "group.assembly":
        return "frame_axes" if type_id != "group.assembly" else "layers"
    if type_id.startswith("measure."):
        return "ruler"
    if type_id == "body.point_mass":
        return "mass"
    return "cube"
