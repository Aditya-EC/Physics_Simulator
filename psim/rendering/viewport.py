"""Interactive 3D viewport (QOpenGLWidget).

Mouse bindings
    LMB click            select (Ctrl: toggle, Shift: add)
    LMB drag on gizmo    move / rotate / scale the selection
    LMB drag on empty    orbit             Alt+LMB drag  orbit (anywhere)
    MMB / RMB drag       pan               Shift+LMB drag pan
    Wheel                zoom towards the cursor
    RMB click            context menu      Double-click  frame object
"""
from __future__ import annotations

import json
import logging
import math
import time

import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .. import numerics
from ..scene.editing import SceneEditor
from ..scene.geometry import Mesh
from ..scene.scene_manager import Scene
from ..units.unit_registry import UnitRegistry
from . import gizmo as gz
from . import overlays, picking
from .camera import OrbitCamera
from .renderer import FrameInput, FrameStats, GhostDraw, GizmoDraw, Renderer
from .view_settings import ViewSettings

log = logging.getLogger("psim.viewport")
LIBRARY_MIME = "application/x-psim-library-item"
TOOLS = ("select", "translate", "rotate", "scale")
FOLLOW_MODES = ("none", "look_at", "follow", "track_frame")


class Viewport(QOpenGLWidget):
    statsUpdated = Signal(object)              # FrameStats
    contextMenuRequested = Signal(QPoint, object)  # global position, object id or None
    dropRequested = Signal(str, list, object)  # payload JSON, world position, surface object id
    toolChanged = Signal(str)
    cameraChanged = Signal()
    message = Signal(str)

    def __init__(self, scene: Scene, editor: SceneEditor, view: ViewSettings, units: UnitRegistry, parent=None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.editor = editor
        self.view = view
        self.units = units
        self.camera = OrbitCamera(view.camera)
        self.renderer = Renderer()
        self.handles = gz.build_handles()
        self.tool = "translate"
        self.gizmo_global = True
        self.hover_id: str | None = None
        self.hover_handle: str | None = None
        self.follow_mode = "none"
        self.follow_id: str | None = None
        self._follow_offset: np.ndarray | None = None
        self._follow_eye: np.ndarray | None = None
        self.ghost_provider = None  # callable(payload: dict) -> Mesh | None
        self.status_provider = None  # callable() -> str shown in the HUD (e.g. simulation time)
        self._ghost: GhostDraw | None = None
        self._drag_payload: dict | None = None
        self._press_pos: QPoint | None = None
        self._last_pos: QPoint | None = None
        self._mode: str | None = None  # orbit | pan | gizmo | pending
        self._drag: gz.DragState | None = None
        self._drag_text = ""
        self.last_stats = FrameStats()
        self._last_render_error = ""
        self._frame_times: list[float] = []
        self.measured_fps = 0.0
        self.continuous = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self.render_fps = 60.0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setMinimumSize(320, 240)
        for sig in (scene.objectsAdded, scene.objectsRemoved, scene.objectsChanged, scene.selectionChanged):
            sig.connect(self._scene_changed)
        scene.sceneReset.connect(self._scene_changed)

    # ---- configuration -------------------------------------------------------------------------
    def set_view_settings(self, view: ViewSettings) -> None:
        self.view = view
        self.camera.state = view.camera
        self.update()

    def set_tool(self, tool: str) -> None:
        if tool in TOOLS and tool != self.tool:
            self.tool = tool
            self.toolChanged.emit(tool)
            self.update()

    def set_gizmo_global(self, on: bool) -> None:
        self.gizmo_global = on
        self.update()

    def set_continuous(self, on: bool, fps: float | None = None) -> None:
        if fps:
            self.render_fps = max(1.0, min(480.0, fps))
        self.continuous = on
        if on:
            self._timer.start(max(1, int(round(1000.0 / self.render_fps))))
        else:
            self._timer.stop()
            self.measured_fps = 0.0
        self.update()

    def set_follow(self, mode: str, oid: str | None) -> None:
        if mode not in FOLLOW_MODES:
            return
        self.follow_mode = mode if oid else "none"
        self.follow_id = oid if mode != "none" else None
        self._follow_offset = None
        self._follow_eye = None
        if self.follow_id and self.follow_id in self.scene:
            p, q = self.scene.world_pose(self.follow_id)
            if mode == "look_at":
                self._follow_eye = self.camera.eye()
            elif mode == "track_frame":
                r = np.asarray(numerics.quat_to_matrix(q))
                self._follow_offset = r.T @ (self.camera.eye() - p)
        self.update()

    def _scene_changed(self, *args) -> None:
        if self.hover_id and self.hover_id not in self.scene:
            self.hover_id = None
        if self.follow_id and self.follow_id not in self.scene:
            self.set_follow("none", None)
        self.update()

    # ---- camera commands -------------------------------------------------------------------------
    def frame_objects(self, ids: list[str] | None) -> None:
        b = self.scene.world_bounds(ids) if ids is None or ids else None
        if b is None:
            self.camera.fit_bounds(np.array([-2, -2, 0.0]), np.array([2, 2, 1.0]), self._aspect())
        else:
            self.camera.fit_bounds(b[0], b[1], self._aspect())
        self.cameraChanged.emit()
        self.update()

    def focus_origin(self) -> None:
        self.camera.target = np.zeros(3)
        self.cameraChanged.emit()
        self.update()

    def set_standard_view(self, name: str) -> None:
        self.camera.set_view(name)
        self.cameraChanged.emit()
        self.update()

    def set_orthographic(self, on: bool) -> None:
        self.camera.state.orthographic = on
        self.cameraChanged.emit()
        self.update()

    def _aspect(self) -> float:
        return self.width() / max(self.height(), 1)

    # ---- GL lifecycle --------------------------------------------------------------------------------
    def initializeGL(self) -> None:  # noqa: N802
        self.renderer.initialize()
        if self.renderer.error:
            self.message.emit(f"OpenGL initialisation failed: {self.renderer.error}")
        ctx = self.context()
        if ctx is not None:
            ctx.aboutToBeDestroyed.connect(self._cleanup)

    def _cleanup(self) -> None:
        self.makeCurrent()
        self.renderer.shutdown()
        self.doneCurrent()

    def paintGL(self) -> None:  # noqa: N802
        self._apply_follow()
        dpr = self.devicePixelRatioF()
        w, h = int(self.width() * dpr), int(self.height() * dpr)
        frame = FrameInput(self.scene, self.view, self.camera, w, h, self.defaultFramebufferObject(),
                           hover_id=self.hover_id, gizmo=self._gizmo_draw(), ghost=self._ghost, dpr=dpr)
        try:
            stats = self.renderer.render(frame)
        except Exception as exc:  # never let a rendering error escape into Qt's event loop
            stats = FrameStats()
            msg = f"Rendering error: {type(exc).__name__}: {exc}"
            if msg != self._last_render_error:
                self._last_render_error = msg
                log.exception(msg)
                self.message.emit(msg)
        self.last_stats = stats
        now = time.perf_counter()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t <= 1.0]
        if self.continuous and len(self._frame_times) > 1:
            self.measured_fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])
        painter = QPainter(self)
        try:
            overlays.paint_overlays(painter, self.scene, self.view, self.camera, self.units, self.width(),
                                    self.height(), self._hud_lines(), self._drag_text)
            if self._drag_payload is not None:
                overlays.paint_drop_hint(painter, self.width(), self.height(),
                                         f"Release to place · work plane {self.view.snap.work_plane}"
                                         f"{' · surface snap' if self.view.snap.surface_snap else ''}")
            if not self.renderer.ready:
                painter.setPen(Qt.GlobalColor.white)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                                 "OpenGL 3.3 is not available.\n" + self.renderer.error)
        finally:
            painter.end()
        self.statsUpdated.emit(stats)

    def _hud_lines(self) -> list[str]:
        if not self.view.environment.show_stats:
            return []
        proj = "Orthographic" if self.camera.state.orthographic else "Perspective"
        tool = {"select": "Select", "translate": "Move", "rotate": "Rotate", "scale": "Scale"}[self.tool]
        space = "Global" if self.gizmo_global else "Local"
        sel = len(self.scene.selection)
        lines = [f"{proj}  ·  {tool} ({space})  ·  {len(self.scene)} objects  ·  {sel} selected"]
        if self.status_provider is not None:
            extra = self.status_provider()
            if extra:
                lines.append(extra)
        if self.follow_mode != "none" and self.follow_id:
            label = {"look_at": "Looking at", "follow": "Following", "track_frame": "Tracking frame of"}[self.follow_mode]
            lines.append(f"Camera: {label} {self.follow_id}")
        return lines

    def _apply_follow(self) -> None:
        if self.follow_mode == "none" or not self.follow_id or self.follow_id not in self.scene:
            return
        p, q = self.scene.world_pose(self.follow_id)
        cam = self.camera
        if self.follow_mode == "follow":
            cam.target = p
            return
        if self.follow_mode == "look_at":
            eye = self._follow_eye if self._follow_eye is not None else cam.eye()
        else:
            if self._follow_offset is None:
                return
            eye = p + np.asarray(numerics.quat_to_matrix(q)) @ self._follow_offset
        d = eye - p
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return
        cam.target = p
        cam.state.distance = dist
        cam.state.elevation_deg = float(np.clip(math.degrees(math.asin(np.clip(d[2] / dist, -1, 1))), -89.9, 89.9))
        cam.state.azimuth_deg = math.degrees(math.atan2(d[1], d[0]))

    # ---- gizmo ----------------------------------------------------------------------------------------
    def _gizmo_targets(self) -> list[str]:
        if self.tool == "select":
            return []
        return [i for i in self.scene.selection if not self.scene.is_effectively_locked(i)]

    def _gizmo_frame(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        targets = self._gizmo_targets()
        primary = self.scene.primary
        if not targets or primary is None or primary.id not in targets:
            return None
        pivot, q = self.scene.world_pose(primary.id)
        local = self.tool == "scale" or not self.gizmo_global
        axes = np.asarray(numerics.quat_to_matrix(q)) if local else np.eye(3)
        size = self.camera.world_per_pixel(pivot, self.height()) * gz.GIZMO_PIXELS
        return pivot, axes, size

    def _gizmo_draw(self) -> GizmoDraw | None:
        fr = self._gizmo_frame()
        if fr is None:
            return None
        pivot, axes, size = fr
        return GizmoDraw(self.handles[self.tool], gz.gizmo_matrix(pivot, axes, size), self.hover_handle,
                         self._drag.handle.name if self._drag else None)

    def _pick_gizmo(self, pos) -> str | None:
        fr = self._gizmo_frame()
        if fr is None:
            return None
        o, d = self._ray(pos)
        return gz.pick_handle(self.handles[self.tool], gz.gizmo_matrix(*fr), o, d)

    # ---- mouse --------------------------------------------------------------------------------------------
    def _ray(self, pos) -> tuple[np.ndarray, np.ndarray]:
        return self.camera.screen_ray(pos.x(), pos.y(), self.width(), self.height())

    def pick_at(self, pos) -> picking.PickResult | None:
        o, d = self._ray(pos)
        return picking.pick(self.scene, o, d)

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        pos = e.position().toPoint()
        self._press_pos = self._last_pos = pos
        btn, mods = e.button(), e.modifiers()
        if btn == Qt.MouseButton.LeftButton:
            if mods & Qt.KeyboardModifier.AltModifier:
                self._mode = "orbit"
                return
            handle = self._pick_gizmo(pos)
            if handle is not None:
                self._start_gizmo_drag(handle, pos)
                return
            self._mode = "pending_pan" if mods & Qt.KeyboardModifier.ShiftModifier and not self.pick_at(pos) else "pending"
        elif btn in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._mode = "pan" if btn == Qt.MouseButton.MiddleButton else "pending_rmb"

    def _start_gizmo_drag(self, handle_name: str, pos) -> None:
        targets = self._gizmo_targets()
        handle = next(h for h in self.handles[self.tool] if h.name == handle_name)
        if handle.kind != "scale":  # children follow their parents; scale is not inherited
            targets = [i for i in targets if not any(a in targets for a in self.scene.ancestors(i))]
        pivot, axes, _ = self._gizmo_frame()
        start = {}
        for i in targets:
            p, q = self.scene.world_pose(i)
            start[i] = (p, q, np.asarray(self.scene.get(i).scale, float))
        o, d = self._ray(pos)
        st = gz.begin_drag(handle, pivot, axes, self.camera.forward(), start, self.scene.primary.id,
                           self.gizmo_global and self.tool != "scale", o, d)
        if st is None:
            return
        self._drag = st
        self._mode = "gizmo"
        self._drag_start_states = self.scene.snapshot(targets)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        pos = e.position().toPoint()
        last = self._last_pos or pos
        dx, dy = pos.x() - last.x(), pos.y() - last.y()
        self._last_pos = pos
        moved = self._press_pos is not None and (pos - self._press_pos).manhattanLength() > 3
        if self._mode == "gizmo" and self._drag is not None:
            o, d = self._ray(pos)
            poses = gz.update_drag(self._drag, o, d, self.view.snap)
            if poses:
                for i, (p, q, s) in poses.items():
                    self.scene.set_world_pose(i, p, q)
                    self.scene.get(i).scale = [float(c) for c in s]
                self._drag_text = self._describe_drag(poses)
                self.scene.notify_changed(list(poses))
            return
        if self._mode == "pending" and moved:
            self._mode = "orbit"
        elif self._mode == "pending_pan" and moved:
            self._mode = "pan"
        elif self._mode == "pending_rmb" and moved:
            self._mode = "pan"
        if self._mode == "orbit":
            self.camera.orbit(-dx * 0.4, dy * 0.4)
            self._camera_moved()
        elif self._mode == "pan":
            self.camera.pan_pixels(dx, dy, self.height())
            self._camera_moved()
        elif self._mode is None:
            self._update_hover(pos)

    def _camera_moved(self) -> None:
        if self.follow_mode in ("look_at", "track_frame"):
            # user camera input re-anchors the tracked offset
            self.set_follow(self.follow_mode, self.follow_id)
        self.cameraChanged.emit()
        self.update()

    def _update_hover(self, pos) -> None:
        handle = self._pick_gizmo(pos)
        hit = None if handle else self.pick_at(pos)
        hid = hit.object_id if hit else None
        if handle != self.hover_handle or hid != self.hover_id:
            self.hover_handle, self.hover_id = handle, hid
            self.setCursor(Qt.CursorShape.SizeAllCursor if handle else
                           (Qt.CursorShape.PointingHandCursor if hid else Qt.CursorShape.ArrowCursor))
            self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        pos = e.position().toPoint()
        mode = self._mode
        self._mode = None
        if mode == "gizmo" and self._drag is not None:
            ids = list(self._drag.start)
            final = self.scene.snapshot(ids)
            # revert, then re-apply through the editor so the drag is one undoable step
            self.scene.apply_states(self._drag_start_states)
            text = {"translate": "Move", "rotate": "Rotate", "scale": "Scale"}[self._drag.handle.kind]
            self.editor.edit(ids, lambda: self.scene.apply_states(final), text)
            self._drag = None
            self._drag_text = ""
            self.update()
            return
        if mode == "pending" and e.button() == Qt.MouseButton.LeftButton:
            hit = self.pick_at(pos)
            mods = e.modifiers()
            if hit is None:
                if not mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
                    self.scene.set_selection([])
            elif mods & Qt.KeyboardModifier.ControlModifier:
                self.scene.toggle_selection(hit.object_id)
            elif mods & Qt.KeyboardModifier.ShiftModifier:
                sel = self.scene.selection
                if hit.object_id in sel:
                    sel.remove(hit.object_id)
                self.scene.set_selection([*sel, hit.object_id])
            else:
                self.scene.set_selection([hit.object_id])
        elif mode == "pending_rmb" and e.button() == Qt.MouseButton.RightButton:
            hit = self.pick_at(pos)
            if hit is not None and hit.object_id not in self.scene.selection:
                self.scene.set_selection([hit.object_id])
            self.contextMenuRequested.emit(self.mapToGlobal(pos), hit.object_id if hit else None)

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        hit = self.pick_at(e.position().toPoint())
        if hit is not None:
            self.scene.set_selection([hit.object_id])
            self.frame_objects([hit.object_id])

    def wheelEvent(self, e: QWheelEvent) -> None:  # noqa: N802
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return
        factor = 0.85 ** steps
        pos = e.position().toPoint()
        hit = self.pick_at(pos)
        toward = hit.point if hit is not None else None
        if toward is None and not self.camera.state.orthographic:
            o, d = self._ray(pos)
            p = gz.ray_plane(o, d, self.camera.target, self.camera.forward())
            toward = p
        self.camera.zoom(factor, toward)
        self._camera_moved()

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Escape and self._mode == "gizmo" and self._drag is not None:
            self.scene.apply_states(self._drag_start_states)
            self._drag = None
            self._mode = None
            self._drag_text = ""
            self.update()
            return
        super().keyPressEvent(e)

    def _describe_drag(self, poses) -> str:
        st = self._drag
        u = self.units
        if st.handle.kind == "translate":
            p0 = st.start[st.primary][0]
            p1 = poses[st.primary][0]
            unit = u.preferred_unit("length")
            d = u.from_si(np.asarray(p1 - p0), unit)
            pos = u.from_si(np.asarray(p1), unit)
            return (f"Δ = ({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}) {unit}    "
                    f"position = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) {unit}")
        if st.handle.kind == "rotate":
            unit = u.preferred_unit("angle")
            return f"Rotation about {'XYZ'[st.handle.axis]}: {u.from_si(st.extra.get('angle', 0.0), unit):.2f} {unit}"
        s = poses[st.primary][2]
        return f"Scale = ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})"

    # ---- drag & drop from the object library -----------------------------------------------------------------
    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasFormat(LIBRARY_MIME):
            try:
                self._drag_payload = json.loads(bytes(e.mimeData().data(LIBRARY_MIME)).decode())
            except ValueError:
                e.ignore()
                return
            e.acceptProposedAction()
            self._update_ghost(e.position().toPoint())
        else:
            e.ignore()

    def dragMoveEvent(self, e) -> None:  # noqa: N802
        if self._drag_payload is None:
            e.ignore()
            return
        e.acceptProposedAction()
        self._update_ghost(e.position().toPoint())

    def dragLeaveEvent(self, e) -> None:  # noqa: N802
        self._drag_payload = None
        self._ghost = None
        self.update()

    def dropEvent(self, e) -> None:  # noqa: N802
        payload = self._drag_payload
        self._drag_payload = None
        self._ghost = None
        if payload is None:
            e.ignore()
            return
        pos, surface = self.placement_at(e.position().toPoint(), payload)
        e.acceptProposedAction()
        self.update()
        self.dropRequested.emit(json.dumps(payload), [float(c) for c in pos], surface)

    def placement_at(self, pos, payload: dict) -> tuple[np.ndarray, str | None]:
        mesh = self.ghost_provider(payload) if self.ghost_provider else None
        lo, hi = mesh.bounds() if mesh is not None else (np.zeros(3), np.zeros(3))
        o, d = self._ray(pos)
        sn = self.view.snap
        return picking.placement_position(self.scene, o, d, lo, hi, sn.work_plane, sn.work_plane_offset,
                                          sn.grid_snap, sn.translate_increment, sn.surface_snap,
                                          fallback=self.camera.target)

    def _update_ghost(self, pos) -> None:
        mesh: Mesh | None = self.ghost_provider(self._drag_payload) if self.ghost_provider else None
        if mesh is None:
            self._ghost = None
        else:
            p, _ = self.placement_at(pos, self._drag_payload)
            m = np.eye(4)
            m[:3, 3] = p
            self._ghost = GhostDraw(mesh, m)
        self.update()

    # ---- screenshots -----------------------------------------------------------------------------------------
    def grab_image(self):
        return self.grabFramebuffer()
