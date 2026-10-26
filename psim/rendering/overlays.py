"""2D overlays painted over the 3D frame with QPainter: labels, axis labels, view triad, HUD."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen

from .. import numerics
from ..scene.object_base import SceneObject
from ..scene.scene_manager import Scene, SceneError
from ..units.unit_registry import UnitRegistry
from . import grid as grid_geo
from .camera import OrbitCamera
from .view_settings import ViewSettings


def qcolor(rgb, alpha: float = 1.0) -> QColor:
    c = QColor.fromRgbF(float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))
    return c


def label_unit(obj: SceneObject | None, quantity: str | None, units: UnitRegistry) -> str | None:
    """The object's own display unit for a quantity in labels (if valid), else the project unit."""
    if not quantity:
        return None
    sym = obj.label_units.get(quantity) if obj is not None else None
    if sym and units.has_unit(sym) and sym in units.units_for(quantity):
        return sym
    return units.preferred_unit(quantity)


def fmt_value(value, quantity: str | None, units: UnitRegistry, precision: int, obj: SceneObject | None = None) -> str:
    unit = label_unit(obj, quantity, units)
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = units.from_si(np.asarray(value, float), unit) if unit else np.asarray(value, float)
        body = "(" + ", ".join(f"{v:.{precision}f}" for v in vals) + ")"
    else:
        v = units.from_si(float(value), unit) if unit else value
        body = f"{v:.{precision}f}" if isinstance(v, float) else str(v)
    return f"{body} {unit}" if unit and unit != "1" else body


def object_label_lines(scene: Scene, obj: SceneObject, units: UnitRegistry) -> list[str]:
    ov = obj.overlays
    allowed = set(obj.typedef.overlays)
    prec = obj.label_precision
    lines: list[str] = []
    if ov.get("name") and "name" in allowed:
        lines.append(obj.name)
    if ov.get("id") and "id" in allowed:
        lines.append(obj.id)
    p, q = scene.world_pose(obj.id)
    if ov.get("position") and "position" in allowed:
        lines.append("p = " + fmt_value(p, "length", units, prec, obj))
    if ov.get("orientation") and "orientation" in allowed:
        lines.append("θ = " + fmt_value(numerics.quat_to_euler(q), "angle", units, prec, obj))
    runtime = scene.runtime.get(obj.id, {})
    for key, sym, attr, qty in (("velocity", "v", "linear_velocity", "velocity"),
                                ("acceleration", "a", "linear_acceleration", "acceleration"),
                                ("angular_velocity", "ω", "angular_velocity", "angular_velocity"),
                                ("angular_acceleration", "α", "angular_acceleration", "angular_acceleration"),
                                ("force", "F", "force", "force"), ("torque", "τ", "torque", "torque")):
        if ov.get(key) and key in allowed:
            # while simulating, show the current total accelerations rather than the input properties
            value = runtime.get(key) if key in ("acceleration", "angular_acceleration") else None
            if value is None:
                value = getattr(obj, attr)
            lines.append(f"{sym} = " + fmt_value(value, qty, units, prec, obj))
    if ov.get("mass") and "mass" in allowed and obj.typedef.physical:
        m = obj.mass()
        lines.append("m = " + (fmt_value(m, "mass", units, prec, obj) if m is not None else "—"))
    if ov.get("custom") and "custom" in allowed:
        for cp in obj.custom.values():
            d = cp.definition
            val = fmt_value(cp.value, d.quantity, units, d.precision, obj) \
                if d.vtype.value in ("float", "int", "vec2", "vec3") \
                else str(cp.value)
            lines.append(f"{d.label} = {val}")
    return lines


def paint_overlays(p: QPainter, scene: Scene, view: ViewSettings, cam: OrbitCamera, units: UnitRegistry,
                   width: float, height: float, hud_lines: list[str], drag_text: str = "",
                   trajectories: list | None = None) -> None:
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    base_font = QFont(p.font())

    # ---- grid labels -----------------------------------------------------------------------------
    small = QFont(base_font)
    small.setPointSizeF(max(base_font.pointSizeF() - 1.5, 7.0))
    p.setFont(small)
    for key, gs in view.grids.items():
        if not (gs.visible and gs.show_labels):
            continue
        items = grid_geo.grid_labels(gs)
        if not items:
            continue
        xy, ok = cam.project(np.array([w for w, _ in items]), width, height)
        p.setPen(qcolor(gs.color, min(1.0, gs.opacity + 0.35)))
        for (_, text), (x, y), vis in zip(items, xy, ok):
            if vis and 0 <= x <= width and 0 <= y <= height:
                p.drawText(QPointF(x + 3, y - 3), text)

    # ---- axis labels -----------------------------------------------------------------------------------
    ax = view.axes
    if ax.visible and ax.show_labels:
        bold = QFont(base_font)
        bold.setBold(True)
        p.setFont(bold)
        pts = np.eye(3) * ax.length * 1.08 + np.asarray(ax.origin, float)
        xy, ok = cam.project(pts, width, height)
        for (x, y), vis, name, col in zip(xy, ok, "XYZ", (ax.x_color, ax.y_color, ax.z_color)):
            if vis:
                p.setPen(qcolor(col))
                p.drawText(QRectF(x - 10, y - 10, 20, 20), Qt.AlignmentFlag.AlignCenter, name)

    # ---- trajectory arrows and time labels -------------------------------------------------------------
    if trajectories:
        _paint_trajectory_marks(p, trajectories, cam, units, width, height, small)

    # ---- named reference frames ------------------------------------------------------------------------
    if view.environment.show_labels:
        _paint_frame_labels(p, scene, cam, width, height, small)
    _paint_measurement_labels(p, scene, cam, units, width, height, small)

    # ---- object labels -------------------------------------------------------------------------------
    if view.environment.show_labels:
        for obj in scene.objects.values():
            if not obj.overlays or not any(obj.overlays.values()) or not scene.is_effectively_visible(obj.id):
                continue
            lines = object_label_lines(scene, obj, units)
            if obj.overlays.get("com") and obj.typedef.physical:
                _paint_com_symbol(p, scene, obj, cam, width, height)
            if obj.overlays.get("local_axes") or view.environment.show_local_axes:
                _paint_local_axis_labels(p, scene, obj, cam, width, height, small)
            if not lines:
                continue
            wp, _ = scene.world_pose(obj.id)
            top = np.array(wp, float)
            lo, hi = scene.world_bounds([obj.id]) or (wp, wp)
            top[2] = hi[2]
            placement = obj.label_placement
            anchor = np.array(wp, float)
            if placement == "below":
                anchor[2] = lo[2]
            elif placement == "above":
                anchor = top
            xy, ok = cam.project(anchor[None, :], width, height)
            if not ok[0]:
                continue
            f = QFont(base_font)
            f.setPointSizeF(obj.label_font_size)
            p.setFont(f)
            fm = QFontMetricsF(f)
            lh = fm.height()
            tw = max(fm.horizontalAdvance(t) for t in lines)
            bw, bh = tw + 12, lh * len(lines) + 8
            x0, y0 = xy[0]
            if placement == "below":
                box = QRectF(x0 - bw / 2, y0 + 12, bw, bh)
            elif placement == "left":
                box = QRectF(x0 - bw - 18, y0 - bh / 2, bw, bh)
            elif placement == "right":
                box = QRectF(x0 + 18, y0 - bh / 2, bw, bh)
            else:
                box = QRectF(x0 - bw / 2, y0 - 12 - bh + 2, bw, bh)
            if obj.label_background:
                p.setPen(QPen(QColor(255, 255, 255, 28), 1))
                p.setBrush(QColor(18, 20, 24, 190))
                p.drawRoundedRect(box, 4, 4)
            p.setPen(qcolor(obj.label_color))
            for k, t in enumerate(lines):
                p.drawText(QRectF(box.left() + 6, box.top() + 4 + k * lh, tw, lh), Qt.AlignmentFlag.AlignLeft, t)

    # ---- view orientation triad -------------------------------------------------------------------------
    if ax.show_view_triad:
        _paint_triad(p, cam, ax, height, base_font)

    # ---- HUD ------------------------------------------------------------------------------------------------
    p.setFont(base_font)
    fm = QFontMetricsF(base_font)
    y = 10.0
    for line in hud_lines:
        if not line:
            continue
        tw = fm.horizontalAdvance(line)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(15, 16, 19, 150))
        p.drawRoundedRect(QRectF(10, y, tw + 14, fm.height() + 6), 4, 4)
        p.setPen(QColor(205, 210, 218))
        p.drawText(QPointF(17, y + 3 + fm.ascent()), line)
        y += fm.height() + 9
    if drag_text:
        tw = fm.horizontalAdvance(drag_text)
        r = QRectF(width / 2 - tw / 2 - 10, height - fm.height() - 26, tw + 20, fm.height() + 10)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(15, 16, 19, 200))
        p.drawRoundedRect(r, 5, 5)
        p.setPen(QColor(250, 200, 120))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, drag_text)


def _paint_com_symbol(p: QPainter, scene: Scene, obj: SceneObject, cam: OrbitCamera, w: float, h: float) -> None:
    com = scene.world_matrix(obj.id) @ np.append(obj.mass_properties()["com"], 1.0)
    xy, ok = cam.project(com[None, :3], w, h)
    if not ok[0]:
        return
    c = QPointF(*xy[0])
    r = 7.0
    p.setPen(QPen(QColor(20, 20, 20), 1.2))
    for k, col in enumerate((QColor(245, 245, 245), QColor(25, 25, 25), QColor(245, 245, 245), QColor(25, 25, 25))):
        p.setBrush(col)
        p.drawPie(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), k * 90 * 16, 90 * 16)


def _paint_trajectory_marks(p: QPainter, paths: list, cam: OrbitCamera, units: UnitRegistry, w: float, h: float,
                            font: QFont) -> None:
    from .trajectories import marks
    tu = units.preferred_unit("time")
    p.setFont(font)
    for path in paths:
        if len(path.points) < 2:
            continue
        color = qcolor(path.color, min(1.0, path.opacity + 0.1))
        if path.arrows:
            idx = marks(path, path.arrow_spacing)
            nxt = [min(i + 1, len(path.points) - 1) if i + 1 < len(path.points) else i - 1 for i in idx]
            xy, ok = cam.project(np.vstack([path.points[idx], path.points[nxt]]), w, h)
            p.setPen(QPen(color, max(1.5, path.width)))
            k = len(idx)
            for j, i in enumerate(idx):
                if not (ok[j] and ok[k + j]):
                    continue
                a, b = np.asarray(xy[j], float), np.asarray(xy[k + j], float)
                d = b - a if nxt[j] > i else a - b
                n = float(np.hypot(*d))
                if n < 1e-6:
                    continue
                d /= n
                nrm = np.array([-d[1], d[0]])
                tip = a + d * 5
                for s in (1, -1):
                    back = tip - d * 9 + nrm * 5 * s
                    p.drawLine(QPointF(*tip), QPointF(*back))
        if path.time_labels:
            idx = marks(path, path.label_spacing)
            xy, ok = cam.project(path.points[idx], w, h)
            p.setPen(color)
            for (x, y), vis, i in zip(xy, ok, idx):
                if vis and 0 <= x <= w and 0 <= y <= h:
                    p.setBrush(color)
                    p.drawEllipse(QPointF(x, y), 2.5, 2.5)
                    p.drawText(QPointF(x + 5, y - 4), f"{units.from_si(float(path.times[i]), tu):.4g} {tu}")


def _paint_measurement_labels(p: QPainter, scene: Scene, cam: OrbitCamera, units: UnitRegistry, w: float, h: float,
                              font: QFont) -> None:
    from .physics_overlay import MEASURE_COLOR, measurement_labels
    items = measurement_labels(scene, units)
    if not items:
        return
    xy, ok = cam.project(np.array([a for a, _ in items]), w, h)
    p.setFont(font)
    fm = p.fontMetrics()
    for (x, y), vis, (_a, text) in zip(xy, ok, items):
        if not vis:
            continue
        rect = fm.boundingRect(text).adjusted(-4, -2, 4, 2)
        rect.moveCenter(QPointF(x, y - 12).toPoint())
        p.fillRect(rect, qcolor((0.08, 0.09, 0.1), 0.75))
        p.setPen(qcolor(MEASURE_COLOR))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


def _paint_frame_labels(p: QPainter, scene: Scene, cam: OrbitCamera, w: float, h: float, font: QFont) -> None:
    for fr in scene.frames:
        if not (fr.visible and fr.show_label):
            continue
        try:
            pos, q = scene.frame_world_pose(fr.id)
        except SceneError:
            continue
        r = np.asarray(numerics.quat_to_matrix(q))
        pts = np.vstack([pos, *(pos + r[:, i] * fr.axis_length * 1.1 for i in range(3))])
        xy, ok = cam.project(pts, w, h)
        p.setFont(font)
        if ok[0]:
            p.setPen(qcolor(fr.color))
            p.drawText(QPointF(xy[0][0] + 6, xy[0][1] + 14), fr.name)
        for (x, y), vis, name, col in zip(xy[1:], ok[1:], ("x", "y", "z"),
                                           ((0.95, 0.45, 0.42), (0.55, 0.86, 0.45), (0.45, 0.65, 1.0))):
            if vis:
                p.setPen(qcolor(col))
                p.drawText(QPointF(x + 2, y - 2), name)


def _paint_local_axis_labels(p, scene, obj, cam, w, h, font) -> None:
    if "local_axes" not in obj.typedef.overlays:
        return
    pos, q = scene.world_pose(obj.id)
    r = np.asarray(numerics.quat_to_matrix(q))
    pts = np.array([pos + r[:, i] * obj.axes_length() * 1.08 for i in range(3)])
    xy, ok = cam.project(pts, w, h)
    p.setFont(font)
    for (x, y), vis, name, col in zip(xy, ok, ("x'", "y'", "z'"),
                                       ((0.95, 0.45, 0.42), (0.55, 0.86, 0.45), (0.45, 0.65, 1.0))):
        if vis:
            p.setPen(qcolor(col))
            p.drawText(QPointF(x + 2, y - 2), name)


def _paint_triad(p: QPainter, cam: OrbitCamera, ax, height: float, font: QFont) -> None:
    center = QPointF(52, height - 52)
    size = 34.0
    view = cam.view_matrix()[:3, :3]
    axes = []
    for i, (name, col) in enumerate(zip("XYZ", (ax.x_color, ax.y_color, ax.z_color))):
        v = view[:, i]
        axes.append((v[2], name, col, v))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(15, 16, 19, 120))
    p.drawEllipse(center, size + 14, size + 14)
    bold = QFont(font)
    bold.setBold(True)
    bold.setPointSizeF(max(font.pointSizeF() - 1, 7.0))
    p.setFont(bold)
    for _, name, col, v in sorted(axes, key=lambda a: a[0]):
        end = QPointF(center.x() + v[0] * size, center.y() - v[1] * size)
        p.setPen(QPen(qcolor(col), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(center, end)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(col))
        p.drawEllipse(end, 7.5, 7.5)
        p.setPen(QColor(20, 20, 22))
        p.drawText(QRectF(end.x() - 7.5, end.y() - 7.5, 15, 15), Qt.AlignmentFlag.AlignCenter, name)


def paint_drop_hint(p: QPainter, width: float, height: float, text: str) -> None:
    path = QPainterPath()
    path.addRoundedRect(QRectF(6, 6, width - 12, height - 12), 8, 8)
    p.setPen(QPen(QColor(90, 160, 255, 160), 2, Qt.PenStyle.DashLine))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    fm = QFontMetricsF(p.font())
    r = QRectF(width / 2 - fm.horizontalAdvance(text) / 2 - 10, 16, fm.horizontalAdvance(text) + 20, fm.height() + 8)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(20, 40, 70, 210))
    p.drawRoundedRect(r, 5, 5)
    p.setPen(QColor(200, 225, 255))
    p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
