"""2D (pyqtgraph PlotWidget) and 3D (pyqtgraph.opengl GLViewWidget) views of a PlotSpec."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...recording.plot_spec import PlotSpec, SeriesData

pg.setConfigOptions(antialias=True)

_STYLES = {"solid": Qt.PenStyle.SolidLine, "dash": Qt.PenStyle.DashLine, "dot": Qt.PenStyle.DotLine,
           "dashdot": Qt.PenStyle.DashDotLine}
_SYMBOLS = {"o": "o", "s": "s", "t": "t", "d": "d", "+": "+", "x": "x"}


def _qcolor(rgb, alpha: float = 1.0) -> QColor:
    return QColor.fromRgbF(float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha)


class PlotView(QWidget):
    """Common interface of 2D and 3D plot views."""
    cursorMoved = Signal(float)   # simulated time

    def set_data(self, data: list[SeriesData], titles: dict[str, str]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def set_cursor(self, t: float | None) -> None:
        pass

    def export_image(self, path: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def grab_png(self) -> bytes:
        from PySide6.QtCore import QBuffer, QIODevice
        img = self.grab_image()
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        img.save(buf, "PNG")
        return bytes(buf.data())

    def grab_image(self):  # pragma: no cover - abstract
        raise NotImplementedError


class Plot2DView(PlotView):
    def __init__(self, spec: PlotSpec, parent=None) -> None:
        super().__init__(parent)
        self.spec = spec
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.widget = pg.PlotWidget()
        self.widget.setBackground(None)
        self.item = self.widget.getPlotItem()
        for side in ("bottom", "left"):   # titles already carry the unit; no extra "(x0.001)" factors
            self.item.getAxis(side).enableAutoSIPrefix(False)
        lay.addWidget(self.widget)
        self.note = QLabel()
        self.note.setObjectName("Muted")
        self.note.setWordWrap(True)
        self.note.hide()
        lay.addWidget(self.note)
        self.legend = self.item.addLegend(offset=(8, 8))
        self.curves: list[pg.PlotDataItem] = []
        self.cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen((230, 200, 90), width=1.5))
        self.cursor.sigPositionChanged.connect(self._cursor_dragged)
        self.readout = pg.TextItem(anchor=(1, 0), color=(220, 220, 220), fill=pg.mkBrush(20, 22, 26, 200))
        self.item.getViewBox().sigRangeChanged.connect(lambda *_a: self._place_readout())
        self._data: list[SeriesData] = []
        self.apply_spec(spec)

    def apply_spec(self, spec: PlotSpec) -> None:
        self.spec = spec
        it = self.item
        it.setTitle(spec.title)
        it.showGrid(x=spec.grid, y=spec.grid, alpha=0.25)
        it.setLogMode(spec.x_log, spec.y_log)
        self.legend.setVisible(spec.legend)
        if spec.x_limits:
            it.setXRange(*self._range(spec.x_limits, spec.x_log), padding=0)
        else:
            it.enableAutoRange(axis="x")
        if spec.y_limits:
            it.setYRange(*self._range(spec.y_limits, spec.y_log), padding=0)
        else:
            it.enableAutoRange(axis="y")
        if spec.x_is_time:
            if self.cursor.scene() is None:
                it.addItem(self.cursor, ignoreBounds=True)
                it.addItem(self.readout, ignoreBounds=True)
        elif self.cursor.scene() is not None:
            it.removeItem(self.cursor)
            it.removeItem(self.readout)

    @staticmethod
    def _range(limits, log: bool):
        a, b = limits
        if log:
            return (np.log10(a) if a > 0 else -12, np.log10(b) if b > 0 else -11)
        return a, b

    def set_data(self, data: list[SeriesData], titles: dict[str, str]) -> None:
        self._data = data
        self.item.setLabel("bottom", titles.get("x", ""))
        self.item.setLabel("left", titles.get("y", ""))
        while len(self.curves) > len(data):
            c = self.curves.pop()
            self.item.removeItem(c)
            self.legend.removeItem(c)
        problems = []
        for k, d in enumerate(data):
            s = d.spec
            pen = None if s.style == "none" else pg.mkPen(_qcolor(s.color), width=s.width, style=_STYLES[s.style])
            symbol = _SYMBOLS.get(s.marker)
            if k >= len(self.curves):
                c = pg.PlotDataItem()
                self.item.addItem(c)
                self.curves.append(c)
            c = self.curves[k]
            x, y = d.x, d.y
            if self.spec.x_log or self.spec.y_log:
                ok = np.ones(len(x), bool)
                if self.spec.x_log:
                    ok &= x > 0
                if self.spec.y_log:
                    ok &= y > 0
                if not ok.all() and len(x):
                    problems.append(f"{d.label}: {int((~ok).sum())} non-positive values hidden on the log axis")
                x, y = x[ok], y[ok]
            c.setData(x, y, pen=pen, symbol=symbol, symbolSize=s.marker_size, symbolPen=None,
                      symbolBrush=_qcolor(s.color) if symbol else None, connect="finite")
            self.legend.removeItem(c)
            if self.spec.legend:
                self.legend.addItem(c, d.label)
            if d.error:
                problems.append(f"{d.label}: {d.error}")
        self.note.setText(" · ".join(problems))
        self.note.setVisible(bool(problems))
        self._update_readout()

    def _cursor_dragged(self, line) -> None:
        self._update_readout()
        self.cursorMoved.emit(float(line.value()))

    def set_cursor(self, t: float | None) -> None:
        if t is None or not self.spec.x_is_time:
            return
        self.cursor.blockSignals(True)
        self.cursor.setValue(t)
        self.cursor.blockSignals(False)
        self._update_readout()

    def _update_readout(self) -> None:
        if not self.spec.x_is_time or not self._data:
            return
        from ...recording.plot_spec import values_at_time
        t = float(self.cursor.value())
        rows = values_at_time(self._data, t)
        self.readout.setText("\n".join([f"t = {t:.6g}"] + [f"{name}: {y:.6g}" for name, _x, y in rows]))
        self._place_readout()

    def _place_readout(self) -> None:
        (_x0, x1), (_y0, y1) = self.item.viewRange()
        self.readout.setPos(x1, y1)

    def export_image(self, path: str) -> None:
        from pyqtgraph.exporters import ImageExporter
        exp = ImageExporter(self.item)
        exp.parameters()["width"] = max(800, self.widget.width())
        exp.export(path)

    def grab_image(self):
        return self.widget.grab().toImage()


class Plot3DView(PlotView):
    """X/Y/Z plot. Each axis is scaled to a common box; tick labels show the real values."""
    BOX = 10.0

    def __init__(self, spec: PlotSpec, parent=None) -> None:
        super().__init__(parent)
        self.spec = spec
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel(spec.title)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.title)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((24, 26, 30))
        self.view.opts["center"] = pg.Vector(self.BOX / 2, self.BOX / 2, self.BOX / 2)
        self.view.setCameraPosition(distance=30, elevation=24, azimuth=-60)
        self._fitted = False
        lay.addWidget(self.view, 1)
        self.note = QLabel()
        self.note.setObjectName("Muted")
        self.note.setWordWrap(True)
        self.note.hide()
        lay.addWidget(self.note)
        self.legend = QLabel()
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.legend)
        self._static: list = []
        self._items: list = []
        self._cursor_items: list = []
        self._data: list[SeriesData] = []
        self._scale = None

    def apply_spec(self, spec: PlotSpec) -> None:
        self.spec = spec
        self.title.setText(spec.title)

    def fit(self) -> None:
        """Camera distance that shows the whole box (pyqtgraph's field of view is horizontal)."""
        w, h = max(self.view.width(), 1), max(self.view.height(), 1)
        half = np.tan(np.radians(self.view.opts.get("fov", 60)) / 2)
        self.view.setCameraPosition(distance=0.8 * self.BOX / half * max(1.0, w / h))
        self._fitted = True

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.fit()

    def _clear(self, items: list) -> None:
        for it in items:
            self.view.removeItem(it)
        items.clear()

    def set_data(self, data: list[SeriesData], titles: dict[str, str]) -> None:
        self._data = data
        self._clear(self._items)
        self._clear(self._static)
        good = [d for d in data if not d.error and len(d.x)]
        lo = np.array([np.nanmin(np.concatenate([d.x for d in good])) if good else 0.0,
                       np.nanmin(np.concatenate([d.y for d in good])) if good else 0.0,
                       np.nanmin(np.concatenate([d.z for d in good])) if good else 0.0])
        hi = np.array([np.nanmax(np.concatenate([d.x for d in good])) if good else 1.0,
                       np.nanmax(np.concatenate([d.y for d in good])) if good else 1.0,
                       np.nanmax(np.concatenate([d.z for d in good])) if good else 1.0])
        lo = np.where(np.isfinite(lo), lo, 0.0)
        hi = np.where(np.isfinite(hi), hi, 1.0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        lo = np.where(hi - lo > 1e-12, lo, lo - 0.5)
        self._scale = (lo, span)
        self._draw_frame(lo, span, titles)
        for d in good:
            pts = self._to_box(np.c_[d.x, d.y, d.z])
            pts = pts[np.isfinite(pts).all(axis=1)]
            color = (*d.spec.color, 1.0)
            if d.spec.style != "none" and len(pts) > 1:
                it = gl.GLLinePlotItem(pos=pts, color=color, width=d.spec.width, antialias=True, mode="line_strip")
                self.view.addItem(it)
                self._items.append(it)
            if d.spec.marker != "none" and len(pts):
                it = gl.GLScatterPlotItem(pos=pts, color=color, size=d.spec.marker_size)
                self.view.addItem(it)
                self._items.append(it)
        problems = [f"{d.label}: {d.error}" for d in data if d.error]
        self.note.setText(" · ".join(problems))
        self.note.setVisible(bool(problems))
        if not self._fitted:
            self.fit()
        self.legend.setText("  ".join(f"<span style='color:{_qcolor(d.spec.color).name()}'>■</span> {d.label}"
                                      for d in data) if self.spec.legend else "")

    def _to_box(self, xyz: np.ndarray) -> np.ndarray:
        lo, span = self._scale
        return (xyz - lo) / span * self.BOX

    def _draw_frame(self, lo, span, titles) -> None:
        b = self.BOX
        if self.spec.grid:
            for plane in ("xy", "xz", "yz"):
                g = gl.GLGridItem()
                g.setSize(b, b)
                g.setSpacing(b / 5, b / 5)
                g.setColor((90, 95, 105, 120))
                if plane == "xy":
                    g.translate(b / 2, b / 2, 0)
                elif plane == "xz":
                    g.rotate(90, 1, 0, 0)
                    g.translate(b / 2, b, b / 2)
                else:
                    g.rotate(90, 0, 1, 0)
                    g.translate(0, b / 2, b / 2)
                self.view.addItem(g)
                self._static.append(g)
        axes_colors = ((235, 90, 80), (110, 205, 85), (85, 140, 245))
        font = QFont()
        font.setPointSize(8)
        for i, (name, col) in enumerate(zip("xyz", axes_colors)):
            end = np.zeros(3)
            end[i] = b
            line = gl.GLLinePlotItem(pos=np.array([np.zeros(3), end]), color=(*[c / 255 for c in col], 1.0), width=2)
            self.view.addItem(line)
            self._static.append(line)
            offset = {0: np.array([0.0, -0.06, 0.0]), 1: np.array([-0.06, 0.0, 0.0]),
                      2: np.array([-0.06, -0.06, 0.0])}[i] * b   # keep tick labels of different axes apart
            for k in range(6):
                p = offset.copy()
                p[i] = b * k / 5
                value = lo[i] + span[i] * k / 5
                t = gl.GLTextItem(pos=p, text=f"{value:.4g}", color=(200, 200, 200, 255), font=font)
                self.view.addItem(t)
                self._static.append(t)
            title_pos = end * 1.12
            title_font = QFont()
            title_font.setPointSize(9)
            t = gl.GLTextItem(pos=title_pos, text=titles.get(name, name), color=(*col, 255), font=title_font)
            self.view.addItem(t)
            self._static.append(t)

    def set_cursor(self, t: float | None) -> None:
        self._clear(self._cursor_items)
        if t is None or self._scale is None:
            return
        pts, cols = [], []
        for d in self._data:
            if d.error or not len(d.t):
                continue
            i = int(np.clip(np.searchsorted(d.t, t), 0, len(d.t) - 1))
            pts.append([d.x[i], d.y[i], d.z[i]])
            cols.append((*d.spec.color, 1.0))
        if pts:
            it = gl.GLScatterPlotItem(pos=self._to_box(np.array(pts)), color=np.array(cols), size=14)
            self.view.addItem(it)
            self._cursor_items.append(it)

    def export_image(self, path: str) -> None:
        self.grab_image().save(path)

    def grab_image(self):
        return self.view.grabFramebuffer()
