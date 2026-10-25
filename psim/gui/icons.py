"""Vector icon set (24x24 stroke glyphs rendered from inline SVG, crisp at any DPI)."""
from __future__ import annotations

import os

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QIconEngine, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_P = {
    "new": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M12 11v6M9 14h6"/>',
    "open": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v1"/><path d="M3 7v11a2 2 0 0 0 2 2h12.5a2 2 0 0 0 1.9-1.4L22 11H7.4a2 2 0 0 0-1.9 1.4L3 19"/>',
    "save": '<path d="M5 3h11l5 5v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M7 3v5h8V3"/><rect x="7" y="13" width="10" height="8" rx="1"/>',
    "save_as": '<path d="M5 3h11l5 5v5"/><path d="M3 5v14a2 2 0 0 0 2 2h7"/><path d="M5 3a2 2 0 0 0-2 2"/><path d="M7 3v5h8V3"/><path d="M16.5 15.5l3-3 2 2-3 3-2.5.5z"/>',
    "undo": '<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    "redo": '<path d="M15 14l5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13"/>',
    "add": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/><path d="M18 2v4M16 4h4" stroke-width="1.6"/>',
    "select": '<path d="M5 3l14 7-6 2-2 6z"/>',
    "move": '<path d="M12 2v20M2 12h20"/><path d="M9 5l3-3 3 3M9 19l3 3 3-3M5 9l-3 3 3 3M19 9l3 3-3 3"/>',
    "rotate": '<path d="M20 12a8 8 0 1 1-2.3-5.6"/><path d="M20 4v5h-5"/>',
    "scale": '<rect x="3" y="11" width="10" height="10" rx="1"/><path d="M13 11V5M21 3l-8 8M15 3h6v6"/>',
    "grid": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>',
    "axes": '<path d="M5 19V4M5 19h15M5 19l-3 3"/><path d="M2 7l3-3 3 3M17 16l3 3-3 3"/>',
    "properties": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "camera": '<path d="M3 8a2 2 0 0 1 2-2h2l2-2h6l2 2h2a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><circle cx="12" cy="13" r="4"/>',
    "frame": '<path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3"/><circle cx="12" cy="12" r="3"/>',
    "fit": '<path d="M3 8V3h5M21 8V3h-5M3 16v5h5M21 16v5h-5"/><path d="M12 7l5 3v5l-5 3-5-3v-5z"/>',
    "origin": '<circle cx="12" cy="12" r="3"/><path d="M12 2v5M12 17v5M2 12h5M17 12h5"/>',
    "perspective": '<path d="M4 6l7-3 9 3v12l-9 3-7-3z"/><path d="M11 3v18M4 6l7 3 9-3M11 9v12"/>',
    "ortho": '<rect x="4" y="4" width="16" height="16" rx="1"/><path d="M4 12h16M12 4v16"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-5-5"/>',
    "trash": '<path d="M3 6h18M8 6V4h8v2M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14"/><path d="M10 11v6M14 11v6"/>',
    "duplicate": '<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M4 16V5a2 2 0 0 1 2-2h11"/>',
    "copy": '<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M4 16V5a2 2 0 0 1 2-2h11"/>',
    "paste": '<rect x="5" y="4" width="14" height="18" rx="2"/><path d="M9 4V3h6v1M9 10h6M9 14h6M9 18h3"/>',
    "cut": '<circle cx="6" cy="18" r="3"/><circle cx="18" cy="18" r="3"/><path d="M8 16L19 3M16 16L5 3"/>',
    "group": '<rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="13" width="8" height="8" rx="1"/><path d="M3 15v4a2 2 0 0 0 2 2h4M21 9V5a2 2 0 0 0-2-2h-4"/>',
    "ungroup": '<rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="13" width="8" height="8" rx="1"/><path d="M15 5h4v4M5 15v4h4" stroke-dasharray="2 2"/>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    "unlock": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 7.5-2"/>',
    "eye": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "eye_off": '<path d="M3 3l18 18"/><path d="M10.6 5.1A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-3.2 4M6.6 6.6A17 17 0 0 0 2 12s3.5 7 10 7a9.6 9.6 0 0 0 5.4-1.6"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "chevron_down": '<path d="M6 9l6 6 6-6"/>',
    "chevron_right": '<path d="M9 6l6 6-6 6"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "minus": '<path d="M5 12h14"/>',
    "edit": '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/>',
    "import": '<path d="M12 3v12M7 10l5 5 5-5"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
    "export": '<path d="M12 15V3M7 8l5-5 5 5"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
    "magnet": '<path d="M6 3v8a6 6 0 0 0 12 0V3"/><path d="M6 3h4v8a2 2 0 0 0 4 0V3h4"/><path d="M6 7h4M14 7h4"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
    "local": '<path d="M12 12l7-4M12 12V4M12 12l-6 5"/><circle cx="12" cy="12" r="1.5"/><rect x="3" y="3" width="18" height="18" rx="3" stroke-dasharray="3 3"/>',
    "bookmark": '<path d="M6 3h12v18l-6-4-6 4z"/>',
    "fullscreen": '<path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.8"/>',
    "tag": '<path d="M3 12V4a1 1 0 0 1 1-1h8l9 9-9 9z"/><circle cx="7.5" cy="7.5" r="1.5"/>',
    "terminal": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3M12 15h5"/>',
    "table": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M3 15h18M9 10v10"/>',
    "warning": '<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18h.01"/>',
    "error": '<circle cx="12" cy="12" r="9"/><path d="M15 9l-6 6M9 9l6 6"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5h.01"/>',
    "activity": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "cube": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "tree": '<rect x="3" y="3" width="6" height="5" rx="1"/><rect x="13" y="10" width="8" height="4" rx="1"/><rect x="13" y="17" width="8" height="4" rx="1"/><path d="M6 8v11h7M6 12h7"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "ruler": '<path d="M3 17L17 3l4 4L7 21z"/><path d="M7 13l2 2M10 10l2 2M13 7l2 2"/>',
    "keyboard": '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10"/>',
    "check": '<path d="M5 12l5 5 9-10"/>',
    "close": '<path d="M6 6l12 12M18 6L6 18"/>',
    "function": '<path d="M14 4h-1a3 3 0 0 0-3 3v10a3 3 0 0 1-3 3H6"/><path d="M7 11h8M15 14l5 5M20 14l-5 5"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/>',
    "play": '<path d="M7 4l13 8-13 8z"/>',
    "mass": '<path d="M7 8h10l3 12H4z"/><circle cx="12" cy="5" r="2"/>',
    "frame_axes": '<path d="M12 13V3M12 13l8 5M12 13l-8 5"/><path d="M10 5l2-2 2 2"/>',
    "align": '<path d="M4 3v18"/><rect x="7" y="6" width="12" height="4" rx="1"/><rect x="7" y="14" width="8" height="4" rx="1"/>',
}

_color = "#c3c8d0"
_on_color = "#ffffff"
_cache: dict[tuple, QIcon] = {}


def set_color(color: str, on_color: str) -> None:
    global _color, _on_color
    _color, _on_color = color, on_color
    _cache.clear()


def svg(name: str, color: str | None = None, stroke: float = 1.8) -> bytes:
    body = _P.get(name, _P["cube"])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color or _color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{body}</svg>').encode()


class _SvgEngine(QIconEngine):
    def __init__(self, name: str, color: str, disabled: str) -> None:
        super().__init__()
        self._name = name
        self._color = color
        self._disabled = disabled

    def paint(self, painter: QPainter, rect, mode, state) -> None:
        color = self._disabled if mode == QIcon.Mode.Disabled else self._color
        renderer = QSvgRenderer(QByteArray(svg(self._name, color)))
        renderer.render(painter, QRectF(rect))

    def pixmap(self, size: QSize, mode, state) -> QPixmap:
        pm = QPixmap(size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        self.paint(p, pm.rect(), mode, state)
        p.end()
        return pm

    def clone(self) -> "QIconEngine":
        return _SvgEngine(self._name, self._color, self._disabled)


def icon(name: str, color: str | None = None) -> QIcon:
    key = (name, color or _color)
    if key not in _cache:
        _cache[key] = QIcon(_SvgEngine(name, color or _color, "#6b717a"))
    return _cache[key]


def names() -> list[str]:
    return sorted(_P)


def write_svg_file(name: str, color: str, directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}-{color.strip('#')}.svg")
    with open(path, "wb") as fh:
        fh.write(svg(name, color, 3.0))
    return path
