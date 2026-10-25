"""Application themes (graphite dark and light) applied via palette + style sheet."""
from __future__ import annotations

import os
from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Theme:
    name: str
    window: str
    panel: str
    panel_alt: str
    input: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_disabled: str
    accent: str
    accent_hover: str
    accent_text: str
    selection: str
    hover: str
    danger: str
    warning: str
    success: str
    icon: str


DARK = Theme(
    name="dark", window="#1b1d20", panel="#222428", panel_alt="#26292e", input="#17191c", border="#30343a",
    border_strong="#3c4148", text="#d7dae0", text_muted="#8d949e", text_disabled="#5a6069", accent="#3d8bfd",
    accent_hover="#5b9dff", accent_text="#ffffff", selection="#2c4a75", hover="#2c3036", danger="#e5534b",
    warning="#d9a33a", success="#4fb36b", icon="#c3c8d0",
)

LIGHT = Theme(
    name="light", window="#eceef1", panel="#f6f7f9", panel_alt="#ffffff", input="#ffffff", border="#d3d7dd",
    border_strong="#bcc2ca", text="#1f2328", text_muted="#5f6670", text_disabled="#a3a9b1", accent="#1f6feb",
    accent_hover="#2f7ff8", accent_text="#ffffff", selection="#cfe0fb", hover="#e3e7ec", danger="#cf222e",
    warning="#9a6700", success="#1a7f37", icon="#3b4149",
)

THEMES = {"dark": DARK, "light": LIGHT}
_current = DARK


def current() -> Theme:
    return _current


def stylesheet(t: Theme, check_icon: str = "") -> str:
    return f"""
* {{ outline: none; }}
QWidget {{ color: {t.text}; font-size: 9pt; }}
QMainWindow, QDialog {{ background: {t.window}; }}
QMainWindow::separator {{ background: {t.window}; width: 4px; height: 4px; }}
QMainWindow::separator:hover {{ background: {t.accent}; }}
QToolTip {{ background: {t.panel_alt}; color: {t.text}; border: 1px solid {t.border_strong}; padding: 4px 6px; }}

QMenuBar {{ background: {t.window}; border-bottom: 1px solid {t.border}; padding: 1px 4px; }}
QMenuBar::item {{ padding: 4px 9px; border-radius: 4px; background: transparent; }}
QMenuBar::item:selected {{ background: {t.hover}; }}
QMenu {{ background: {t.panel_alt}; border: 1px solid {t.border_strong}; padding: 4px; }}
QMenu::item {{ padding: 5px 26px 5px 26px; border-radius: 4px; }}
QMenu::item:selected {{ background: {t.accent}; color: {t.accent_text}; }}
QMenu::item:disabled {{ color: {t.text_disabled}; }}
QMenu::separator {{ height: 1px; background: {t.border}; margin: 4px 6px; }}
QMenu::icon {{ padding-left: 6px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 6px; }}

QToolBar {{ background: {t.window}; border: none; border-bottom: 1px solid {t.border}; spacing: 2px; padding: 3px 6px; }}
QToolBar::separator {{ background: {t.border}; width: 1px; margin: 4px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 5px; padding: 4px; }}
QToolButton:hover {{ background: {t.hover}; border-color: {t.border}; }}
QToolButton:pressed {{ background: {t.border}; }}
QToolButton:checked {{ background: {t.selection}; border-color: {t.accent}; }}
QToolButton:disabled {{ color: {t.text_disabled}; }}
QToolButton[popupMode="1"] {{ padding-right: 14px; }}
QToolButton::menu-button {{ border: none; width: 12px; }}

QDockWidget {{ titlebar-close-icon: none; color: {t.text_muted}; }}
QDockWidget::title {{ background: {t.window}; padding: 6px 8px; border-bottom: 1px solid {t.border};
    text-transform: uppercase; font-weight: 600; }}
QDockWidget > QWidget {{ background: {t.panel}; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {t.border}; background: {t.panel}; }}
QTabBar::tab {{ background: transparent; color: {t.text_muted}; padding: 6px 12px; border: none;
    border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {t.text}; border-bottom: 2px solid {t.accent}; }}
QTabBar::tab:hover {{ color: {t.text}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QKeySequenceEdit {{
    background: {t.input}; border: 1px solid {t.border}; border-radius: 4px; padding: 3px 6px;
    selection-background-color: {t.accent}; selection-color: {t.accent_text}; min-height: 18px; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {t.accent}; }}
QLineEdit:read-only {{ background: {t.panel_alt}; color: {t.text_muted}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {t.text_disabled}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; width: 0px; border: none; }}
QComboBox::drop-down {{ border: none; width: 16px; }}
QComboBox QAbstractItemView {{ background: {t.panel_alt}; border: 1px solid {t.border_strong};
    selection-background-color: {t.accent}; selection-color: {t.accent_text}; padding: 2px; }}

QPushButton {{ background: {t.panel_alt}; border: 1px solid {t.border_strong}; border-radius: 4px; padding: 4px 12px; }}
QPushButton:hover {{ background: {t.hover}; }}
QPushButton:pressed {{ background: {t.border}; }}
QPushButton:default {{ background: {t.accent}; border-color: {t.accent}; color: {t.accent_text}; }}
QPushButton:default:hover {{ background: {t.accent_hover}; }}
QPushButton:disabled {{ color: {t.text_disabled}; }}
QPushButton[flat="true"] {{ background: transparent; border: none; }}

QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
QCheckBox::indicator {{ border: 1px solid {t.border_strong}; border-radius: 3px; background: {t.input}; }}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent};
    image: url("{check_icon}"); }}
QCheckBox::indicator:hover {{ border-color: {t.accent}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {t.border_strong}; border-radius: 4px; min-height: 24px; min-width: 24px; margin: 2px; }}
QScrollBar::handle:hover {{ background: {t.text_disabled}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QTreeView, QTableView, QListView {{ background: {t.panel}; alternate-background-color: {t.panel_alt};
    border: none; selection-background-color: {t.selection}; selection-color: {t.text}; }}
QTreeView::item, QListView::item {{ padding: 2px; border-radius: 3px; }}
QTreeView::item:hover, QListView::item:hover {{ background: {t.hover}; }}
QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {{ background: {t.selection}; color: {t.text}; }}
QHeaderView::section {{ background: {t.window}; color: {t.text_muted}; border: none; border-right: 1px solid {t.border};
    border-bottom: 1px solid {t.border}; padding: 4px 6px; }}

QStatusBar {{ background: {t.window}; border-top: 1px solid {t.border}; color: {t.text_muted}; }}
QStatusBar QLabel {{ color: {t.text_muted}; padding: 0 6px; }}
QSplitter::handle {{ background: {t.border}; }}
QGroupBox {{ border: 1px solid {t.border}; border-radius: 6px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t.text_muted}; }}
QSlider::groove:horizontal {{ height: 4px; background: {t.border_strong}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t.accent}; width: 12px; margin: -5px 0; border-radius: 6px; }}

/* custom widgets */
#SectionHeader {{ background: {t.panel_alt}; border: none; border-top: 1px solid {t.border};
    border-bottom: 1px solid {t.border}; text-align: left; padding: 5px 6px; font-weight: 600; }}
#SectionHeader:hover {{ background: {t.hover}; }}
#InspectorTitle {{ font-size: 11pt; font-weight: 600; }}
#Muted {{ color: {t.text_muted}; }}
#Badge {{ border-radius: 3px; padding: 0 3px; font-size: 7pt; font-weight: 700; }}
#AxisLabelX {{ color: #e8524f; font-weight: 700; }}
#AxisLabelY {{ color: #6dcc52; font-weight: 700; }}
#AxisLabelZ {{ color: #538cf6; font-weight: 700; }}
#CategoryHeader {{ background: transparent; border: none; text-align: left; padding: 6px 4px; color: {t.text_muted};
    font-weight: 600; }}
#CategoryHeader:hover {{ color: {t.text}; }}
#LibraryList {{ background: transparent; }}
#LibraryList::item {{ border: 1px solid transparent; border-radius: 6px; padding: 4px; color: {t.text}; }}
#LibraryList::item:hover {{ background: {t.hover}; border-color: {t.border_strong}; }}
#LibraryList::item:selected {{ background: {t.selection}; border-color: {t.accent}; }}
#ViewportFrame {{ border: 1px solid {t.border}; }}
"""


def apply(app: QApplication, name: str) -> Theme:
    global _current
    t = THEMES.get(name, DARK)
    _current = t
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: t.window, QPalette.ColorRole.WindowText: t.text, QPalette.ColorRole.Base: t.input,
        QPalette.ColorRole.AlternateBase: t.panel_alt, QPalette.ColorRole.ToolTipBase: t.panel_alt,
        QPalette.ColorRole.ToolTipText: t.text, QPalette.ColorRole.Text: t.text, QPalette.ColorRole.Button: t.panel_alt,
        QPalette.ColorRole.ButtonText: t.text, QPalette.ColorRole.Highlight: t.accent,
        QPalette.ColorRole.HighlightedText: t.accent_text, QPalette.ColorRole.Link: t.accent,
        QPalette.ColorRole.PlaceholderText: t.text_disabled, QPalette.ColorRole.Mid: t.border,
        QPalette.ColorRole.Dark: t.border_strong, QPalette.ColorRole.Light: t.hover,
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(t.text_disabled))
    app.setPalette(pal)
    from . import icons
    icons.set_color(t.icon, t.accent_text)
    from ..project.autosave import data_dir
    check = icons.write_svg_file("check", t.accent_text, os.path.join(data_dir(), "ui")).replace("\\", "/")
    app.setStyleSheet(stylesheet(t, check))
    return t
