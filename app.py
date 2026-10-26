"""PSim Studio launcher.

    python app.py [project.psim] [--theme dark|light] [--template primitives|mechanical|simulation|formulas]
                  [--no-recovery] [--reset-settings]

Developer option (used by automated GUI checks):
    --screenshot out.png [--screenshot-delay ms]   render the window, save it and exit
"""
from __future__ import annotations

import argparse
import os
import sys

# PyOpenGL error checking costs a round trip per call; enable it only for debugging.
import OpenGL

OpenGL.ERROR_CHECKING = os.environ.get("PSIM_GL_DEBUG") == "1"

from PySide6.QtCore import QCoreApplication, QSettings, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QGuiApplication, QIcon, QSurfaceFormat  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from psim import APP_NAME, ORG_NAME, __version__  # noqa: E402


def configure_opengl(samples: int = 4) -> None:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(samples)
    QSurfaceFormat.setDefaultFormat(fmt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="psim-studio", description=f"{APP_NAME} {__version__}")
    parser.add_argument("project", nargs="?", help="project file to open")
    parser.add_argument("--theme", choices=("dark", "light"))
    parser.add_argument("--template", choices=("empty", "primitives", "mechanical", "simulation", "formulas"))
    parser.add_argument("--no-recovery", action="store_true", help="do not offer autosave recovery")
    parser.add_argument("--reset-settings", action="store_true", help="forget window layout and preferences")
    parser.add_argument("--screenshot", help=argparse.SUPPRESS)
    parser.add_argument("--screenshot-delay", type=int, default=1500, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    QCoreApplication.setOrganizationName(ORG_NAME)
    QCoreApplication.setApplicationName(APP_NAME)
    QCoreApplication.setApplicationVersion(__version__)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    configure_opengl()
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")

    settings = QSettings()
    if args.reset_settings:
        settings.clear()

    from psim.gui import icons, theme
    from psim.gui.main_window import MainWindow

    theme.apply(app, args.theme or str(settings.value("theme", "dark")))
    app.setWindowIcon(QIcon(icons.icon("cube", "#3d8bfd")))
    window = MainWindow(settings, check_recovery=not (args.no_recovery or args.screenshot))
    if args.template:
        window.new_project(template=args.template)
    if args.project:
        window.open_project(args.project, confirm=False)
    window.show()

    if args.screenshot:
        def capture() -> None:
            # window.grab() does not include QPainter overlays drawn on the GL widget, so the
            # viewport's own framebuffer grab (GL + overlays) is composited on top.
            from PySide6.QtGui import QPainter
            from PySide6.QtCore import QPoint
            img = window.grab()
            vp = window.viewport
            painter = QPainter(img)
            painter.drawImage(vp.mapTo(window, QPoint(0, 0)), vp.grabFramebuffer())
            painter.end()
            img.save(args.screenshot)
            window.undo.setClean()
            window._extra_dirty = False
            app.quit()
        QTimer.singleShot(args.screenshot_delay, capture)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
