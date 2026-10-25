"""Shared test configuration.

GUI tests need an OpenGL 3.3 capable display. On headless Linux run the suite
under a virtual X server:   xvfb-run -a python -m pytest
Tests that need OpenGL are skipped automatically when no context can be created.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import OpenGL  # noqa: E402

OpenGL.ERROR_CHECKING = True

from PySide6.QtCore import QStandardPaths  # noqa: E402

QStandardPaths.setTestModeEnabled(True)  # never touch the real user profile

import app as launcher  # noqa: E402

launcher.configure_opengl()


def _gl_available() -> bool:
    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    ctx = QOpenGLContext()
    if not ctx.create():
        return False
    surf = QOffscreenSurface()
    surf.create()
    ok = ctx.makeCurrent(surf) and ctx.format().majorVersion() * 10 + ctx.format().minorVersion() >= 33
    ctx.doneCurrent()
    return bool(ok)


@pytest.fixture(scope="session")
def gl_available(qapp) -> bool:
    return _gl_available()


@pytest.fixture
def requires_gl(gl_available):
    if not gl_available or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        pytest.skip("OpenGL 3.3 display not available (run under xvfb-run on headless Linux)")


@pytest.fixture
def scene(qapp):
    from psim.scene.editing import SceneEditor
    from psim.scene.scene_manager import Scene
    s = Scene()
    return s, SceneEditor(s)
