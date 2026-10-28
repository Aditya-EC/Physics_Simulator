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

from psim import app as launcher  # noqa: E402

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


# Hang diagnosis (CI): with PSIM_HANG_DUMP=<file>, a test running longer than PSIM_TEST_TIMEOUT seconds (default 300)
# makes faulthandler write the stacks of all threads, after the test's id, to that file and end the process.
_HANG_DUMP = os.environ.get("PSIM_HANG_DUMP")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if not _HANG_DUMP:
        yield
        return
    import faulthandler
    fh = open(_HANG_DUMP, "w", encoding="utf-8")
    fh.write(f"Test running too long: {item.nodeid}\n")
    fh.flush()
    faulthandler.dump_traceback_later(float(os.environ.get("PSIM_TEST_TIMEOUT", "300")), exit=True, file=fh)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        fh.close()
        os.remove(_HANG_DUMP)


@pytest.fixture(scope="session")
def gl_available(qapp) -> bool:
    return _gl_available()


@pytest.fixture
def requires_gl(gl_available):
    if not gl_available or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        if os.environ.get("PSIM_REQUIRE_GL"):        # CI jobs that provide OpenGL must not skip silently
            pytest.fail("PSIM_REQUIRE_GL is set but no OpenGL 3.3 context could be created")
        pytest.skip("OpenGL 3.3 display not available (run under xvfb-run on headless Linux)")


@pytest.fixture
def scene(qapp):
    from psim.scene.editing import SceneEditor
    from psim.scene.scene_manager import Scene
    s = Scene()
    return s, SceneEditor(s)
