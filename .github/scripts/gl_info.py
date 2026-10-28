"""Print the OpenGL context the tests will get (version, renderer, vendor) as a GitHub Actions notice.

Exits with status 1 when no OpenGL 3.3 context can be created, so a job meant to run the rendering tests stops
early with a clear message.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PySide6.QtGui import QOffscreenSurface, QOpenGLContext  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from psim.app import configure_opengl  # noqa: E402


def main() -> int:
    configure_opengl()
    app = QApplication.instance() or QApplication([])  # noqa: F841
    ctx = QOpenGLContext()
    surf = QOffscreenSurface()
    surf.create()
    if not ctx.create() or not ctx.makeCurrent(surf):
        print(f"::error::No OpenGL context (Qt platform {QApplication.platformName()})")
        return 1
    # through Qt's resolved GL functions: PyOpenGL's own lookup returned nothing on Windows with Mesa's opengl32.dll
    gl = ctx.functions()
    info = [gl.glGetString(k) for k in (0x1F02, 0x1F01, 0x1F00)]          # GL_VERSION, GL_RENDERER, GL_VENDOR
    text = " | ".join((x.decode(errors="replace") if isinstance(x, bytes) else str(x)) if x else "?" for x in info)
    version = ctx.format().majorVersion() * 10 + ctx.format().minorVersion()
    ctx.doneCurrent()
    print(f"::notice title=OpenGL::{QApplication.platformName()}: {text}")
    if version < 33:
        print(f"::error::OpenGL {version / 10:.1f} context; 3.3 core is needed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
