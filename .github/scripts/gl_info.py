"""Print the OpenGL context the tests will get (version, renderer, vendor) as a GitHub Actions notice.

The context is that of a shown QOpenGLWidget, like the application's viewport. Its strings are read twice:
through Qt's resolved functions and through PyOpenGL, which the renderer uses. Both must name the same
implementation; on Windows, Qt can fall back to its bundled software opengl32sw.dll while PyOpenGL loads
opengl32.dll, and the two would then disagree. Exits with status 1 when there is no OpenGL 3.3 context or the two
views differ, so a job meant to run the rendering tests stops early with a clear message.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from psim.app import configure_opengl  # noqa: E402

STRINGS = (0x1F02, 0x1F01, 0x1F00)          # GL_VERSION, GL_RENDERER, GL_VENDOR


def _text(x) -> str:
    if not x:
        return "?"
    return x.decode(errors="replace") if isinstance(x, bytes) else str(x)


class Probe(QOpenGLWidget):
    result: dict | None = None

    def initializeGL(self):  # noqa: N802
        from OpenGL import GL
        ctx = self.context()
        fmt = ctx.format()
        qt = [_text(ctx.functions().glGetString(k)) for k in STRINGS]
        try:
            py = [_text(GL.glGetString(k)) for k in STRINGS]
        except Exception as exc:  # noqa: BLE001 — reported, not raised
            py = [f"error: {exc}"] * 3
        Probe.result = {"version": fmt.majorVersion() * 10 + fmt.minorVersion(), "qt": qt, "pyopengl": py}


def main() -> int:
    configure_opengl()
    app = QApplication.instance() or QApplication([])
    w = Probe()
    w.resize(64, 64)
    w.show()
    QTest.qWaitForWindowExposed(w)
    w.grabFramebuffer()
    QTest.qWait(50)
    res = Probe.result
    platform = app.platformName()
    w.close()
    if res is None:
        print(f"::error::No OpenGL context (Qt platform {platform})")
        return 1
    qt, py = " | ".join(res["qt"]), " | ".join(res["pyopengl"])
    print(f"::notice title=OpenGL::{platform}: {qt}")
    if res["version"] < 33:
        print(f"::error::OpenGL {res['version'] / 10:.1f} context; 3.3 core is needed")
        return 1
    if res["qt"] != res["pyopengl"]:
        print(f"::error::Qt and PyOpenGL see different OpenGL implementations: Qt {qt}; PyOpenGL {py}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
